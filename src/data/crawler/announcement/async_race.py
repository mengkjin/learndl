"""
Async scheduler for normal mode + race mode over a shared proxy pool.
"""
import asyncio
from typing import Any
from src.proj.util.proxy.ppool import AsyncAdaptiveProxyPool, ProxyStats

from .fetcher import FetcherTask
from .util import Announcement, crawler_log

class AsyncProxyRaceExecutor:
    """Async scheduler for normal mode + race mode over a shared proxy pool."""
    def __init__(
        self,
        pool : AsyncAdaptiveProxyPool,
        *,
        race_ratio: float = 0.5,
        min_race_tasks: int = 2,
        max_replicas_per_task: int = 5,
        max_total_inflight_per_exchange: int = 20,
    ):
        self.pool = pool
        self.race_ratio = race_ratio
        self.min_race_tasks = min_race_tasks
        self.max_replicas_per_task = max_replicas_per_task
        self.max_total_inflight_per_exchange = max_total_inflight_per_exchange

    async def _acquire_proxy(self, url: str):
        return await self.pool.acquire_async(url)

    async def _release_proxy(self, proxy, success: bool, *, counted: bool = True):
        await self.pool.release_async(proxy, success, counted=counted)

    def _is_success(self, value: Any) -> bool:
        return isinstance(value, list)

    async def _run_replica(self, task: FetcherTask, proxy_stats: ProxyStats, attempt_id: str):
        proxy = proxy_stats.url
        task_key = f"{task.start}_{task.end}_{task.exchange}"
        try:
            value = await task.fetch_payload_async(proxy, attempt_id=attempt_id)
            # Save temp payload only for successful download attempts.
            if isinstance(value, list):
                task.exporter.save_temp_attempt(task_key, attempt_id, value)
                crawler_log(f"[race-attempt] saved success attempt={attempt_id} task={task.title} rows={len(value)}")
            else:
                crawler_log(f"[race-attempt] no temp saved attempt={attempt_id} task={task.title} type={type(value)}")
            if self._is_success(value):
                await self._release_proxy(proxy_stats, True, counted=True)
                return {"ok": True, "payload": value, "proxy": proxy, "attempt_id": attempt_id}
            if isinstance(value, Exception):
                await self._release_proxy(proxy_stats, False, counted=True)
                return {"ok": False, "error": value, "proxy": proxy, "attempt_id": attempt_id}
            await self._release_proxy(proxy_stats, False, counted=True)
            return {"ok": False, "error": ValueError(f"Unexpected return type: {type(value)}"), "proxy": proxy, "attempt_id": attempt_id}
        except asyncio.CancelledError:
            await self._release_proxy(proxy_stats, False, counted=False)
            raise
        except Exception as e:
            await self._release_proxy(proxy_stats, False, counted=True)
            return {"ok": False, "error": e, "proxy": proxy, "attempt_id": attempt_id}

    async def _start_replica(self, task: FetcherTask, inflight_by_task: dict[str, set[asyncio.Task]], attempt_idx: dict[str, int]):
        proxy_stats = await self._acquire_proxy(task.url)
        if proxy_stats is None:
            return None
        attempt_idx[task.title] += 1
        attempt_id = f"{task.title}#{attempt_idx[task.title]}"
        fut = asyncio.create_task(self._run_replica(task, proxy_stats, attempt_id))
        inflight_by_task[task.title].add(fut)
        return fut

    async def run_exchange_tasks(self, tasks: list[FetcherTask], *, workers: int = 10) -> dict[str, Any]:
        if not tasks:
            return {"ok": True, "results": {}, "errors": {}}
        pending = [task for task in tasks]
        if not pending:
            return {"ok": True, "results": {}, "errors": {}}
        inflight_by_task: dict[str, set[asyncio.Task]] = {task.title: set() for task in pending}
        task_map = {task.title: task for task in pending}
        results: dict[str, list[Announcement]] = {}
        errors: dict[str, Exception] = {}
        winner_attempt: dict[str, str] = {}
        done_titles: set[str] = set()
        attempt_idx: dict[str, int] = {task.title: 0 for task in pending}
        max_workers = max(1, workers)
        crawler_log(f"[race-run-exchange-tasks] max_workers={max_workers} max_total_inflight_per_exchange={self.max_total_inflight_per_exchange}", type='stdout')

        def remaining_titles() -> list[str]:
            return [task.title for task in pending if task.title not in done_titles]

        async def fill_slots():
            nonlocal pending
            total_inflight = sum(len(v) for v in inflight_by_task.values())
            cap = min(max_workers, self.max_total_inflight_per_exchange)
            crawler_log(f"[race-fill-slots] total_inflight={total_inflight} cap={cap}", type='stdout')
            while total_inflight < cap:
                remaining = remaining_titles()
                if not remaining:
                    break
                valid_proxy_count = self.pool.proxies[task_map[remaining[0]].url].valid_count
                threshold = max(self.min_race_tasks, int(valid_proxy_count * self.race_ratio))
                race_mode = len(remaining) < threshold
                candidates = remaining if race_mode else [remaining[0]]
                started_any = False
                for title in candidates:
                    if total_inflight >= cap:
                        break
                    current = len(inflight_by_task[title])
                    allow = self.max_replicas_per_task if race_mode else 1
                    if current >= allow:
                        continue
                    fut = await self._start_replica(task_map[title], inflight_by_task, attempt_idx)
                    if fut is None:
                        continue
                    started_any = True
                    total_inflight += 1
                if not started_any:
                    break

        while True:
            await fill_slots()
            all_inflight = {f for futs in inflight_by_task.values() for f in futs}
            if not all_inflight:
                break
            done, _ = await asyncio.wait(all_inflight, return_when=asyncio.FIRST_COMPLETED)
            for fut in done:
                title = next((k for k, v in inflight_by_task.items() if fut in v), None)
                if title is None:
                    continue
                inflight_by_task[title].discard(fut)
                if title in done_titles:
                    # consume completed duplicate task to avoid dropped exceptions and stale states
                    await asyncio.gather(fut, return_exceptions=True)
                    continue
                ret = await fut
                if ret.get("ok"):
                    done_titles.add(title)
                    results[title] = ret["payload"]
                    winner_attempt[title] = ret.get("attempt_id", "")
                    # cancel same-task replicas and don't count these cancels as success/failure
                    cancel_count = 0
                    for dup in list(inflight_by_task[title]):
                        dup.cancel()
                        cancel_count += 1
                    if inflight_by_task[title]:
                        await asyncio.gather(*inflight_by_task[title], return_exceptions=True)
                    inflight_by_task[title].clear()
                    task = task_map[title]
                    task_key = f"{task.start}_{task.end}_{task.exchange}"
                    task.exporter.cleanup_temp_attempts(task_key, keep_attempt_id=winner_attempt[title])
                    crawler_log(
                        f"[race-winner] task={title} winner={winner_attempt[title]} "
                        f"rows={len(ret['payload']) if isinstance(ret.get('payload'), list) else 'NA'} "
                        f"cancelled_replicas={cancel_count}" ,
                        type='success'
                    )
                elif title not in errors:
                    errors[title] = ret.get("error") or RuntimeError("Unknown fetch error")
                    # cleanup failed attempts if no winner yet
                    if len(inflight_by_task[title]) == 0:
                        task = task_map[title]
                        task_key = f"{task.start}_{task.end}_{task.exchange}"
                        task.exporter.cleanup_temp_attempts(task_key, keep_attempt_id=None)
                        crawler_log(f"[race-failed] task={title} exhausted without winner" , type='alert')
        ok = len(done_titles) == len(pending)
        crawler_log(f"[race-summary] task={title} total_tasks={len(pending)} success={len(done_titles)} failed={len(errors)}",type='note')
        return {"ok": ok, "results": results, "errors": errors, "winner_attempt": winner_attempt}
