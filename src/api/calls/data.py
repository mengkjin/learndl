"""
Direct calls related to data operations of this project.
"""

from __future__ import annotations

from typing import Sequence

from src.proj import Logger , Proj , MACHINE , Base
from src.api.util.direct_call import DirectCall

__all__ = ['ReconstructPreprocessedData' , 'RecalculateHistNorm' , 'UpdateProjectData']


class ReconstructPreprocessedData(DirectCall):
    """Reconstruct the preprocessed data."""
    category = 'Data'
    def run(self) -> None:
        from src.proj.util.cli import AskFor
        from src.data.preprocess.processors import PrePros

        data_keys = PrePros.keys()
        if not data_keys:
            Logger.note('No data keys found for preprocess.')
            return

        with Proj.vb.temporary_vb('max'):
            for loop in AskFor.LoopTillExit(message = f'Do you want to reconstruct more data?'):
                flag_key = AskFor.Options(
                    data_keys , confirm = False , multiple = False , allow_back = False ,
                    title = f'Which data preprocessor to reconstruct?',
                    help_description='Pick a registered PrePros processor key; rebuilds feather/parquet under data/.',
                )
                if not loop.set_flag(flag_key) or flag_key.result is None:
                    continue
                
                Logger.note(f'Select [{flag_key.result}] data to reconstruct...')
                flag_type = AskFor.Options(
                    ['fit' , 'predict' , 'both'] , confirm = False , multiple = False ,
                    title = f'Which type of data to reconstruct? (fit/predict/both)',
                    help_description='fit: training samples; predict: inference features; both: rebuild both frames.',
                    option_help={
                        'fit': 'Rebuild fit frame used for model training.',
                        'predict': 'Rebuild predict frame used at inference time.',
                        'both': 'Rebuild fit then predict (fit skipped on coding machines).',
                    },
                )
                if not loop.set_flag(flag_type) or flag_type.result is None:
                    continue
                data_type = flag_type.result
                if MACHINE.platform_coding and flag_type.result != 'predict':
                    Logger.alert1('This is a coding machine, fit data reconstruction costs a lot of time.')
                if flag_type.result == 'fit':
                    PrePros.get_processor(flag_key.result, frame = 'fit').build(reconstruct = True)
                elif flag_type.result == 'predict':
                    PrePros.get_processor(flag_key.result, frame = 'predict').build(reconstruct = True)
                elif flag_type.result == 'both':
                    PrePros.get_processor(flag_key.result, frame = 'fit').build(reconstruct = True)
                    PrePros.get_processor(flag_key.result, frame = 'predict').build(reconstruct = True , confirm = False)
                else:
                    raise ValueError(f'Invalid data type: {data_type}')

class RecalculateHistNorm(DirectCall):
    """Recalculate the historical normalisation statistics."""
    category = 'Data'
    def run(self) -> None:
        from src.data.preprocess.hist_norm import PreProHistNorm
        PreProHistNorm.recalculate_all()


class UpdateProjectData(DirectCall):
    """Full or partial project data update with optional forced date range."""

    category = 'Data'

    @classmethod
    def get_description(cls , **kwargs) -> str:
        return (
            'Update project data: full pipeline (DataAPI.update) or selective subtrees '
            '(Core/Sellside/Risk/Custom/Announcement) with optional force start/end.'
        )

    def run(self) -> None:
        from src.proj.util.cli import AskFor

        for loop in AskFor.LoopTillExit(message='Run another data update?'):
            flag_mode = AskFor.Options(
                ['Full Update' , 'Partial Update'],
                confirm=False,
                multiple=False,
                allow_back=False,
                title='Data update mode',
                help_description=(
                    'Full Update runs DataAPI.update (core → sellside → risk → custom → announcement). '
                    'Partial Update drills into a selection tree, then asks whether to force a date range.'
                ),
                option_help={
                    'Full Update': 'Existing incremental DataAPI.update; no force prompt.',
                    'Partial Update': 'Pick modules/items, then optional force with start/end (empty = default incremental).',
                },
            )
            if not loop.set_flag(flag_mode) or flag_mode.result is None:
                continue

            if flag_mode.result == 'Full Update':
                from src.api.pkgs.data import DataAPI
                Logger.note('Running full DataAPI.update ...')
                DataAPI.update()
                continue

            selection = self._walk_selection_tree(self._root_tree())
            if selection is None:
                continue
            if not selection:
                Logger.alert1('No items selected.')
                continue

            force , start , end = self._ask_force_range()
            if force is None:
                continue

            self._dispatch_partial(selection , force=force , start=start , end=end)

    @classmethod
    def _root_tree(cls) -> Base.UpdateMenuNode:
        from src.data import CoreDataUpdater , SellsideDataUpdater , CustomDataUpdater
        from src.data.crawler import AnnouncementAgent
        from src.res.factor.api import RiskModelUpdater

        return Base.UpdateMenuNode(
            label='Partial Update',
            children=[
                CoreDataUpdater.selection_tree(),
                SellsideDataUpdater.selection_tree(),
                RiskModelUpdater.selection_tree(),
                CustomDataUpdater.selection_tree(),
                AnnouncementAgent.selection_tree(),
            ],
        )

    @classmethod
    def _walk_selection_tree(cls , root: Base.UpdateMenuNode) -> list[str] | None:
        """Walk nested menus; return leaf keys or None if user backs out at top."""
        from src.proj.util.cli import AskFor

        node = root
        stack: list[Base.UpdateMenuNode] = []
        while True:
            if node.is_leaf:
                return [node.key] if node.key is not None else []

            labels = [c.label for c in node.children if not c.disabled]
            if not labels:
                Logger.alert1(f'No selectable children under [{node.label}].')
                return [] if stack else None

            option_help = {
                c.label: (c.help or ('Select all leaves under this branch.' if c.label == Base.ALL_UNDER_LABEL else c.label))
                for c in node.children
                if not c.disabled
            }
            # Intermediate: single choice (incl. All). Leaf level: multi-select.
            children_are_leaves = all(c.is_leaf or c.label == Base.ALL_UNDER_LABEL for c in node.children)
            if children_are_leaves:
                # Offer multi-select among leaves; keep All as expanding all.
                leaf_labels = [c.label for c in node.children if c.is_leaf and not c.disabled]
                all_child = node.child_by_label(Base.ALL_UNDER_LABEL)
                choices = ([Base.ALL_UNDER_LABEL] if all_child is not None else []) + leaf_labels
                flag = AskFor.Options(
                    choices,
                    confirm=False,
                    multiple=True,
                    allow_back=bool(stack),
                    title=f'{node.label} — select items',
                    help_description='Multi-select leaf items, or * All under this. Back (q) returns to parent.',
                    option_help=option_help,
                )
                if not flag.valid:
                    if not stack:
                        return None
                    node = stack.pop()
                    continue
                selected_labels = list(flag.results)
                if Base.ALL_UNDER_LABEL in selected_labels:
                    return node.all_leaf_keys()
                keys: list[str] = []
                for label in selected_labels:
                    child = node.child_by_label(label)
                    if child is not None and child.key is not None:
                        keys.append(child.key)
                return keys

            # Non-leaf children: drill one level (with All shortcut).
            drill_labels = [Base.ALL_UNDER_LABEL] + labels
            flag = AskFor.Options(
                drill_labels,
                confirm=False,
                multiple=False,
                allow_back=bool(stack),
                title=f'{node.label} — choose a branch',
                help_description='Pick a subdirectory to drill into, or * All under this. Back (q) returns to parent.',
                option_help={
                    Base.ALL_UNDER_LABEL: 'Select every enabled leaf under this branch.',
                    **option_help,
                },
            )
            if not flag.valid or flag.result is None:
                if not stack:
                    return None
                node = stack.pop()
                continue
            if flag.result == Base.ALL_UNDER_LABEL:
                return node.all_leaf_keys()
            child = node.child_by_label(flag.result)
            if child is None:
                Logger.alert1(f'Unknown branch: {flag.result}')
                continue
            stack.append(node)
            node = child

    @classmethod
    def _ask_force_range(cls) -> tuple[bool | None , int | None , int | None]:
        """Return (force, start, end). force is None if user cancelled."""
        from src.proj.util.cli import AskFor

        flag_force = AskFor.Options(
            ['No (default incremental)' , 'Yes (force date range)'],
            confirm=False,
            multiple=False,
            allow_back=True,
            title='Force update selected items?',
            help_description=(
                'Default: incremental update for the selection. '
                'Force: prompt start/end; leave both empty to fall back to default incremental.'
            ),
        )
        if not flag_force.valid or flag_force.result is None:
            return None , None , None
        if flag_force.result.startswith('No'):
            return False , None , None

        flag_start = AskFor.String(
            title='Force start date (YYYYMMDD), empty = None',
            help_description='Leave empty for None. If both start and end are empty, runs default incremental update.',
        )
        if not flag_start.valid:
            return None , None , None
        flag_end = AskFor.String(
            title='Force end date (YYYYMMDD), empty = None',
            help_description='Leave empty for None. If both start and end are empty, runs default incremental update.',
        )
        if not flag_end.valid:
            return None , None , None

        start = cls._parse_optional_date(flag_start.result)
        end = cls._parse_optional_date(flag_end.result)
        force , start , end = Base.resolve_force_range(True , start , end)
        if not force:
            Logger.note('Force selected but start/end empty — running default incremental update.')
        else:
            Logger.note(f'Force update for range [{start} , {end}].')
        return force , start , end

    @staticmethod
    def _parse_optional_date(value: str | None) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(text)

    @classmethod
    def _dispatch_partial(
        cls,
        selection: Sequence[str],
        * ,
        force: bool,
        start: int | None,
        end: int | None,
    ) -> None:
        from src.data import CoreDataUpdater , SellsideDataUpdater , CustomDataUpdater
        from src.data.crawler import AnnouncementAgent
        from src.res.factor.api import RiskModelUpdater

        groups: dict[str , list[str]] = {
            'core': [] ,
            'sellside': [] ,
            'risk': [] ,
            'custom': [] ,
            'announcement': [] ,
        }
        for key in selection:
            prefix = key.split('.' , 1)[0]
            if prefix not in groups:
                raise ValueError(f'Unknown selection key prefix: {key}')
            groups[prefix].append(key)

        if groups['core']:
            CoreDataUpdater.selective_update(groups['core'] , force=force , start=start , end=end)
        if groups['sellside']:
            SellsideDataUpdater.selective_update(groups['sellside'] , force=force , start=start , end=end)
        if groups['risk']:
            RiskModelUpdater.selective_update(groups['risk'] , force=force , start=start , end=end)
        if groups['custom']:
            CustomDataUpdater.selective_update(groups['custom'] , force=force , start=start , end=end)
        if groups['announcement']:
            AnnouncementAgent.selective_update(groups['announcement'] , force=force , start=start , end=end)
