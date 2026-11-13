---
name: learndl-models
description: >-
  Develop learndl neural network models, registration and training configurations;
  diagnose and resolve training CUDA OOM, including ActivationCheckpointing recovery.
  Use for src/res/algo/nn and model training/resume integration, not data ingestion,
  factor definitions or watchdog operations.
---

# learndl model development and OOM recovery

## Objective

Build correct models that can train within the available memory budget while preserving the intended modeling semantics and existing-weight compatibility where required.

For an OOM task, success means the affected training workload completes the previously failing operation and the relevant training path within the available GPU memory. Enabling `ActivationCheckpointing` is one possible recovery mechanism, not the objective or proof of success. Distinguish implementation completion, recovery activation, measured memory improvement and successful workload execution. If CUDA or production data is unavailable, report the remaining acceptance work explicitly.

Choose a remedy from the observed cause. Activation recomputation can reduce tensors retained for backward, but does not guarantee that a single operator's workspace, resident data, parameters or optimizer state will fit. Do not automatically enable it for every model, or require trying it before a different remedy when the evidence points elsewhere. Respect the user's chosen scope and staged experiments.

## Entry points and model contracts

- Architectures: `src/res/algo/nn/model/`; registration: `src/res/algo/registry.py`, `src/res/algo/nn/api.py` and model package exports.
- Parameters: `configs/algo/nn/`; schedules: `configs/schedule/`; effective configuration merging: `src/res/model/util/config/config.py`.
- Training entry: `src/res/model/model_module/application/trainer.py`; NN training: `src/res/model/model_module/module/nn.py`.
- Identify the actual class behind the registry key, input axes and dtype, prediction/hidden output contract, losses and cross-sectional normalization. Verify that `model.module` matches `algo.<module>`; similar filenames do not establish model identity.
- Read dimensions from effective configuration, not unused YAML fields. Check constructor defaults when saved configurations omit parameters.
- Memory fixes for resumed models should preserve parameter names, shapes, architecture and strict loading of old weights unless an architectural change is intended. Explain compatibility consequences before changing dimensions to reduce memory.

## Diagnose OOM before selecting a remedy

Read the full log: code revision, model date/index, resume state, effective parameters, Callback list, input shapes, training/validation stage, exception stack and memory statistics. An old failure log is not evidence that a later fix failed.

Reshaping `[stocks, days, bars, features]` to `[stocks*days, bars, features]` increases the sequence batch passed to a GRU. Distinguish a possible reshape copy, GRU workspace/reserve allocation and activations retained for backward. High reserved-but-unallocated memory alone does not establish fragmentation as the sole cause.

If checkpointing was expected, first establish whether recovery was enabled and recomputation actually activated. Do not interpret a failure on the ordinary path as failure of checkpointed training. If the user requests fixing only the enablement path first, complete that fix and its validation, then wait for feedback before introducing chunking or other changes.

When chunking is justified and within scope, split independent sequences and checkpoint individual chunks where appropriate. Preserve sample order, temporal dependencies, full-cross-section BatchNorm and loss computation. Wrapping only the whole loop may leave a high backward recomputation peak. Chunking can change dropout RNG assignment; do not promise bitwise-identical training.

## Integrate ActivationCheckpointing when appropriate

Evaluate pure recomputation regions in the target model. When this mechanism is selected, implement and verify the model, configuration and training-entry integration together. Unsupported models must not be opted in silently.

Read and reuse the existing implementation:

- `src/res/algo/nn/layer/checkpoint.py`: `ActivationCheckpointMixin`.
- `src/res/model/callback/fit/activation_checkpointing.py`: switch, recovery semantics and integration example.
- `src/res/model/callback/consolidate.py`: Callback registration.
- `src/res/model/util/trainer/base_trainer.py`: `fit_current_model` and failed-state cleanup.
- `src/res/model/util/advance/torch_compile.py`: OOM propagation.
- `tests/test_activation_checkpointing.py`: regression coverage and optional CUDA tests.

Inherit `ActivationCheckpointMixin`, declare nonempty callable `activation_checkpoint_regions`, and invoke them through `checkpoint_region` in forward. Exclude BatchNorm running-stat updates, cache writes, counters and other side effects. Reuse `use_reentrant=False` and `preserve_rng_state=True`; eval/no-grad bypass recomputation. Declaring regions without calling the helper is insufficient.

Opt in explicitly in the target schedule:

```yaml
callbacks:
  ActivationCheckpointing:
    enabled: true
```

Report evidence for each state separately:

1. **Model support:** valid region declarations and actual forward calls.
2. **Recovery enabled:** effective callback kwargs are true and the actual Callback list contains the class.
3. **Recomputation active:** the existing policy starts with ordinary training. The first CUDA OOM inside fit discards all attempts of the current model_date/model_num and restarts at trial 0 / epoch 0 with the network switch enabled.
4. **CUDA acceptance:** the representative workload completes, with measured full forward/backward peaks and the relevant training stages verified.

Subsequent models and quality retries in the same trainer retain activation; a second CUDA OOM propagates. A new process starts in ordinary mode again. Previously completed models survive. This is distinct from restoring a failed epoch's disk checkpoint. Preserve this bounded recovery policy unless explicitly changing it is part of the task.

## Verify the effective configuration path

A YAML assertion that `enabled: true` exists, or Mock callback kwargs alone, does not verify resumed training.

- With base_path, `ScheduleConfig.find_path` normally reads the model's saved schedule; `BaseModelConfig` may also load saved model.yaml. Trace the actual schedule_name, base_path, resume and override arguments.
- Current code refreshes only `callbacks.ActivationCheckpointing` from the current schedule for NN resumes with an explicit schedule_name. Other saved settings remain in force; explicit runtime overrides take precedence. Model-path-only resumes do not automatically refresh the current schedule. Verify these rules against the source at hand.
- `FlattenDict.update` defaults to existing keys only. Missing legacy Callback keys require a narrowly scoped `update(..., relevant_only=False)`; direct assignment to a missing key also fails. Do not replace all saved settings to migrate one key.
- Construct a real `ModelConfig` and `ConsolidateCallBack` with temporary legacy model.yaml/schedule.yaml files. Cover a missing key, a legacy Callback list, current explicit true/false, runtime override precedence, model-path-only resume and no explicit opt-in.
- Assert that model hyperparameters and other saved settings remain intact and old weights still load.
- Server startup should expose effective configuration provenance and `ActivationCheckpointing: enabled=True`; the first OOM should log the full restart and activation. Pulling code does not prove configuration took effect.

## Validation and delivery

- For recomputation changes, compare ordinary/checkpointed outputs, parameter gradients and buffers using identical weights and RNG with dropout enabled; cover eval/no-grad and strict loading from the old structure or old weights. CUDA/cuDNN behavior requires separate acceptance; CPU equivalence does not establish it.
- Reuse and extend OOM injection tests as needed: current-model cleanup, preservation of completed models, persistent activation for later models and stopping on the second failure.
- For checkpointing work, run `.venv/bin/python -m unittest discover -s tests -p test_activation_checkpointing.py`, adding coverage for the target model. Run project Ruff and Pyright with `.venv/bin/python`, including changed tests explicitly.
- On CUDA, measure full forward/backward peaks with the failed batch's actual shape and effective parameters; include optimizer steps when evaluating the total training peak. `AC_TEST_CUDA_BATCH_SIZE` changes stock count only; verify the remaining dimensions. A small synthetic batch is not production acceptance.
- Without CUDA, report: “CPU regressions passed; server memory acceptance remains pending.” Do not claim the OOM is resolved. Specify the startup/restart evidence and workload still needed.
- Git pushes, worklist changes and server training must remain within the user's authorization. Use `learndl-automation` for scheduling work. This skill does not itself authorize training or runtime-ledger mutations.

## Reusable prompt

> Use $learndl-models to diagnose and resolve the specified model's training CUDA OOM while preserving modeling semantics and old-weight compatibility. Evaluate ActivationCheckpointing as one recovery mechanism; if selected, integrate pure regions and verify effective configuration, Callback registration and activation after OOM for both fresh and legacy-resume paths. Test outputs, gradients, weight loading and recovery lifecycle. Validate the previously failing workload and memory peak on CUDA, or explicitly report pending acceptance. Respect the requested experiment scope; do not treat enabling the mechanism as resolving the OOM or introduce unrequested changes to chunking, precision, dimensions or server execution.

The versioned source is `.cursor/skills/learndl-models/`; the local Codex discovery copy is `.agents/skills/learndl-models/`. Synchronize them after edits.

## Research artifact delivery

When adding or changing research result outputs, use
[learndl-artifact-exchange](../learndl-artifact-exchange/SKILL.md) for batch identity,
completed snapshots, Syncthing publication and experiment-panel archival.
