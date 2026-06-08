# VoxTell-Style 3D Prompt Student Migration

## Decision

Stop the 2D conversion route. The next student design should keep 3D CT volumes as
3D inputs and use a prompt-based 3D segmentation architecture.

VISTA3D is no longer the center of the student design. It remains one teacher or
reference component in the teacher pool, but its 127-label space must not define
the system target space.

The target label space for the student should be the teacher-covered global
organ space, not the VISTA3D 127 labels. The current accepted exact 3D prompt
target count is 373 organs: 384 global organs minus 8 SAROS coarse-label
unresolvable organs and 3 no-enabled-route organs.

## VoxTell Takeaways

Reference:
`third_party/VoxTell`

Paper:
`https://arxiv.org/abs/2511.11450`

VoxTell is directly aligned with the new direction:

- It consumes 3D NIfTI volumes, not 2D slices.
- It accepts free-text prompts such as organ names or clinical descriptions.
- It predicts one 3D binary mask per prompt.
- It supports multiple prompts in one call by producing a `(num_prompts, X, Y, Z)` output.
- It uses a 3D encoder and decoder, with text-image fusion inside the 3D model.

The important architecture pieces in the released code are:

- `voxtell/model/voxtell_model.py`: `VoxTellModel`, the 3D promptable student model.
- `VoxTellModel.encoder`: 3D residual encoder from `dynamic-network-architectures`.
- `project_text_embed`: projects prompt embeddings into the model query space.
- `transformer_decoder`: prompt decoder that lets text query features attend to 3D image features.
- `VoxTellDecoder`: 3D decoder with multi-stage mask embedding fusion.
- `voxtell/inference/predictor.py`: sliding-window 3D inference and text prompt embedding.
- `voxtell/utils/text_embedding.py`: wraps prompt text with an instruction before Qwen embedding.

The text encoder is frozen Qwen3-Embedding-4B in the official implementation.
The prompt encoder therefore does not require a fixed label-id vocabulary in the
way VISTA3D does.

## Legacy Student Problem

The old student code path was VISTA3D-centered:

- `agent-harness/cli_anything/medai/core/vista3d_student.py` maps text prompts to VISTA3D label IDs.
- `agent-harness/cli_anything/medai/core/teacher_branch_map.py` has a VISTA3D label map and converts prompt text to label IDs.
- legacy helper scripts used to save student predictions for "127 organs".

This makes the student label space depend on VISTA3D. That is the part to
avoid in the mainline. The current default route in `scripts/run_em_training.py`
is `voxtell_style_3d_prompt`, and `scripts/run_student_infer_then_round2.py`
now uses the VoxTell-style 3D prompt student. Older VISTA3D repair helpers are
guarded behind `MEDAI_ALLOW_VISTA3D_LEGACY=1` for historical reproduction only.

## Target Design

Use a VoxTell-style student:

```text
3D CT volume
  -> 3D image encoder
organ text prompts from global organ space
  -> frozen text embedding model
image features + text query embeddings
  -> prompt decoder / multi-stage fusion
  -> one 3D mask per prompt
```

The prompt list should come from the global teacher-routed organ space instead
of VISTA3D label IDs.

The model should learn from the current teacher merge outputs:

- `outputs/<case>/segmentations/<organ>.nii.gz`
- `outputs/<case>/unified_labels.nii.gz`
- `outputs/<case>/merge_report.json`

Policy-skipped SAROS organs should stay skipped until an explicit approximation
policy is approved.

Current policy: missing or non-one-to-one classes are skipped directly. No
coarse-label approximation or pseudo-label fabrication is allowed for the 8
SAROS organs.

## Migration Plan

1. Add a new student wrapper instead of extending `VISTA3DStudent`.

   Proposed file:
   `agent-harness/cli_anything/medai/core/voxtell_student.py`

   Initial responsibilities:

   - Load a VoxTell model directory.
   - Build text prompts from global organ names.
   - Run 3D sliding-window inference.
   - Save one NIfTI mask per organ.
   - Support dry-run and command/report JSON output.

2. Add a student target-space config.

   Proposed file:
   `configs/student_3d_prompt_target_organs.json`

   It should derive from global routing/merge state, not from VISTA3D labels.
   The current accepted count is 373 exact prompt targets. It must explicitly
   record excluded/skipped classes, including the 8 SAROS coarse-label skipped
   organs and 3 no-enabled-route organs.

3. Replace VISTA3D M-step dataset building.

   Legacy:
   `build_vista3d_dataset()`

   New default:
   `build_3d_prompt_student_dataset()`

   Dataset entries should point to:

   - CT image path
   - per-organ binary masks or a combined label map
   - prompt text for each organ
   - organ ID in the global student target space

4. Replace M-step training entry.

   Legacy:
   `run_mstep()` -> `VISTA3DStudent.continual_finetune()`

   New default:
   `run_prompt_student_mstep()` -> VoxTell-style student training.

   The official VoxTell repository currently exposes inference but does not
   provide a ready fine-tuning script in this release. So the practical path is:

   - Reuse `VoxTellModel` architecture.
   - Implement training around teacher pseudo-label masks.
   - Use binary losses per prompted organ, not a VISTA3D label-id mapping.
   - Batch prompts per case to control memory.

5. Replace student inference.

   Legacy:
   `save_round_predictions()` calls `VISTA3DStudent.segment()` over 127 organs.

   New default:
   `save_prompt_student_predictions()` calls the prompt student over global
   target prompts and writes one mask per organ.

6. Keep VISTA3D as a teacher/reference model only.

   Keep:

   - `scripts/vista3d_predict_and_split.py`
   - `vista3d` registry entry
   - VISTA3D as one candidate teacher

   Stop using:

   - VISTA3D label IDs as the student target class space
   - `teacher_branch_map.yaml` as the student label-space authority

## Risks and Implementation Notes

- VoxTell v1.1 requires a model directory with `plans.json` and `fold_0/checkpoint_final.pth`.
- Official text embeddings use Qwen3-Embedding-4B and may be memory-heavy.
- VoxTell inference warns that input images must be in RAS orientation.
- The official release does not yet include custom fine-tuning scripts, so we
  need to implement the training loop while reusing the released architecture.
- Prompt batching is important. Running all 373 prompts at once may be too
  memory-heavy.
- Multi-label overlap handling should remain teacher-merge controlled. VoxTell
  itself predicts binary masks per prompt; combined label maps should be built
  by our merge policy.

## Immediate Next Steps

1. Run the project-specific VoxTell fine-tuning loop around `VoxTellModel`:

   ```bash
   export MEDAI_VOXTELL_MODEL_DIR=/home/teacher1/JHU-project1/medical_agent/checkpoints/VoxTell/voxtell_v1.1
   export MEDAI_TEXT_ENCODING_MODEL=/home/teacher1/JHU-project1/medical_agent/checkpoints/Qwen/Qwen3-Embedding-4B
   export MEDAI_VOXTELL_TRAIN_CMD='python scripts/train_voxtell_prompt_student.py'
   ```

2. Use `scripts/train_voxtell_prompt_student.py --dry-run` before spending GPU
   time.
3. Keep legacy VISTA3D student code available for reproducibility, but stop
   presenting it as the main method.
