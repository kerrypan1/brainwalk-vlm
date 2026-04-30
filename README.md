# Brainwalk VLM Pipeline

This folder contains an end-to-end workflow for:

1. Creating short gait video clips from source videos.
2. Running multiple video-language models (VLMs) on those clips.
3. Saving per-clip model text outputs.
4. Converting source labels into a canonical ground-truth CSV.
5. Evaluating prediction error against ground truth.
6. Exploring aggregate metrics and bootstrap confidence intervals in the notebook.

## Folder Structure

- `models/`
  - Model wrappers that all implement a shared `BaseVLM` interface.
  - Includes InternVL and LLaVA (large + small variants).
- `scripts/`
  - Operational scripts for clip generation, inference, label conversion, and evaluation.
- `prompts/`
  - Prompt templates selected by model family, ICL mode, and dataset variant.
- `clips/`
  - Generated clip sets organized by `clips_fps_<fps>_length_<clip_len>`.
- `vlm_output/`
  - Model outputs organized by model + ICL mode + dataset + clip set.
- `analysis.ipynb`
  - Analysis notebook that aggregates evaluation results and computes statistics/plots.
- `gt.csv`
  - Canonical ground-truth table used by scripts and notebook analysis.

## Code Components

### `models/base_vlm.py`

Defines the abstract interface all model wrappers must follow:

- `run(video_path, prompt)` -> `(text_output, embedding_or_none)`

This allows the inference script to swap models without changing loop logic.

### `models/internvl.py` and `models/internvl_small.py`

- `InternVL` loads InternVL checkpoints, decodes first N clip frames, applies transforms, and runs chat-style generation.
- `InternVLSmall` is a thin subclass that switches to a smaller checkpoint (`InternVL2-1B`) for lower memory use.

### `models/videollava.py` and `models/videollava_small.py`

- LLaVA wrappers decode first N frames and feed them into chat-template-style multimodal prompts.
- Large wrapper uses LLaVA-NeXT-Video 7B.
- Small wrapper uses OneVision 0.5B.
- Both return text output compatible with the shared `BaseVLM` contract.

### `scripts/clip.py`

Builds clip datasets from raw videos by:

- computing simple motion scores from frame differences,
- selecting up to 3 high-motion candidate windows,
- cutting fixed-duration clips with ffmpeg,
- writing clips under `clips/clips_fps_<fps>_length_<clip_len>/`.

### `scripts/inference.py`

Runs a chosen model over all clip files in a selected clip set.

Inputs:

- `--model` (`intern_l`, `intern_s`, `llava_l`, `llava_s`)
- `--fps`
- `--clip_len`
- `--icl` (`y`, `n`, `generate`)
- `--dataset` (prompt suffix such as `therapist`, `zeno`, etc.)

Behavior:

- picks prompt from `prompts/`,
- applies model to each `clip_*.mp4`,
- writes one `.txt` output per clip,
- skips existing files to support resumable runs.

### `scripts/xlsx_to_csv.py`

Converts raw XLSX review data into canonical `gt.csv`:

- cleans IDs and numeric values,
- maps metric columns,
- emits long-format rows (`id` as `<base>_1` and `<base>_2`).

### `scripts/evaluate.py`

Parses model outputs, averages per-sample clip predictions, and compares against `gt.csv`:

- computes per-field MAE,
- computes overall MAE,
- reports missing/overlapping IDs and skipped folders.

## Typical Workflow

1. Generate clips:
   - Run `scripts/clip.py` with desired `--fps` and `--clip_len`.
2. Run inference:
   - Run `scripts/inference.py` for each model + prompt setup.
3. Prepare labels (if needed):
   - Run `scripts/xlsx_to_csv.py` to produce/update `gt.csv`.
4. Evaluate outputs:
   - Run `scripts/evaluate.py` (or notebook equivalents) for error metrics.
5. Analyze in notebook:
   - Use `analysis.ipynb` for cross-setup comparison, fold averages, bootstrap CIs, and plots.

## Output Naming Conventions

- Clip sets: `clips_fps_<fps>_length_<clip_len>`
- Inference outputs:
  - `<model>_<icl_tag>_<dataset>/clips_fps_<fps>_length_<clip_len>/<sample_id>/clip_k.txt`
- Common model tags:
  - `intern_l`, `intern_s`, `llava_l`, `llava_s`
- Common ICL tags:
  - `icl`, `noicl`, `generate`

## Notes

- Script paths are mostly resolved relative to the repository layout to reduce cwd-related issues.
- Model wrappers currently return text output and `None` for embedding output.
- Existing output folders can be large; keep only required artifacts under version control.
