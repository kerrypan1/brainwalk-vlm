## Folder Structure

- `models/`
  - Model wrappers that all implement a shared `BaseVLM` interface
  - Includes InternVL and LLaVA (large + small variants)
- `scripts/`
  - Operational scripts for clip generation, inference, and parsing for GT
- `prompts/`
  - Prompt templates selected by model family, ICL mode, and dataset variant
- `analysis.ipynb`
  - Analysis notebook that aggregates evaluation results and computes statistics/plots

## Code Components

### `models/base_vlm.py`

Defines the abstract interface all model wrappers must follow

### `models/internvl.py` and `models/internvl_small.py`

- `InternVL` loads InternVL checkpoints, decodes first N clip frames, applies transforms, and runs chat-style generation
- `InternVLSmall` is a thin subclass that switches to a smaller checkpoint (`InternVL2-1B`) for lower memory use

### `models/videollava.py` and `models/videollava_small.py`

- LLaVA wrappers decode first N frames and feed them into chat-template-style multimodal prompts.
- Large wrapper uses LLaVA-NeXT-Video 7B
- Small wrapper uses OneVision 0.5B
- Both return text output compatible with the shared `BaseVLM` contract

### `scripts/clip.py`

Builds clip datasets from raw videos by:

- computing simple motion scores from frame differences,
- selecting up to 3 high-motion candidate windows,
- cutting fixed-duration clips with ffmpeg,
- writing clips under `clips/clips_fps_<fps>_length_<clip_len>/` (not included in repo)

### `scripts/inference.py`

Runs a chosen model over all clip files in a selected clip set

Inputs:

- `--model` (`intern_l`, `intern_s`, `llava_l`, `llava_s`)
- `--fps`
- `--clip_len`
- `--icl` (`y`, `n`, `generate`)
- `--dataset` (prompt suffix such as `therapist`, `zeno`, etc. Note: I used this mostly for prompt changes, "generate" was used for video descriptions used in ICL, "therapist/zeno" was used to request the correct output values, and "therapist1-5" were used for prompt ablations)

Behavior:

- picks prompt from `prompts/`,
- applies model to each `clip_*.mp4`,
- writes one `.txt` output per clip (not included in repo)

### `scripts/xlsx_to_csv.py`

Converts raw XLSX review data into `gt.csv` (not included in repo)

## Process

1. Generate clips:
   - Run `scripts/clip.py` with desired `--fps` and `--clip_len`
2. Run inference:
   - Run `scripts/inference.py` for each model + prompt setup
3. Prepare labels (once):
   - Run `scripts/xlsx_to_csv.py` to produce/update `gt.csv`
4. Analyze in notebook:
   - Use `analysis.ipynb` for cross-setup comparison, fold averages, bootstrap CIs, and plots

## Notes

- Source Data, Video Clips, VLM Output, and GT Files are not included to protect patient data. All file references resolve correctly.
