import argparse
import cv2
import numpy as np
import subprocess
from pathlib import Path

# File Summary
# ------------
# Creates fixed-length clips from raw gait videos.
# For each input video, it:
# 1) computes simple frame-difference motion scores,
# 2) picks up to 3 start times (prefer high-motion, non-overlapping windows),
# 3) uses ffmpeg to cut clips and enforce output FPS.
#
# Outputs are written under:
#   brainwalk-vlm/clips/clips_fps_<fps>_length_<clip_len>/<video_id>/clip_i.mp4
#
# Paths resolved from repository structure, independent of invocation cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
NEW_DIR = SCRIPT_DIR.parent
REPO_ROOT = NEW_DIR.parent
INPUT_DIR = REPO_ROOT / "data" / "bath_pws"  # folder with many .mp4
OUTPUT_DIR = NEW_DIR / "clips"               # where to create clips_fps_* folders

def motion_scores(video_path, sample_fps):
    """Return per-sampled-frame times + motion intensity scores for one video."""
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Sample approximately `sample_fps` by stepping every N source frames.
    stride = max(1, int(src_fps // sample_fps))

    prev = None
    motions = []
    times = []
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_i % stride != 0:
            frame_i += 1
            continue

        # Downsample spatially for faster diff computation.
        gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)

        if prev is None:
            motion = 0.0
        else:
            diff = cv2.absdiff(gray, prev)
            motion = float((diff > 15).mean())

        motions.append(motion)
        times.append(frame_i / src_fps)

        prev = gray
        frame_i += 1

    cap.release()

    # Mark extreme spikes as likely noise/transitions and zero them out.
    max_m = max(motions) if motions else 0.0
    HIGH = 0.35 * max_m

    scores = []
    for m in motions:
        if m > HIGH:
            scores.append(0.0)
        else:
            scores.append(m)

    return np.array(times, dtype=np.float32), np.array(scores, dtype=np.float32)

def pick_top3_no_overlap_else_allow(times, scores, clip_len, skip_front_ratio=0.15):
    """
    Pick up to 3 candidate clip starts.
    Preference:
    - avoid first portion of video,
    - maximize summed motion within the clip window,
    - enforce non-overlap when possible.
    """
    if len(times) == 0:
        return []

    t_end = float(times[-1])
    if t_end <= clip_len:
        return [0.0]

    t_min = skip_front_ratio * t_end

    candidates = []
    for t0 in times:
        t0 = float(t0)
        if t0 < t_min or t0 + clip_len > t_end:
            continue
        mask = (times >= t0) & (times < (t0 + clip_len))
        candidates.append((float(scores[mask].sum()), t0))

    if not candidates:
        return [max(0.0, (t_end - clip_len) / 2.0)]

    candidates.sort(reverse=True)

    picked = []
    for score, t0 in candidates:
        # First pass: enforce non-overlap constraint.
        if all(abs(t0 - s) >= clip_len for s in picked):
            picked.append(t0)
        if len(picked) == 3:
            return sorted(picked)

    # Second pass: if not enough non-overlapping candidates, allow overlap.
    for score, t0 in candidates:
        if t0 not in picked:
            picked.append(t0)
        if len(picked) == 3:
            break

    return sorted(picked)

def cut_clip(video, t0, length, out_path, fps):
    """Use ffmpeg to cut one clip at start `t0` with fixed duration + output fps."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t0:.3f}",
        "-i", str(video),
        "-t", f"{length:.3f}",
        "-vf", f"fps={fps}",
        "-loglevel", "error",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--clip_len", type=float, required=True)
    args = p.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.clip_len <= 0:
        raise ValueError("--clip_len must be > 0")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {INPUT_DIR}")

    out_root = OUTPUT_DIR / f"clips_fps_{args.fps}_length_{args.clip_len}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Process every source video independently.
    videos = sorted(INPUT_DIR.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in INPUT_DIR: {INPUT_DIR}")

    for idx, video in enumerate(videos, start=1):
        times, scores = motion_scores(video, args.fps)
        starts = pick_top3_no_overlap_else_allow(times, scores, args.clip_len)

        vid_dir = out_root / video.stem
        vid_dir.mkdir(exist_ok=True)

        # Emit up to three ranked clips per input video.
        for i, t0 in enumerate(starts, start=1):
            out_clip = vid_dir / f"clip_{i}.mp4"
            cut_clip(video, float(t0), args.clip_len, out_clip, args.fps)
            if not out_clip.exists() or out_clip.stat().st_size == 0:
                raise RuntimeError(f"ffmpeg produced empty clip: {out_clip}")

        print(
            f"[{idx}/{len(videos)}] processed:",
            video.name,
            "starts:",
            [round(float(x), 2) for x in starts],
        )

if __name__ == "__main__":
    main()
