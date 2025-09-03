"""
blink_detector_mp.py
Lightweight blink detection using MediaPipe Face Mesh + EAR.

Install:
    pip install mediapipe opencv-python numpy tqdm

Usage (Python):
    from blink_detector_mp import BlinkDetector, BlinkMetrics
    bd = BlinkDetector(ear_thresh=0.20, consec_frames=3, annotate=False)
    m = bd.process_video("/path/to/video.mp4",
                         out_csv="/path/to/per_frame.csv",
                         out_video="/path/to/annot.mp4",
                         adaptive_thresh=False)
    print(m)

CLI:
    python blink_detector_mp.py --video /path/to/video.mp4 --out_csv out.csv --out_video annot.mp4 \
        --ear_thresh 0.20 --consec_frames 3 --annotate --adaptive

Notes:
- Rule-based (no training). Threshold can be fixed or adaptive.
- EAR eye indices use MediaPipe Face Mesh (refine_landmarks=True).
"""

from __future__ import annotations

import os
import math
import csv
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp


@dataclass
class BlinkMetrics:
    video: str
    ok: bool
    reason: str = ""
    frames: int = 0
    fps: float = 0.0
    duration_s: float = 0.0
    blinks: int = 0
    blinks_per_min: float = 0.0
    ear_mean: float = float("nan")
    ear_median: float = float("nan")
    csv_path: str = ""
    annot_path: str = ""
    def to_dict(self):
        return asdict(self)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||); eye is 6x2 array."""
    A = _euclidean(eye[1], eye[5])
    B = _euclidean(eye[2], eye[4])
    C = _euclidean(eye[0], eye[3]) + 1e-8
    return (A + B) / (2.0 * C)


class BlinkDetector:
    """
    MediaPipe Face Mesh-based blink detector.
    """
    # Eye landmark indices (MediaPipe Face Mesh, refined)
    # Left eye:  p1=33,  p2=160, p3=158, p4=133, p5=153, p6=144
    # Right eye: p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
    L_EYE = [33, 160, 158, 133, 153, 144]
    R_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(
        self,
        ear_thresh: float = 0.20,
        consec_frames: int = 3,
        annotate: bool = False,
        mp_det_conf: float = 0.5,
        mp_track_conf: float = 0.5,
    ) -> None:
        self.ear_thresh = float(ear_thresh)
        self.consec_frames = int(consec_frames)
        self.annotate = bool(annotate)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=mp_det_conf,
            min_tracking_confidence=mp_track_conf,
        )

    @staticmethod
    def _landmarks_to_eye_points(landmarks, indices, w: int, h: int) -> np.ndarray:
        """Extract 6 (x,y) points in pixel coords from mediapipe landmarks."""
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            x = int(round(lm.x * w))
            y = int(round(lm.y * h))
            pts.append([x, y])
        return np.array(pts, dtype=np.float32)

    def _extract_eyes(self, frame_bgr) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (leftEye[6x2], rightEye[6x2]) or (None, None) if not found."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_face_mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return None, None
        face = res.multi_face_landmarks[0].landmark
        left = self._landmarks_to_eye_points(face, self.L_EYE, w, h)
        right = self._landmarks_to_eye_points(face, self.R_EYE, w, h)
        return left, right

    def process_video(
        self,
        video_path: str,
        out_csv: Optional[str] = None,
        out_video: Optional[str] = None,
        adaptive_thresh: bool = False,
        clamp_range: Tuple[float, float] = (0.15, 0.28),
    ) -> BlinkMetrics:
        """Process one video; returns BlinkMetrics."""
        base = os.path.splitext(os.path.basename(video_path))[0]
        metrics = BlinkMetrics(video=base, ok=False)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            metrics.reason = "OpenCV failed to open video"
            return metrics

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        writer = None
        if self.annotate and out_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

        csv_file = None
        writer_csv = None
        if out_csv:
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            csv_file = open(out_csv, "w", newline="")
            writer_csv = csv.writer(csv_file)
            writer_csv.writerow(["frame_idx", "time_s", "ear_left", "ear_right", "ear_avg", "blink_state"])

        blink_count = 0
        consec = 0
        frame_idx = 0
        ear_series: List[float] = []

        # Pass 1: collect EARs for adaptive threshold if requested
        if adaptive_thresh:
            pbar1 = tqdm(total=total_frames if total_frames else None, desc=f"{base} [pass1]", unit="f")
            ears_tmp: List[float] = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                left, right = self._extract_eyes(frame)
                if left is not None and right is not None:
                    ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
                    ears_tmp.append(ear)
                pbar1.update(1)
            pbar1.close()
            cap.release()
            if len(ears_tmp) >= 30:
                p10 = float(np.percentile(ears_tmp, 10))
                thr = max(clamp_range[0], min(clamp_range[1], p10 * 0.95))
            else:
                thr = self.ear_thresh
            cap = cv2.VideoCapture(video_path)
        else:
            thr = self.ear_thresh

        # Pass 2: main loop
        pbar = tqdm(total=total_frames if total_frames else None, desc=f"{base}", unit="f")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            left, right = self._extract_eyes(frame)
            if left is not None and right is not None:
                earL = eye_aspect_ratio(left)
                earR = eye_aspect_ratio(right)
                ear = (earL + earR) / 2.0
                ear_series.append(ear)

                blink_state = 0
                if ear < thr:
                    consec += 1
                else:
                    if consec >= self.consec_frames:
                        blink_count += 1
                        blink_state = 1
                    consec = 0

                if writer is not None:
                    cv2.polylines(frame, [cv2.convexHull(left.astype(np.int32))], True, (0, 255, 0), 1)
                    cv2.polylines(frame, [cv2.convexHull(right.astype(np.int32))], True, (0, 255, 0), 1)
                    cv2.putText(frame, f"EAR:{ear:.3f} thr:{thr:.3f} blinks:{blink_count}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    writer.write(frame)
            else:
                earL = earR = ear = float("nan")
                blink_state = 0
                ear_series.append(np.nan)
                if writer is not None:
                    writer.write(frame)

            if writer_csv is not None:
                writer_csv.writerow([frame_idx, frame_idx / fps, earL, earR, ear, blink_state])

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()

        duration_s = frame_idx / fps if fps > 0 else 0.0
        blinks_per_min = (blink_count / duration_s * 60.0) if duration_s > 0 else 0.0
        ear_clean = np.array([e for e in ear_series if not (isinstance(e, float) and math.isnan(e))])
        ear_mean = float(np.mean(ear_clean)) if ear_clean.size else float("nan")
        ear_median = float(np.median(ear_clean)) if ear_clean.size else float("nan")

        metrics.ok = True
        metrics.frames = frame_idx
        metrics.fps = float(fps)
        metrics.duration_s = duration_s
        metrics.blinks = blink_count
        metrics.blinks_per_min = blinks_per_min
        metrics.ear_mean = ear_mean
        metrics.ear_median = ear_median
        metrics.csv_path = out_csv or ""
        metrics.annot_path = out_video or ""
        return metrics

    def process_dir(
        self,
        video_dir: str,
        out_dir: str,
        patterns: Tuple[str, ...] = ("mp4", "mov", "avi", "mkv", "MP4", "MOV", "AVI", "MKV"),
        annotate: Optional[bool] = None,
        adaptive_thresh: bool = False,
    ) -> List[BlinkMetrics]:
        """Batch process a folder; returns list of BlinkMetrics."""
        os.makedirs(out_dir, exist_ok=True)
        csv_dir = os.path.join(out_dir, "per_frame_csv")
        ann_dir = os.path.join(out_dir, "annotated_video")
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        files = []
        for ext in patterns:
            files.extend([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(f".{ext}")])
        files = sorted(files)

        results: List[BlinkMetrics] = []
        for path in files:
            base = os.path.splitext(os.path.basename(path))[0]
            out_csv = os.path.join(csv_dir, f"{base}.csv")
            out_vid = os.path.join(ann_dir, f"{base}_annot.mp4") if (self.annotate or annotate) else None
            m = self.process_video(path, out_csv=out_csv, out_video=out_vid, adaptive_thresh=adaptive_thresh)
            results.append(m)

        # Summary CSV
        summary_path = os.path.join(out_dir, "blink_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video","ok","reason","frames","fps","duration_s","blinks","blinks_per_min",
                             "ear_mean","ear_median","csv_path","annot_path"])
            for m in results:
                writer.writerow([m.video, m.ok, m.reason, m.frames, m.fps, m.duration_s, m.blinks,
                                 m.blinks_per_min, m.ear_mean, m.ear_median, m.csv_path, m.annot_path])
        return results


# -------------------------
# CLI support
# -------------------------
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Blink detection via MediaPipe EAR")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="Single video path")
    g.add_argument("--video_dir", help="Directory of videos to process")
    p.add_argument("--out_csv", help="Per-frame CSV (single video)")
    p.add_argument("--out_video", help="Annotated MP4 path (single video)")
    p.add_argument("--out_dir", help="Output dir (for --video_dir)")
    p.add_argument("--ear_thresh", type=float, default=0.20, help="EAR threshold for closed eye")
    p.add_argument("--consec_frames", type=int, default=3, help="Frames below threshold to count blink")
    p.add_argument("--annotate", action="store_true", help="Render annotated video")
    p.add_argument("--adaptive", action="store_true", help="Use per-video adaptive thresholding")
    return p.parse_args()


def _main():
    args = _parse_args()
    bd = BlinkDetector(
        ear_thresh=args.ear_thresh,
        consec_frames=args.consec_frames,
        annotate=args.annotate,
    )
    if args.video:
        m = bd.process_video(args.video, out_csv=args.out_csv, out_video=args.out_video, adaptive_thresh=args.adaptive)
        print(m.to_dict())
    else:
        if not args.out_dir:
            raise SystemExit("--out_dir is required when using --video_dir")
        results = bd.process_dir(args.video_dir, args.out_dir, adaptive_thresh=args.adaptive)
        for m in results:
            print(m.to_dict())


if __name__ == "__main__":
    _main()
