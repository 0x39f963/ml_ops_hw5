from pathlib import Path
import json

import cv2
import gdown
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "artifacts" / "images"
VIDEO_DIR = ROOT / "artifacts" / "videos"

VIDEO_ID = "1OjSrdAS4zVfa1u7vtyh48PKEoVW20vPN"
VIDEO_URL = f"https://drive.google.com/uc?export=download&id={VIDEO_ID}"
INPUT_VIDEO = VIDEO_DIR / "HW5_Woman_Happy.mp4"
BLURRED_VIDEO = VIDEO_DIR / "HW5_Woman_Happy_blurred.mp4"

INPUT_PREVIEW = IMAGE_DIR / "video_input_preview.gif"
BLURRED_PREVIEW = IMAGE_DIR / "video_blurred_preview.gif"
FRAME_BEFORE = IMAGE_DIR / "video_frame_before.jpg"
FRAME_AFTER = IMAGE_DIR / "video_frame_after.jpg"
DIAGRAM = IMAGE_DIR / "face_blur_ml_system.png"
VIDEO_STATS = ROOT / "artifacts" / "reports" / "video_processing_stats.json"


def download_video() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    if INPUT_VIDEO.exists() and INPUT_VIDEO.stat().st_size > 100_000:
        print({"video": str(INPUT_VIDEO), "status": "already_exists"})
        return

    print("downloading source video from Google Drive...")
    gdown.download(id=VIDEO_ID, output=str(INPUT_VIDEO), quiet=False)
    if not INPUT_VIDEO.exists() or INPUT_VIDEO.stat().st_size < 100_000:
        raise RuntimeError(f"Video was not downloaded correctly: {INPUT_VIDEO}")


def mosaic_roi(face_roi: np.ndarray, pixel_size: int = 72) -> np.ndarray:
    h, w = face_roi.shape[:2]
    if h == 0 or w == 0:
        return face_roi
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    face_roi = cv2.GaussianBlur(face_roi, (99, 99), 0)
    small = cv2.resize(face_roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaiced = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(mosaiced, (51, 51), 0)


def detect_faces(gray_frame: np.ndarray, cascade: cv2.CascadeClassifier) -> list[tuple[int, int, int, int]]:
    faces = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(45, 45),
    )
    return list(faces)


def resize_rgb_for_preview(frame_rgb: np.ndarray, max_width: int = 640) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    if w <= max_width:
        return frame_rgb
    new_h = int(h * max_width / w)
    return cv2.resize(frame_rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)


def save_preview_jpg(path: Path, frame_bgr: np.ndarray, max_width: int = 1280) -> None:
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        new_h = int(h * max_width / w)
        frame_bgr = cv2.resize(frame_bgr, (max_width, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def process_video() -> dict[str, int | float]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Cannot load cascade: {cascade_path}")

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(BLURRED_VIDEO), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video writer: {BLURRED_VIDEO}")

    preview_every = max(1, frame_count // 16)
    input_preview_frames = []
    output_preview_frames = []
    first_before = None
    first_after = None
    processed = 0
    frames_with_faces = 0
    total_faces = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, cascade)

        if faces:
            frames_with_faces += 1
            total_faces += len(faces)

        for x, y, w, h in faces:
            pad_x = int(w * 0.18)
            pad_y = int(h * 0.24)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(width, x + w + pad_x)
            y2 = min(height, y + h + pad_y)

            roi = out[y1:y2, x1:x2]
            out[y1:y2, x1:x2] = mosaic_roi(roi, pixel_size=72)

        if processed % preview_every == 0 and len(input_preview_frames) < 12:
            input_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            input_preview_frames.append(resize_rgb_for_preview(input_rgb))
            output_preview_frames.append(resize_rgb_for_preview(output_rgb))

        if faces and first_before is None:
            first_before = frame.copy()
            first_after = out.copy()

        writer.write(out)
        processed += 1

    cap.release()
    writer.release()

    if first_before is None:
        raise RuntimeError("No faces detected in the source video; blur result is not valid")

    save_preview_jpg(FRAME_BEFORE, first_before)
    save_preview_jpg(FRAME_AFTER, first_after)

    imageio.mimsave(INPUT_PREVIEW, input_preview_frames, fps=4)
    imageio.mimsave(BLURRED_PREVIEW, output_preview_frames, fps=4)

    return {
        "fps": round(float(fps), 2),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "processed_frames": processed,
        "frames_with_faces": frames_with_faces,
        "total_faces": total_faces,
        "blur_pixel_size": 72,
    }


def build_architecture_diagram() -> None:
    blocks = [
        "source mp4",
        "OpenCV\nVideoCapture",
        "frame queue",
        "parallel\nface workers",
        "mosaic/blur",
        "ordered\nVideoWriter",
        "blurred mp4",
        "logs/metrics",
    ]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    xs = np.linspace(0.05, 0.95, len(blocks))
    for idx, (x, label) in enumerate(zip(xs, blocks)):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "fc": "#eef4ff", "ec": "#365f91"},
        )
        if idx < len(blocks) - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - 0.045, 0.55),
                xytext=(x + 0.045, 0.55),
                arrowprops={"arrowstyle": "->", "lw": 1.5},
            )
    ax.text(
        0.5,
        0.12,
        "data parallelism: frames/chunks -> workers -> ordered join",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(DIAGRAM, dpi=160)
    plt.close(fig)


def main() -> None:
    download_video()
    stats = process_video()
    build_architecture_diagram()
    VIDEO_STATS.parent.mkdir(parents=True, exist_ok=True)
    VIDEO_STATS.write_text(
        json.dumps(
            {
                "source": str(INPUT_VIDEO.relative_to(ROOT)),
                "result": str(BLURRED_VIDEO.relative_to(ROOT)),
                "preview_input": str(INPUT_PREVIEW.relative_to(ROOT)),
                "preview_blurred": str(BLURRED_PREVIEW.relative_to(ROOT)),
                "frame_before": str(FRAME_BEFORE.relative_to(ROOT)),
                "frame_after": str(FRAME_AFTER.relative_to(ROOT)),
                **stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        {
            "input_video": str(INPUT_VIDEO),
            "blurred_video": str(BLURRED_VIDEO),
            "input_preview": str(INPUT_PREVIEW),
            "blurred_preview": str(BLURRED_PREVIEW),
            "frame_before": str(FRAME_BEFORE),
            "frame_after": str(FRAME_AFTER),
            "diagram": str(DIAGRAM),
            "stats_json": str(VIDEO_STATS),
            "stats": stats,
        }
    )


if __name__ == "__main__":
    main()
