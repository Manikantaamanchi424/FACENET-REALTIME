"""
Usage:
  python src/face_capture.py --person "Alice" --num 60 --camera 0
Press 'q' to quit early.
"""
import argparse
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from utils import ensure_dir, draw_box_with_label

IMAGE_SIZE = 160
DETECTION_THRESH = 0.90

def main(person: str, num: int, camera: int):
    out_dir = ensure_dir(Path("data/known_faces") / person)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(image_size=IMAGE_SIZE, margin=20, keep_all=False, post_process=True, device=device)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found. Try --camera 1 or check permissions.")

    saved, last_save = 0, 0.0
    print(f"[INFO] Collecting faces for '{person}' into: {out_dir}")

    while saved < num:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        boxes, probs = mtcnn.detect(pil_img)
        if boxes is not None:
            for box, p in zip(boxes, probs):
                if p and p >= DETECTION_THRESH:
                    draw_box_with_label(frame_bgr, box, person)

        face_aligned = mtcnn(pil_img)  # 3x160x160 tensor (aligned) or None
        if face_aligned is not None:
            now = time.time()
            if now - last_save > 0.2:  # throttle saves (~5 fps)
                fname = out_dir / f"{int(now*1000)}.png"
                Image.fromarray((face_aligned.permute(1, 2, 0).numpy() * 255).astype("uint8")).save(fname)
                saved += 1
                last_save = now

        cv2.putText(frame_bgr, f"Saved: {saved}/{num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Collecting Faces", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Saved {saved} images → {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="Person name (folder)")
    ap.add_argument("--num", type=int, default=60, help="Number of images to save")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (0/1)")
    args = ap.parse_args()
    main(args.person, args.num, args.camera)
