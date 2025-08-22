"""
Run real-time face recognition.
Usage:
  python src/recognize_faces.py --camera 0
Press 'q' to quit.
"""
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from joblib import load

from utils import draw_box_with_label

IMAGE_SIZE = 160
DETECTION_THRESH = 0.90    # Face detection confidence
UNKNOWN_THRESH = 0.60      # Min class probability to accept a name

CLASSIFIER_PATH = "models/classifier.joblib"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"

def main(camera: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(image_size=IMAGE_SIZE, keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    clf = load(CLASSIFIER_PATH)
    le = load(LABEL_ENCODER_PATH)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Try --camera 1")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        boxes, probs = mtcnn.detect(pil_img)
        faces = mtcnn(pil_img)

        if boxes is not None and faces is not None:
            for i, (box, p) in enumerate(zip(boxes, probs)):
                if p is None or p < DETECTION_THRESH:
                    continue
                face_tensor = faces[i].unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = resnet(face_tensor).cpu().numpy()
                proba = clf.predict_proba(emb)[0]
                j = int(np.argmax(proba))
                conf = float(proba[j])
                name = le.inverse_transform([j])[0]
                label = name if conf >= UNKNOWN_THRESH else "Unknown"
                draw_box_with_label(frame_bgr, box, label, conf)

        cv2.imshow("Face Recognition", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (0/1)")
    args = ap.parse_args()
    main(args.camera)
