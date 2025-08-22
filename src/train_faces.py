"""
Build FaceNet embeddings from data/known_faces/<person>/*.png
Train an SVM classifier and save artifacts into models/
"""
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as T

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

IMAGE_SIZE = 160
DATA_DIR = Path("data/known_faces")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = MODELS_DIR / "embeddings.npz"
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

def load_image(path: Path):
    img = Image.open(path).convert("RGB")
    tfm = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])
    return tfm(img)

def main():
    people_dirs = [p for p in DATA_DIR.iterdir() if p.is_dir()]
    if not people_dirs:
        raise SystemExit(f"No folders in {DATA_DIR}. Collect faces first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    X, y = [], []
    for person_dir in people_dirs:
        label = person_dir.name
        images = sorted([*person_dir.glob("*.png"), *person_dir.glob("*.jpg"), *person_dir.glob("*.jpeg")])
        if not images: 
            print(f"[WARN] No images in {person_dir}, skipping.")
            continue
        for img_path in tqdm(images, desc=label):
            tensor = load_image(img_path).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = resnet(tensor).cpu().numpy().flatten()
            X.append(emb)
            y.append(label)

    X = np.vstack(X)
    y = np.array(y)
    np.savez_compressed(EMBEDDINGS_PATH, X=X, y=y)
    print(f"[OK] Saved embeddings → {EMBEDDINGS_PATH} (X: {X.shape}, y: {y.shape})")

    # Train SVM classifier
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    clf = SVC(kernel="linear", probability=True, C=1.0)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    dump(clf, CLASSIFIER_PATH)
    dump(le, LABEL_ENCODER_PATH)
    print(f"[DONE] Saved: {CLASSIFIER_PATH} and {LABEL_ENCODER_PATH}")

if __name__ == "__main__":
    main()
