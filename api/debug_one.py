import json, os
from pathlib import Path
import numpy as np
import cv2
from elasticsearch import Elasticsearch
from features import feat_vgg16, feat_hog, feat_lbp, load_pcas
from indexer import DATA_DIR, INDEX

ES_HOST = os.getenv("ES_HOST", "http://localhost:9220")
es = Elasticsearch(ES_HOST)

def list_one_image():
    # reprise de list_images() mais on retourne la première image trouvée
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
    for root in sorted((DATA_DIR).glob("images*")):
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                return p
    return None

def sanitize(v, expected):
    v = np.asarray(v, dtype=np.float32).ravel()
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    n = float(np.linalg.norm(v))
    if n > 0: v = v / n
    if v.shape[0] > expected: v = v[:expected]
    if v.shape[0] < expected: v = np.pad(v, (0, expected - v.shape[0]))
    return v.astype(np.float32)

if __name__ == "__main__":
    img_path = list_one_image()
    if not img_path:
        print("Aucune image trouvée sous data/images*/**"); raise SystemExit(1)
    print("Test sur:", img_path)

    img = cv2.imread(str(img_path))
    if img is None:
        print("Impossible de lire l'image"); raise SystemExit(1)

    # PCA (doivent exister si tu as déjà lancé indexer une fois; sinon on teste sans PCA)
    p_hog, p_lbp = load_pcas()

    # Features
    vgg = sanitize(feat_vgg16(img), 512)
    hog = sanitize(feat_hog(img, p_hog), 256)
    lbp = sanitize(feat_lbp(img, p_lbp), 128)

    print("lens → vgg16:", len(vgg), "hog:", len(hog), "lbp:", len(lbp))
    print("Any NaN?  vgg:", np.isnan(vgg).any(), "hog:", np.isnan(hog).any(), "lbp:", np.isnan(lbp).any())

    rel = str(img_path.relative_to(DATA_DIR)).replace("\\","/")
    doc = {
        "id": rel, "path": rel, "folder": "debug",
        "width": int(img.shape[1]), "height": int(img.shape[0]),
        "tags": "", "tags_raw": "",
        "vgg16": vgg.tolist(), "hog": hog.tolist(), "lbp": lbp.tolist()
    }

    try:
        resp = es.index(index=INDEX, id=doc["id"], document=doc)
        print("INDEX OK:", resp.get("result"))
    except Exception as e:
        # Affiche l'info ES détaillée si disponible
        info = getattr(e, "info", None)
        print("INDEX ERROR RAW:", repr(e))
        if info:
            print("INDEX ERROR INFO:", json.dumps(info, indent=2))
        raise