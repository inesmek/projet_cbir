# api/indexer.py
import os, glob, json, time
from pathlib import Path
from typing import List
import cv2
import numpy as np
from elasticsearch import Elasticsearch, helpers, ConnectionTimeout, TransportError

from features import (
    feat_vgg16, feat_hog, feat_lbp,
    raw_hog, raw_lbp, fit_pcas, load_pcas
)

# ------------------------
# CONFIG
# ------------------------
DATA_DIR = Path("data")
INDEX = "images_cbir"
ES_HOST = os.getenv("ES_HOST", "http://localhost:9220")

# client ES avec timeouts et retries
ES = Elasticsearch(
    ES_HOST,
    request_timeout=120,
    retry_on_timeout=True,
    max_retries=5
)

# Extensions d'images supportées
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ------------------------
# UTILITAIRES
# ------------------------
def list_images() -> List[str]:
    """Liste récursive de toutes les images sous data/images*/**"""
    files = []
    for root in sorted(DATA_DIR.glob("images*")):
        files += glob.glob(str(root / "**" / "*.*"), recursive=True)
    return [f for f in files if Path(f).suffix.lower() in IMG_EXTS]


def get_tags_for_image(image_path: Path):
    """Associe image -> fichier texte de tags (un tag par ligne)."""
    tags_root = DATA_DIR / "tags" / "tags"
    num_dir = image_path.parent.name
    stem = image_path.stem
    candidate = tags_root / num_dir / f"{stem}.txt"
    if candidate.exists():
        lines = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
        tags = [ln.strip().lower() for ln in lines if ln.strip()]
        seen, uniq = set(), []
        for t in tags:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    hits = list((DATA_DIR / "tags").rglob(f"{stem}.txt"))
    if hits:
        lines = hits[0].read_text(encoding="utf-8", errors="ignore").splitlines()
        return [ln.strip().lower() for ln in lines if ln.strip()]

    return []


def ensure_index_exists():
    """Vérifie l'existence de l'index avant d'indexer."""
    if not ES.indices.exists(index=INDEX):
        raise RuntimeError(
            f"L'index '{INDEX}' n'existe pas. Crée-le avant avec :\n"
            f'  Invoke-RestMethod -Method Put -Uri http://localhost:9220/{INDEX} -ContentType "application/json" -InFile "es/mapping.json"'
        )


def build_pcas(sample_size: int = 1000):
    """Fit PCA HOG/LBP sur un échantillon et sauvegarde dans models/."""
    print(f">> Construction PCA (échantillon de {sample_size} images)...")
    paths = list_images()[:sample_size]
    Xh, Xl = [], []
    for i, p in enumerate(paths, 1):
        img = cv2.imread(p)
        if img is None:
            continue
        Xh.append(raw_hog(img))
        Xl.append(raw_lbp(img))
        if i % 50 == 0:
            print(f"  → {i}/{len(paths)} images traitées pour PCA")
    if len(Xh) == 0 or len(Xl) == 0:
        raise RuntimeError("Pas assez d'images valides pour fitter les PCA.")
    fit_pcas(np.array(Xh), np.array(Xl), save_dir="models")
    print(">> PCA HOG/LBP entraînées et enregistrées dans models/")


# ------------------------
# PIPELINE D'INDEXATION
# ------------------------
def index_all(batch_size: int = 200):
    """Indexe toutes les images dans Elasticsearch avec gestion des erreurs."""
    ensure_index_exists()

    # Charger/entraîner PCA si besoin
    p_hog, p_lbp = load_pcas()
    if p_hog is None or p_lbp is None:
        build_pcas(sample_size=200)
        p_hog, p_lbp = load_pcas()

    imgs = list_images()
    total = len(imgs)
    if total == 0:
        print("Aucune image trouvée dans data/images*/**")
        return

    print(f">> {total} images détectées. Début de l'indexation vers '{INDEX}' ...")
    actions, done, batches = [], 0, 0

    for path in imgs:
        p = Path(path)
        img = cv2.imread(str(p))
        if img is None:
            continue

        rel = str(p.relative_to(DATA_DIR)).replace("\\", "/")
        # Skip si déjà indexé
        try:
            if ES.exists(index=INDEX, id=rel):
                continue
        except Exception:
            pass

        folder = "/".join(Path(rel).parts[:3]) if len(Path(rel).parts) >= 3 else "unknown"
        tags = get_tags_for_image(p)

        try:
            vgg = feat_vgg16(img).tolist()
            hog_v = feat_hog(img, p_hog).tolist()
            lbp_v = feat_lbp(img, p_lbp).tolist()
        except Exception as e:
            print(f"[WARN] Feature extraction échouée pour {rel}: {e}")
            continue

        doc = {
            "id": rel,
            "path": rel,
            "folder": folder,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "tags": " ".join(tags),
            "tags_raw": "",
            "vgg16": vgg,
            "hog": hog_v,
            "lbp": lbp_v,
        }
        actions.append({"_op_type": "index", "_index": INDEX, "_id": doc["id"], "_source": doc})

        if len(actions) >= batch_size:
            try:
                helpers.bulk(
                    ES, actions,
                    chunk_size=batch_size,
                    request_timeout=120,
                    max_retries=5,
                    raise_on_error=False
                )
                done += len(actions)
                batches += 1
                actions = []
                print(f">> {done}/{total} indexées ...")
                if batches % 20 == 0:
                    time.sleep(2)  # pause légère
            except (ConnectionTimeout, TransportError) as e:
                print(f"[WARN] Timeout ou transport error : {e}, reprise dans 5s...")
                time.sleep(5)

    # Reste à envoyer
    if actions:
        helpers.bulk(
            ES, actions,
            chunk_size=batch_size,
            request_timeout=120,
            max_retries=5,
            raise_on_error=False
        )
        done += len(actions)

    print(f"✅ Indexation terminée : {done}/{total} images indexées dans '{INDEX}'.")


# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    try:
        index_all(batch_size=200)
    except Exception as e:
        print(f"[ERREUR] {e}")
