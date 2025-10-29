# ui_streamlit.py
import os
import requests
import streamlit as st
from PIL import Image
import io
import time
from typing import Optional
from cachetools import LRUCache
from concurrent.futures import ThreadPoolExecutor
import logging
import aiohttp
import asyncio

# ---------------- Config ----------------
st.set_page_config(page_title="CBIR Hybrid", layout="wide")
API_BASE = os.environ.get("CBIR_API", "http://127.0.0.1:8000")  # backend FastAPI
CBIR_DATA_DIR = os.environ.get("CBIR_DATA_DIR", r"D:\data\cbir\data")  # fallback local
MAX_WORKERS = 4  # For parallel image loading
IMAGE_WIDTH = 300  # Target display width
CACHE_SIZE = 1000  # Cache for paths and images
BACKEND_CHECK_INTERVAL = 60  # Seconds between backend availability checks

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cbir_ui")

# Caches
path_cache = LRUCache(maxsize=CACHE_SIZE)
image_cache = LRUCache(maxsize=CACHE_SIZE)
backend_available = False  # Track backend /files/ endpoint availability
last_backend_check = 0.0  # Timestamp of last backend check

# Validate CBIR_DATA_DIR
if not os.path.exists(CBIR_DATA_DIR):
    log.warning(f"CBIR_DATA_DIR {CBIR_DATA_DIR} does not exist. Local path fallback may fail.")
    st.warning(f"Data directory {CBIR_DATA_DIR} not found. Ensure it is accessible or rely on backend.")

st.title("CBIR — Hybrid (Image+Texte)")
st.markdown("Entrez des **mots-clés** et uploadez une **image**.")

# ---------------- Check Backend Health and Files Endpoint ----------------
async def check_backend_files_endpoint():
    """Asynchronously check if the backend /files/ endpoint is available."""
    global backend_available, last_backend_check
    if time.time() - last_backend_check < BACKEND_CHECK_INTERVAL:
        return backend_available
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(f"{API_BASE}/files/test", timeout=1) as resp:
                backend_available = resp.status == 200
                last_backend_check = time.time()
                log.info(f"Backend /files/ endpoint available: {backend_available}")
    except Exception as e:
        backend_available = False
        last_backend_check = time.time()
        log.warning(f"Backend /files/ check failed: {e}")
    return backend_available

health = None
try:
    t0 = time.time()
    r = requests.get(f"{API_BASE}/health", timeout=3)
    if r.ok:
        health = r.json()
    log.info(f"Health check: {time.time() - t0:.3f}s")
    # Run async check for backend files endpoint
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    backend_available = loop.run_until_complete(check_backend_files_endpoint())
    loop.close()
except Exception as e:
    log.warning(f"Health check failed: {e}")

with st.expander("État de l'API", expanded=False):
    if health:
        st.write(health)
        st.write(f"Backend files endpoint available: {backend_available}")
    else:
        st.warning("Impossible de joindre /health. Vérifie que le serveur tourne bien.")

# Upload de l'image en dehors du formulaire
img_file = st.file_uploader(
    "Image requête",
    type=["jpg", "jpeg", "png"]
)

# Affichage immédiat de l'image
if img_file is not None:
    try:
        img = Image.open(img_file).convert("RGB")
        img_resized = img.resize((IMAGE_WIDTH, int(IMAGE_WIDTH * img.height / img.width)), Image.Resampling.LANCZOS)
        st.image(img_resized, caption="Image sélectionnée", width=IMAGE_WIDTH)
    except Exception as e:
        st.warning(f"Erreur d'affichage de l'image requête : {e}")

# ---------------- Formulaire ----------------
with st.form("hybrid_form", clear_on_submit=False):
    q = st.text_input("Mots-clés", value="")
    topk = st.slider("Top-K", min_value=5, max_value=20, value=10, step=1)
    submitted = st.form_submit_button("Rechercher")

# ---------------- Helpers ----------------
def _join_local_path(rel_path: str) -> Optional[str]:
    """Construct local path with caching and efficient normalization."""
    if not rel_path:
        return None
    if rel_path in path_cache:
        return path_cache[rel_path]
    
    # Use os.path.normpath for robust normalization
    rel_norm = os.path.normpath(rel_path).replace("\\", "/")
    abs_path = os.path.join(CBIR_DATA_DIR, rel_norm)
    if os.path.exists(abs_path):  # Validate path existence
        path_cache[rel_path] = abs_path
        return abs_path
    log.warning(f"Local path {abs_path} does not exist.")
    return None

def _best_image_source(hit: dict) -> Optional[str]:
    """
    Return the best image source. Prefer backend URL if available, else construct local path.
    """
    rel_path = hit.get("path", "")
    if not rel_path:
        return None
    
    if backend_available:
        url = f"{API_BASE}/files/{rel_path}"
        return url  # Assume backend URL is valid if endpoint is available
    return _join_local_path(rel_path)

def _load_image(src: str) -> Optional[Image.Image]:
    """Load and resize image with caching."""
    if not src:
        return None
    if src in image_cache:
        return image_cache[src]
    
    try:
        if src.startswith(("http://", "https://")):
            r = requests.get(src, timeout=5)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        else:
            if not os.path.exists(src):
                log.warning(f"Image file {src} not found.")
                return None
            img = Image.open(src).convert("RGB")
        # Resize to reduce rendering load
        img_resized = img.resize((IMAGE_WIDTH, int(IMAGE_WIDTH * img.height / img.width)), Image.Resampling.LANCZOS)
        image_cache[src] = img_resized
        return img_resized
    except Exception as e:
        log.warning(f"Failed to load image {src}: {e}")
        return None

# ---------------- Soumission ----------------
if submitted:
    # Prépare la requête HTTP multipart/form-data
    data = {"q": q, "topk": str(topk)}
    files = None
    if img_file is not None:
        files = {"file": (img_file.name, img_file.getvalue(), img_file.type or "application/octet-stream")}
    
    try:
        with st.spinner("Recherche en cours..."):
            t0 = time.time()
            if files:
                resp = requests.post(f"{API_BASE}/search/hybrid", data=data, files=files, timeout=90)
            else:
                resp = requests.post(f"{API_BASE}/search/hybrid", data=data, timeout=90)
            resp.raise_for_status()
            results = resp.json()
            log.info(f"API call: {time.time() - t0:.3f}s")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur d'appel API : {e}")
        st.stop()
    except ValueError:
        st.error("Réponse API invalide (JSON non parseable).")
        st.stop()
    
    if not isinstance(results, list):
        st.error("Format de réponse inattendu : l'API doit renvoyer une liste d'objets.")
        st.stop()
    
    if len(results) == 0:
        st.info("Aucun résultat.")
        st.stop()
    
    st.subheader(f"Résultats ({len(results)})")
    
    # --------- Affichage en grille (3 colonnes) avec lazy loading ---------
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        t0 = time.time()
        # Parallelize both source resolution and image loading
        def process_hit(hit):
            src = _best_image_source(hit)
            img = _load_image(src)
            return hit, img, src
        
        futures = [executor.submit(process_hit, hit) for hit in results]
        results_with_images = [f.result() for f in futures]
        
        log.info(f"Image loading: {time.time() - t0:.3f}s")
        
        cols = st.columns(3)
        for idx, (hit, img, src) in enumerate(results_with_images):
            with cols[idx % 3]:
                if img is not None:
                    st.image(img, caption=f"Score: {hit.get('score', 0.0):.3f}", width=IMAGE_WIDTH)
                else:
                    st.warning(f"Image introuvable pour {hit.get('path', 'N/A')}")
                # Display additional metadata if available
                for key, value in hit.items():
                    if key not in ["path", "score"]:
                        st.write(f"**{key}**: {value}")

# ---------------- Pre-warm Cache (Optional) ----------------
def prewarm_path_cache():
    """Pre-warm path cache with known paths (if available)."""
    # Example: Load from a manifest file or API endpoint
    try:
        manifest = requests.get(f"{API_BASE}/manifest", timeout=5).json()
        for path in manifest.get("paths", []):
            _join_local_path(path)
        log.info(f"Pre-warmed path cache with {len(path_cache)} paths.")
    except Exception as e:
        log.info(f"No manifest available for pre-warming: {e}")

if os.path.exists(CBIR_DATA_DIR):
    prewarm_path_cache()