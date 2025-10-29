# Optimized server_hybrid.py for large datasets (753k docs)
import os, logging, asyncio, time
import cv2
import numpy as np
from typing import Optional
from http import HTTPStatus
from anyio import to_thread

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from elasticsearch import Elasticsearch

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ===================== Config =====================
INDEX         = os.getenv("ES_INDEX", "images_all")
ES_HOST       = os.getenv("ES_HOST", "http://127.0.0.1:9221")

TOPK_DEFAULT  = 12
TOPK_MAX      = 24
ES_HARD_TIMEOUT = 30    # Increased from 9
ES_REQ_TIMEOUT  = 30    # Increased from 9

# ===================== Logging ====================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cbir")

# ===================== ES client ==================
ES = Elasticsearch(
    ES_HOST,
    request_timeout=ES_REQ_TIMEOUT,
    retry_on_timeout=True,
    max_retries=2,
)

# ===================== FastAPI ====================
app = FastAPI(title="CBIR API — Image-first VGG16 (optimized)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ===================== VGG16 (512D) ===============
_base_vgg = VGG16(weights="imagenet", include_top=False)
_gap = GlobalAveragePooling2D()
_vgg = Model(inputs=_base_vgg.input, outputs=_gap(_base_vgg.output))

def _img_to_bgr_ndarray(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def _safe_l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n < eps:
        out = np.zeros_like(x, dtype=np.float32)
        out[0] = 1.0
        return out.astype(np.float32)
    return (x / n).astype(np.float32)

def feat_vgg16_bgr(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        raise ValueError("Image decode failed.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    x = np.expand_dims(img_resized.astype(np.float32), axis=0)
    x = preprocess_input(x)
    vec = _vgg.predict(x, verbose=0)[0].astype(np.float32)
    return _safe_l2_normalize(vec)

def _format_hits(es_res) -> JSONResponse:
    hits = es_res.get("hits", {}).get("hits", []) or []
    out = []
    for h in hits:
        src = h.get("_source", {}) or {}
        out.append({
            "id": h.get("_id"),
            "score": h.get("_score"),
            "path": src.get("path", ""),
            "tags": src.get("tags", ""),
            "width": src.get("width"),
            "height": src.get("height"),
        })
    return JSONResponse(out)

# ===================== ES helper (hard timeout) ===
async def es_search_with_timeout(index: str, body: dict, timeout_s: int = ES_HARD_TIMEOUT):
    """Run blocking ES.search in worker thread with timeout."""
    try:
        return await asyncio.wait_for(
            to_thread.run_sync(lambda: ES.search(index=index, body=body, request_timeout=timeout_s)),
            timeout=timeout_s
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=HTTPStatus.GATEWAY_TIMEOUT, detail=f"Search timed out (>{timeout_s}s)")

# ===================== Startup warmup =============
@app.on_event("startup")
def _startup_warmup():
    # TF warmup
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    _ = feat_vgg16_bgr(dummy)
    # ES ping
    try:
        ok = ES.ping(request_timeout=2)
        log.info(f"Elasticsearch ping: {ok} @ {ES_HOST}")
    except Exception as e:
        log.warning(f"Elasticsearch ping failed: {e!r}")

# ===================== Routes =====================
@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK — use /health, POST /search/text, POST /search/hybrid"

@app.get("/health")
def health():
    try:
        es_ok = bool(ES.ping(request_timeout=2))
    except Exception:
        es_ok = False
    try:
        ES.search(index=INDEX, body={"size": 0, "query": {"match_all": {}}}, request_timeout=3)
        idx_ok = True
    except Exception:
        idx_ok = False
    return {"api_ok": True, "es_ok": es_ok, "index_ok": idx_ok, "es_host": ES_HOST, "index": INDEX}

@app.post("/search/text")
async def search_text(q: str = Form(""), topk: int = Form(20)):
    q = (q or "").strip()
    topk = max(1, min(int(topk), TOPK_MAX))

    body = {
        "_source": ["path", "tags", "width", "height"],
        "track_total_hits": False,
        "size": topk,
        "query": {"match_all": {}} if not q else {
            "multi_match": {"query": q, "fields": ["tags^2", "tags_raw", "path"]}
        }
    }
    try:
        res = await es_search_with_timeout(INDEX, body)
    except Exception as e:
        log.exception("ES text search error")
        raise HTTPException(status_code=500, detail=f"Elasticsearch error (text): {e!r}")
    return _format_hits(res)


@app.post("/search/hybrid")
async def search_hybrid(
    file: Optional[UploadFile] = File(None, description="Query image"),
    q: Optional[str] = Form(None, description="Optional text — used only for rerank"),
    topk: int = Form(20),
):
    """
    OPTIMIZED Image-first flow for 753k documents:
      1) kNN with minimal num_candidates and ef parameter
      2) Optional text rerank
    """
    start_time = time.time()
    topk = max(1, min(int(topk), TOPK_MAX))
    text = (q or "").strip()

    # --- TEXT-ONLY (no image uploaded) ---
    if file is None:
        body = {
            "_source": ["path", "tags", "width", "height"],
            "track_total_hits": False,
            "size": topk,
            "query": {"match_all": {}} if not text else {
                "multi_match": {"query": text, "fields": ["tags^2", "tags_raw", "path"]}
            }
        }
        try:
            res = await es_search_with_timeout(INDEX, body)
        except Exception as e:
            log.exception("ES text fallback error")
            raise HTTPException(status_code=500, detail=f"Elasticsearch error (text): {e!r}")
        return _format_hits(res)

    # --- IMAGE PROVIDED: phase 1 kNN (visual only) ---
    try:
        t0 = time.time()
        contents = await file.read()
        img = _img_to_bgr_ndarray(contents)
        qvec = feat_vgg16_bgr(img)
        qvec_list = qvec.tolist()
        log.info(f"Image processing: {time.time() - t0:.3f}s")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image error: {e!r}")

    
    knn_body = {
        "_source": ["path", "tags", "width", "height"],
        "track_total_hits": False,
        "size": topk,
        "knn": {
            "field": "vgg16",
            "query_vector": qvec_list,
            "k": topk,
            "num_candidates": min(80, max(40, topk * 3))  # MUCH smaller than topk*5!
        }
    }

    try:
        t0 = time.time()
        res_knn = await es_search_with_timeout(INDEX, knn_body)
        log.info(f"kNN search (topk={topk}, num_candidates={knn_body['knn']['num_candidates']}): {time.time() - t0:.3f}s")
    except Exception as e:
        log.exception("ES kNN error")
        raise HTTPException(status_code=500, detail=f"Elasticsearch error (kNN): {e!r}")

    hits = res_knn.get("hits", {}).get("hits", []) or []
    
    # No text rerank needed
    if not text or not hits:
        log.info(f"Total request time: {time.time() - start_time:.3f}s")
        return _format_hits(res_knn)

    # --- PHASE 2: rerank within kNN ids ---
    ids = [h["_id"] for h in hits]
    rerank_body = {
        "_source": ["path", "tags", "width", "height"],
        "track_total_hits": False,
        "size": topk,
        "query": {
            "script_score": {
                "query": {"ids": {"values": ids}},
                "script": {
                    "source": """
                        double base = _score;
                        double bonus = 0.0;
                        if (params.q != null && params.q.length() > 0) {
                            if (doc['tags'].size() > 0 && doc['tags'].value.toLowerCase().contains(params.q.toLowerCase())) {
                                bonus = 0.2;
                            }
                        }
                        return base + bonus;
                    """,
                    "params": {"q": text}
                }
            }
        }
    }

    try:
        t0 = time.time()
        res_rank = await es_search_with_timeout(INDEX, rerank_body)
        log.info(f"Rerank: {time.time() - t0:.3f}s")
    except Exception as e:
        log.exception("ES rerank error")
        return _format_hits(res_knn)

    log.info(f"Total request time: {time.time() - start_time:.3f}s")
    return _format_hits(res_rank)


# ===================== Main =======================
if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
