import os, cv2, numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as kimage
import joblib

# --- VGG16 global avg pooled (512D) ---
_vgg = VGG16(include_top=False, pooling="avg", weights="imagenet")
def feat_vgg16(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (224,224))
    x = kimage.img_to_array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    x = np.expand_dims(x, 0)
    x = preprocess_input(x)
    vec = _vgg.predict(x, verbose=0)[0].astype(np.float32)
    n = np.linalg.norm(vec) + 1e-8
    return vec / n

# --- HOG raw + PCA(256) ---
def raw_hog(img_bgr):
    gray = cv2.cvtColor(cv2.resize(img_bgr,(256,256)), cv2.COLOR_BGR2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
               block_norm="L2-Hys", visualize=False, feature_vector=True).astype(np.float32)

# --- LBP (uniform) en 4x4 blocs + histos concat ---
def raw_lbp(img_bgr):
    gray = cv2.cvtColor(cv2.resize(img_bgr,(256,256)), cv2.COLOR_BGR2GRAY)
    P, R = 8, 1
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    h, w = lbp.shape; bh, bw = h//4, w//4
    bins = P + 2  # 10
    hists = []
    for i in range(4):
        for j in range(4):
            block = lbp[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            hist, _ = np.histogram(block.ravel(), bins=np.arange(bins+1), range=(0,bins))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-8)
            hists.append(hist)
    return np.concatenate(hists).astype(np.float32)  # 160D

def fit_pcas(Xh, Xl, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    pca_hog = PCA(n_components=256, random_state=0).fit(Xh)
    pca_lbp = PCA(n_components=128, random_state=0).fit(Xl)
    joblib.dump(pca_hog, os.path.join(save_dir,"pca_hog.pkl"))
    joblib.dump(pca_lbp, os.path.join(save_dir,"pca_lbp.pkl"))

def load_pcas(save_dir="models"):
    p_hog = joblib.load(os.path.join(save_dir,"pca_hog.pkl")) if os.path.exists(os.path.join(save_dir,"pca_hog.pkl")) else None
    p_lbp = joblib.load(os.path.join(save_dir,"pca_lbp.pkl")) if os.path.exists(os.path.join(save_dir,"pca_lbp.pkl")) else None
    return p_hog, p_lbp

def feat_hog(img_bgr, pca):
    v = raw_hog(img_bgr)
    if pca is not None:
        v = pca.transform([v])[0]
    v = v.astype(np.float32)
    v /= (np.linalg.norm(v)+1e-8)
    return v

def feat_lbp(img_bgr, pca):
    v = raw_lbp(img_bgr)
    if pca is not None:
        v = pca.transform([v])[0]
    v = v.astype(np.float32)
    v /= (np.linalg.norm(v)+1e-8)
    return v
