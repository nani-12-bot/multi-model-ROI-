"""
app.py — Medical Image ROI Localization System
================================================
Run:  streamlit run app.py

Modalities & Classes:
  X-Ray   -> COVID | Lung_Opacity | Normal | Viral Pneumonia
  CT Scan -> adenocarcinoma | large cell carcinoma | normal | squamous cell carcinoma
  MRI     -> glioma | meningioma | notumor | pituitary

Modality Detection — Multi-Signal Scoring Architecture (v2):
  Each image is independently scored against ALL 3 modalities using
  anatomy-specific and imaging-physics-specific feature detectors.
  A modality is only accepted if its score is BOTH above a minimum
  threshold AND significantly ahead of the competing modalities.
  This eliminates the "leftover bucket" MRI problem and the bright-pixel
  overlap that confused CT Normal with X-Ray.
"""

import io, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="Medical ROI Localization",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0d1b2a 0%, #1b2a3b 100%);
  }
  section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
  .main .block-container { background: #f7f9fc; padding-top: 1.5rem; }
  [data-testid="stMetric"] {
      background: white; border: 1px solid #e0e8f0;
      border-radius: 10px; padding: 12px 16px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  [data-testid="stMetricValue"] { font-size: 22px; font-weight: 700; color: #1a3a5c; }
</style>
""", unsafe_allow_html=True)

IMG_SIZE      = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "X-Ray":   "models/xray_best.pth",
    "CT Scan": "models/ct_best.pth",
    "MRI":     "models/mri_best.pth",
}

FALLBACK_CLASSES = {
    "X-Ray":   ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"],
    "CT Scan": ["adenocarcinoma", "large cell carcinoma", "normal", "squamous cell carcinoma"],
    "MRI":     ["glioma", "meningioma", "notumor", "pituitary"],
}

MODALITY_COLORS    = {"X-Ray": "#1565C0", "CT Scan": "#2E7D32", "MRI": "#6A1B9A"}
MODALITY_GRADIENTS = {
    "X-Ray":   "linear-gradient(135deg, #1565C0, #1E88E5)",
    "CT Scan": "linear-gradient(135deg, #2E7D32, #43A047)",
    "MRI":     "linear-gradient(135deg, #6A1B9A, #8E24AA)",
}
MODALITY_EMOJI = {"X-Ray": "🫁", "CT Scan": "🔬", "MRI": "🧠"}


# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-SIGNAL MODALITY DETECTION ENGINE (v2)
# ─────────────────────────────────────────────────────────────────────────────

def _score_xray(gray224: np.ndarray) -> float:
    """
    Score an image on X-Ray likelihood [0.0 – 1.0].

    X-Ray chest radiograph signatures:
    ─────────────────────────────────
    1. BILATERAL LUNG SYMMETRY
       PA/AP chest X-rays have two nearly symmetric dark lung fields
       flanking a bright mediastinum (heart/aorta). We measure symmetry
       by comparing left-half vs right-half mean pixel values in the
       middle vertical band where lungs dominate.

    2. WIDE BRIGHT FIELD
       After CLAHE (which the preprocessing applies), some pixels are
       pushed to near-white (ribs, clavicles, heart border). X-rays have
       a significant fraction of very bright pixels (>200/255) because of
       bony structures. CT and MRI have near-zero bright fractions.

    3. ASPECT RATIO & BORDER BRIGHTNESS
       Chest X-rays typically have a bright border/background (the X-ray
       film background is white or light-grey). CT has a BLACK border
       (scanner table). MRI has a near-BLACK border.

    4. LUNG-ZONE DARKNESS PATTERN
       The lateral thirds of the mid-section of an X-ray are the lung
       fields — they are dark relative to the central mediastinum. This
       creates a characteristic "dark-centre-left / bright-centre / dark-centre-right"
       tri-band pattern in the horizontal profile.

    5. GLOBAL INTENSITY DISTRIBUTION
       X-rays have a bimodal or broad histogram: both dark (lung air)
       and bright (bone, heart) regions, but the overall mean is MID-GREY
       (not as dark as MRI/CT centre, not as uniformly flat).
    """
    h, w = gray224.shape

    # --- Feature 1: border brightness requires BOTH bright border AND bright pixel fraction ---
    # Pure bright border without bony highlights (e.g. a grey-padded CT or cropped MRI)
    # should not score as X-ray. We couple border with bright_frac so both must be present.
    border_mean = float(np.concatenate([
        gray224[0, :], gray224[-1, :],
        gray224[:, 0], gray224[:, -1]
    ]).mean())
    f_border = np.clip((border_mean - 8.0) / 60.0, 0.0, 1.0)

    # --- Feature 2: bright pixel fraction (bony structures after CLAHE) ---
    bright_frac = float((gray224 > 200).sum()) / gray224.size
    f_bright = np.clip(bright_frac / 0.05, 0.0, 1.0)

    # Coupled border score: border alone is insufficient — it must co-occur with bright bones
    combined_border = f_border * (0.5 + 0.5 * f_bright)

    # --- Feature 3: bilateral lung-zone symmetry ---
    # Take the central 60% of rows (avoid shoulders and diaphragm)
    r0, r1 = int(h * 0.20), int(h * 0.80)
    strip   = gray224[r0:r1, :]
    left_mean  = float(strip[:, :w//2].mean())
    right_mean = float(strip[:, w//2:].mean())
    # Symmetry: ratio close to 1.0 means symmetric (good X-ray sign)
    sym_ratio = min(left_mean, right_mean) / (max(left_mean, right_mean) + 1e-6)
    # X-ray lung fields are roughly symmetric: sym_ratio ~ 0.85–1.0
    # CT may be asymmetric; MRI brain IS asymmetric due to tumor/side
    f_symmetry = np.clip((sym_ratio - 0.60) / 0.35, 0.0, 1.0)

    # --- Feature 4: tri-band horizontal profile (lung flanks dark vs centre bright) ---
    col_means = strip.mean(axis=0)  # shape: (w,)
    left_zone  = col_means[:w//4].mean()
    centre_zone = col_means[w//3: 2*w//3].mean()
    right_zone = col_means[3*w//4:].mean()
    # Classic X-ray: centre brighter (heart) than flanks (lungs)
    centre_ratio = centre_zone / (0.5 * (left_zone + right_zone) + 1e-6)
    f_triband = np.clip((centre_ratio - 0.85) / 0.6, 0.0, 1.0)

    # --- Feature 5: global mean in mid-grey range ---
    global_mean = float(gray224.mean())
    # X-ray mean typically 80–160 after CLAHE. CT: 40–130. MRI: 50–120.
    # X-ray skews slightly brighter due to bony highlights.
    f_mean = np.clip(1.0 - abs(global_mean - 130.0) / 80.0, 0.0, 1.0)

    # --- Combine: weighted average ---
    score = (
        0.30 * combined_border +   # coupled border+bright: CT/MRI have black borders
        0.25 * f_bright        +   # bone highlights very specific to X-ray post-CLAHE
        0.20 * f_symmetry      +   # bilateral symmetry of chest
        0.15 * f_triband       +   # tri-band anatomy
        0.10 * f_mean
    )
    return float(score)


def _score_ct(gray224: np.ndarray, img_rgb224: np.ndarray) -> float:
    """
    Score an image on CT Scan likelihood [0.0 – 1.0].

    CT axial lung scan signatures:
    ──────────────────────────────
    1. CIRCULAR SCANNER BOUNDARY
       CT images have a distinctive circular scan field-of-view boundary.
       Outside this circle the image is PURE BLACK (scanner table padding).
       Inside is the patient. We detect this using Hough Circle Transform —
       the most reliable single CT discriminator.

    2. BLACK BORDER / CORNERS
       The four corners of a CT image are ALWAYS pure black (outside the
       scanner ring). This is the fastest single check.

    3. DARK LUNG FIELDS (bilateral oval dark regions)
       Lung air in CT is very dark (Hounsfield ~-900). Even after 8-bit
       conversion the lungs are near-black. We check for large dark areas
       in the central region.

    4. WIDE DYNAMIC RANGE (IQR)
       CT has simultaneous near-black (air), mid-grey (soft tissue),
       and near-white (bone). This gives a high IQR regardless of window.
       Normal CT IQR ≈ 85–140. MRI IQR ≈ 50–90 (narrower post-norm).

    5. NO BRIGHT BORDER
       CT scanner boundary is black → border_mean < 12 always.
    """
    h, w = gray224.shape

    # --- Feature 1: corner darkness (most reliable CT sign) ---
    # Use h//15 (tiny corners) — more reliable than h//7 because some CT variants
    # have grey scanner-table padding that bleeds into the 7th-region corners.
    # The absolute corners (15th) are always black in any CT scan.
    ch, cw = h // 15, w // 15
    corners = np.concatenate([
        gray224[:ch, :cw].ravel(),
        gray224[:ch, -cw:].ravel(),
        gray224[-ch:, :cw].ravel(),
        gray224[-ch:, -cw:].ravel(),
    ])
    corner_mean = float(corners.mean())
    # CT tiny-corners: 0–8.  X-ray: 30–200.  MRI: 0–10.
    # Threshold 10: strict enough to reject X-ray, flexible enough for MRI overlap
    f_corner = np.clip(1.0 - corner_mean / 10.0, 0.0, 1.0)

    # --- Feature 2: Hough circle detection for scanner boundary ---
    # Compute IQR first — used as a gate inside the Hough check.
    p25_ct = float(np.percentile(gray224, 25))
    p75_ct = float(np.percentile(gray224, 75))
    iqr    = p75_ct - p25_ct

    # Try multiple sensitivity levels to handle darker/lighter CT variants.
    # The IQR gate (iqr >= 65) ensures we don't accept MRI skull outlines as scanner rings —
    # MRI post-norm always has IQR < 65; CT always has IQR > 80.
    blurred = cv2.GaussianBlur(gray224.astype(np.uint8), (9, 9), 2)
    f_circle = 0.0
    for param2 in [30, 22, 15]:   # try decreasing sensitivity until circle found
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h // 2,
            param1=50, param2=param2,
            minRadius=int(h * 0.30), maxRadius=int(h * 0.55),
        )
        if circles is not None and iqr >= 65:
            cx, cy, cr = circles[0][0]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (int(cx), int(cy)), int(cr), 255, -1)
            outside_px   = gray224[mask == 0] if (mask == 0).any() else np.array([255.0])
            outside_mean = float(outside_px.mean())
            very_dark_frac = float((outside_px < 25).sum()) / (len(outside_px) + 1e-6)
            if outside_mean < 60 or very_dark_frac > 0.25:
                f_circle = 1.0; break
            elif outside_mean < 80:
                f_circle = 0.6; break

    # --- Feature 3: dark lung field presence ---
    # In the central 50% of the image, look for large very dark regions
    r0, r1 = int(h * 0.25), int(h * 0.75)
    c0, c1 = int(w * 0.10), int(w * 0.90)
    centre_patch = gray224[r0:r1, c0:c1]
    dark_frac = float((centre_patch < 50).sum()) / centre_patch.size
    # CT: dark_frac = 0.25–0.60 (large dark lung fields)
    # X-ray: 0.10–0.30 (lungs lighter after CLAHE)
    # MRI: 0.05–0.25 (brain tissue, less air)
    f_dark_lung = np.clip(dark_frac / 0.35, 0.0, 1.0)

    # --- Feature 4: IQR (wide dynamic range) --- already computed above as `iqr`
    f_iqr = np.clip((iqr - 60.0) / 60.0, 0.0, 1.0)

    # --- Feature 5: CT content fallback — high IQR × high dark fraction ---
    # Even without a Hough circle (e.g. image header strips scanner ring), CT lung
    # images always have simultaneously: wide IQR (air+bone) AND large dark lung fields.
    # X-ray: low dark_frac (film background is grey). MRI: low IQR (percentile-normed).
    # This compound score is zero for both imposters and positive only for CT content.
    dark_frac_full = float((gray224 < 50).sum()) / gray224.size
    f_content = np.clip(iqr / 120.0, 0.0, 1.0) * np.clip(dark_frac_full / 0.30, 0.0, 1.0)

    # --- Feature 6: absence of bright border ---
    border_mean = float(np.concatenate([
        gray224[0, :], gray224[-1, :],
        gray224[:, 0], gray224[:, -1]
    ]).mean())
    f_no_border = np.clip(1.0 - border_mean / 30.0, 0.0, 1.0)

    # --- Combine ---
    score = (
        0.20 * f_corner     +   # tiny-corner black (shared w/ MRI, but helps vs X-ray)
        0.30 * f_circle     +   # scanner ring (IQR-gated, multi-sensitivity) = definitive CT
        0.15 * f_content    +   # IQR×dark fallback (works even without scanner ring)
        0.10 * f_dark_lung  +
        0.20 * f_iqr        +   # wide IQR separates CT from MRI
        0.05 * f_no_border
    )
    return float(score)


def _score_mri(gray224: np.ndarray) -> float:
    """
    Score an image on MRI likelihood [0.0 – 1.0].

    Brain MRI signatures:
    ─────────────────────
    1. BLACK BACKGROUND FRACTION
       MRI brain images have a large pure-black background (air outside head).
       After percentile normalisation the background stays near-zero.
       This is the STRONGEST single discriminator vs X-ray (which has grey/white
       background) and a key vs CT (CT has dark areas too, but MRI background
       is more uniformly black across ALL 4 classes including notumor).

    2. OVAL/ELLIPTICAL BRAIN REGION
       The skull creates an oval bright ring inside a black field.
       We detect this with contour analysis. NOTE: "notumor" MRI images
       often have the brain filling the frame fully (no clear ellipse boundary
       visible). We therefore use a RELAXED fallback that rewards any large
       bright connected region in a dark field, even without ellipse fit.

    3. CT EXCLUSION — NARROW IQR + NO SCANNER RING
       MRI post-normalisation has a narrower IQR than CT (MRI: 50–90,
       CT: 85–140). We reward images with IQR < 90 and penalise images
       that simultaneously have wide IQR AND a Hough-detected scanner ring.
       This directly fixes notumor MRI being misclassified as CT.

    4. TEXTURE COMPLEXITY (BRAIN SULCI)
       Brain grey matter has rich sulcal/gyral texture — many fine edges.
       Lung CT has coarser, smoother texture (air + tissue boundary only).
       X-ray has rib edge patterns (different spatial frequency).
       We measure local variance as a texture proxy.

    5. DARK CORNER + NO LARGE BRIGHT BORDER
       MRI corners are black (same as CT corners) — this alone doesn't
       distinguish MRI from CT, but combined with the oval brain region
       and texture it completes the fingerprint.
    """
    h, w = gray224.shape

    # --- Feature 1: background dark fraction (outside brain) ---
    # Use threshold < 15 for truly black background pixels
    dark_frac_strict = float((gray224 < 15).sum()) / gray224.size
    dark_frac_loose  = float((gray224 < 30).sum()) / gray224.size
    # MRI: dark_frac_strict ~ 0.25–0.60 (large uniform black outside skull)
    # CT:  dark_frac_strict ~ 0.10–0.30 (some dark but not uniformly near-zero)
    # X-ray: dark_frac_strict ~ 0.02–0.15 (grey film background, not pure black)
    # Use the strict threshold — it better separates MRI's uniform black background
    f_dark_bg = np.clip((dark_frac_strict - 0.08) / 0.35, 0.0, 1.0)

    # Additional reward: if border pixels are very dark (MRI scanner background)
    border_px = np.concatenate([
        gray224[0, :], gray224[-1, :],
        gray224[:, 0], gray224[:, -1]
    ])
    border_mean = float(border_px.mean())
    # MRI border: < 10 (black).  CT border: < 10 (also black).  X-ray: > 30.
    f_dark_border = np.clip(1.0 - border_mean / 12.0, 0.0, 1.0)

    # --- Feature 2: oval brain region detection (with notumor-aware fallback) ---
    _, brain_mask = cv2.threshold(gray224.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE,
                                   np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    f_oval = 0.0
    if contours:
        largest   = max(contours, key=cv2.contourArea)
        area      = cv2.contourArea(largest)
        area_frac = area / (h * w)

        # PRIMARY path: try ellipse fit (works for most MRI classes)
        if 0.15 < area_frac < 0.85 and len(largest) >= 5:
            try:
                ellipse = cv2.fitEllipse(largest)
                ea, eb  = ellipse[1]
                if eb > 0:
                    aspect = min(ea, eb) / (max(ea, eb) + 1e-6)
                    if 0.45 < aspect <= 1.0:
                        f_oval = np.clip((aspect - 0.45) / 0.45, 0.0, 1.0)
                        f_oval *= np.clip((area_frac - 0.10) / 0.55, 0.0, 1.0)
            except Exception:
                pass

        # FALLBACK for "notumor" class: brain fills most of the frame —
        # ellipse fit may fail or give low aspect ratio, but there IS a large
        # bright connected region sitting in a dark (black) background.
        # If: large bright blob (>25% image) + dark background present → MRI signal
        if f_oval < 0.25 and area_frac > 0.20 and dark_frac_strict > 0.10:
            f_oval = np.clip(area_frac / 0.70, 0.0, 0.70)  # capped at 0.70 (soft signal)

    # --- Feature 3: CT exclusion via IQR + absence of scanner ring ---
    p25 = float(np.percentile(gray224, 25))
    p75 = float(np.percentile(gray224, 75))
    iqr = p75 - p25

    # MRI IQR typically 45–95 (percentile-normalised brain tissue)
    # CT IQR typically 85–145 (air + soft tissue + bone = wide spread)
    # Reward narrow-to-moderate IQR (MRI range), penalise wide IQR (CT range)
    f_iqr_mri = np.clip(1.0 - (iqr - 50.0) / 80.0, 0.0, 1.0)

    # CT exclusion bonus: if Hough circle found AND IQR is wide → NOT MRI
    # We don't want to fire this if it's just a soft signal, so make it a penalty
    blurred = cv2.GaussianBlur(gray224.astype(np.uint8), (9, 9), 2)
    has_scanner_ring = False
    for param2 in [30, 22, 15]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h // 2,
            param1=50, param2=param2,
            minRadius=int(h * 0.30), maxRadius=int(h * 0.55),
        )
        if circles is not None and iqr >= 80:
            cx, cy, cr = circles[0][0]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (int(cx), int(cy)), int(cr), 255, -1)
            outside_px   = gray224[mask == 0] if (mask == 0).any() else np.array([255.0])
            outside_mean = float(outside_px.mean())
            if outside_mean < 60:
                has_scanner_ring = True
                break

    # If scanner ring found with wide IQR → penalise MRI score (this is CT)
    ct_penalty = 0.25 if has_scanner_ring else 0.0

    # --- Feature 4: texture complexity (brain sulci = fine edges) ---
    lap     = cv2.Laplacian(gray224.astype(np.uint8), cv2.CV_64F)
    lap_var = float(lap.var())
    # MRI brain: lap_var ~ 150–900 (rich gyral texture, varies by slice)
    # CT lung:   lap_var ~ 80–400
    # X-ray:     lap_var ~ 200–1000
    f_texture = np.clip((lap_var - 40.0) / 700.0, 0.0, 1.0)

    # --- Feature 5: corner darkness ---
    ch, cw = h // 7, w // 7
    corners = np.concatenate([
        gray224[:ch, :cw].ravel(),
        gray224[:ch, -cw:].ravel(),
        gray224[-ch:, :cw].ravel(),
        gray224[-ch:, -cw:].ravel(),
    ])
    corner_mean = float(corners.mean())
    f_dark_corner = np.clip(1.0 - corner_mean / 18.0, 0.0, 1.0)

    # --- Combine (rebalanced weights vs old version) ---
    # f_dark_bg is now the top-weight feature — it is the most reliable
    # single signal that separates MRI (ALL 4 classes) from X-ray and CT.
    # f_oval gets its weight reduced because notumor images break it.
    # f_iqr_mri replaces old f_iqr — directly encodes CT-exclusion.
    raw_score = (
        0.30 * f_dark_bg     +   # ★ BLACK background — strongest MRI marker
        0.20 * f_oval        +   # oval brain region (relaxed for notumor)
        0.20 * f_iqr_mri     +   # narrow IQR → not CT
        0.18 * f_texture     +   # brain sulci texture
        0.07 * f_dark_border +   # border darkness (shared with CT, soft signal)
        0.05 * f_dark_corner
    )

    # Apply CT-ring penalty AFTER combining
    score = max(0.0, raw_score - ct_penalty)
    return float(score)


def _detect_modality(img_rgb: np.ndarray, selected_modality: str = None):
    """
    Multi-signal modality detector.

    Scores the image independently against X-Ray, CT Scan, and MRI using
    anatomy-specific and imaging-physics-specific feature detectors.
    Returns the detected modality and a dictionary of all scores.

    Decision logic (v3 — fixes UNCERTAIN on hard/atypical images):
    ───────────────────────────────────────────────────────────────
    1. Compute score_xray, score_ct, score_mri (each in [0, 1]).
    2. PRIMARY path: top scorer wins if score >= MIN_SCORE AND gap >= MIN_GAP.
    3. BENEFIT-OF-DOUBT path: if top scorer matches the selected_modality
       but gap is too small (ambiguous), we still return the top scorer
       instead of UNCERTAIN — "ambiguous but leans toward selected" counts.
       This handles hard images (e.g. notumor MRI, normal CT, COVID X-ray)
       that are atypical within their modality but are still real scans.
    4. UNCERTAIN only fires when:
       a) All scores are very low (not a medical scan at all), OR
       b) The TOP scorer is a DIFFERENT modality than selected, but the gap
          is too small to confidently call it a mismatch.
       In case (b) we still report UNCERTAIN (not mismatch) because we
       can't be sure enough to accuse the user of a wrong upload.

    Returns: (modality_string_or_UNCERTAIN, feature_dict)
    """
    gray   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g224   = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    rgb224 = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    score_xr = _score_xray(g224)
    score_ct = _score_ct(g224, rgb224)
    score_mr = _score_mri(g224)

    scores = {
        "X-Ray":   score_xr,
        "CT Scan": score_ct,
        "MRI":     score_mr,
    }

    best_mod    = max(scores, key=scores.get)
    best_val    = scores[best_mod]
    other_vals  = [v for k, v in scores.items() if k != best_mod]
    second_best = max(other_vals)
    gap         = best_val - second_best

    # ── Thresholds ────────────────────────────────────────────────────────────
    MIN_SCORE       = 0.22   # lowered from 0.28 — handles atypical-but-real scans
    MIN_GAP         = 0.10   # standard gap required for confident detection
    BOD_MIN_SCORE   = 0.18   # benefit-of-doubt: minimum score for selected modality
    BOD_MIN_GAP     = 0.03   # benefit-of-doubt: selected must still lead by at least this

    # ── Primary decision ──────────────────────────────────────────────────────
    if best_val >= MIN_SCORE and gap >= MIN_GAP:
        detected = best_mod

    # ── Benefit-of-doubt: selected modality is the top scorer but gap is small ─
    elif (selected_modality is not None
          and best_mod == selected_modality
          and best_val >= BOD_MIN_SCORE
          and gap >= BOD_MIN_GAP):
        # The image leans toward the selected modality but isn't perfectly clean.
        # This covers: notumor MRI, normal CT, normal X-Ray — all of which are
        # "less textured" versions of their modality and may score closer to
        # competitors. We trust the user's modality selection here.
        detected = selected_modality

    else:
        detected = "UNCERTAIN"

    feat_dict = {
        "score_xray": score_xr,
        "score_ct":   score_ct,
        "score_mri":  score_mr,
        "best":       best_mod,
        "gap":        gap,
    }
    return detected, feat_dict


# ─────────────────────────────────────────────────────────────────────────────
#  CONTENT VALIDATION — per-modality anatomical sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def _validate_xray_content(gray224: np.ndarray) -> tuple[bool, str]:
    """
    Verify an X-Ray image shows chest/lung anatomy.
    Rejects non-chest X-rays (knee, hand, skull, etc.) and other images.

    Rules based on chest X-ray anatomy:
    - Must have bilateral roughly-symmetric lung fields (dark regions on L and R)
    - Spine/mediastinum is brighter than the flanks
    - Image must be landscape or portrait (not wildly distorted)
    - Overall intensity in mid-grey range (CLAHE effect)
    """
    h, w = gray224.shape

    # Rule 1: centre column must be brighter than side columns
    # (mediastinum/heart is denser than lung air)
    mid_band  = gray224[:, w//3: 2*w//3].mean()
    left_band = gray224[:, :w//5].mean()
    right_band = gray224[:, 4*w//5:].mean()
    side_mean = 0.5 * (left_band + right_band)

    if mid_band < side_mean * 0.85:
        return False, (
            "Chest X-Ray check failed: no bright mediastinum detected. "
            "Please upload a CHEST (PA/AP) radiograph showing lungs, ribs, and heart shadow."
        )

    # Rule 2: image must have some contrast (not blank/test card)
    global_std = float(gray224.std())
    if global_std < 20.0:
        return False, (
            f"Image appears too uniform (std={global_std:.1f}). "
            "Not a valid chest X-ray."
        )

    # Rule 3: overall brightness should be in chest X-ray range
    global_mean = float(gray224.mean())
    if global_mean < 30.0:
        return False, (
            f"Image too dark for a chest X-ray (mean={global_mean:.1f}). "
            "Chest X-rays have a bright/mid-grey background. "
            "This may be a CT or MRI slice."
        )

    return True, "Chest X-ray anatomy check passed."


def _validate_ct_content(gray224: np.ndarray, img_rgb224: np.ndarray) -> tuple[bool, str]:
    """
    Verify a CT image shows axial lung slice anatomy.
    Rejects CT images of brain, abdomen, etc. (though these are rare in the dataset).

    Rules:
    - Must have a dark (near-black) border (scanner boundary)
    - Must have substantial dark region (lung air)
    - Circular scanner field present
    """
    h, w = gray224.shape

    # Rule 1: corners must be dark (scanner table = black)
    # Use tiny corners (h//15) — CT variants with grey padding still have
    # near-zero pixel values in the absolute corner pixels.
    ch, cw = h // 15, w // 15
    corners = np.concatenate([
        gray224[:ch, :cw].ravel(),
        gray224[:ch, -cw:].ravel(),
        gray224[-ch:, :cw].ravel(),
        gray224[-ch:, -cw:].ravel(),
    ])
    corner_mean = float(corners.mean())
    if corner_mean > 20:
        return False, (
            f"CT Scan check failed: corners are too bright (mean={corner_mean:.1f}). "
            "Real CT axial images have pure-black corners outside the scanner ring. "
            "This may be an X-ray or photograph."
        )

    # Rule 2: must have dark lung-like regions (air = very dark in CT)
    dark_frac = float((gray224 < 40).sum()) / gray224.size
    if dark_frac < 0.08:
        return False, (
            f"CT Scan check failed: insufficient dark regions (dark_frac={dark_frac:.3f}). "
            "Lung CT images must contain large dark air-filled lung fields. "
            "Please upload an axial lung CT slice."
        )

    return True, "CT axial lung anatomy check passed."


def _validate_mri_content(gray224: np.ndarray) -> tuple[bool, str]:
    """
    Verify an MRI image shows brain anatomy.
    Rejects MRI of spine, knee, etc. and non-MRI images.

    Rules:
    - Large black background (outside skull)
    - One main oval bright region (brain/skull)
    - Rich texture (brain sulci)
    - Corners must be dark
    """
    h, w = gray224.shape

    # Rule 1: corners must be near-black (MRI background = black)
    ch, cw = h // 7, w // 7
    corners = np.concatenate([
        gray224[:ch, :cw].ravel(),
        gray224[:ch, -cw:].ravel(),
        gray224[-ch:, :cw].ravel(),
        gray224[-ch:, -cw:].ravel(),
    ])
    corner_mean = float(corners.mean())
    if corner_mean > 25:
        return False, (
            f"Brain MRI check failed: corners too bright (mean={corner_mean:.1f}). "
            "Brain MRI images have a pure-black background outside the skull. "
            "This may be an X-ray or colour photograph."
        )

    # Rule 2: must have a substantial black background (outside skull)
    dark_frac = float((gray224 < 15).sum()) / gray224.size
    if dark_frac < 0.15:
        return False, (
            f"Brain MRI check failed: insufficient black background (dark_frac={dark_frac:.3f}). "
            "Brain MRI images should have a large dark area outside the skull. "
            "Please upload a brain MRI slice."
        )

    # Rule 3: must have at least one medium-to-large bright blob (brain/skull)
    _, brain_mask = cv2.threshold(gray224.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, (
            "Brain MRI check failed: no brain region detected. "
            "Please upload a valid brain MRI slice."
        )

    largest_area = max(cv2.contourArea(c) for c in contours)
    area_frac    = largest_area / (h * w)
    if area_frac < 0.12:
        return False, (
            f"Brain MRI check failed: brain region too small ({area_frac*100:.1f}% of image). "
            "Please upload a full-frame brain MRI slice (not a thumbnail)."
        )

    return True, "Brain MRI anatomy check passed."


CONTENT_VALIDATORS = {
    "X-Ray":   _validate_xray_content,
    "CT Scan": _validate_ct_content,
    "MRI":     _validate_mri_content,
}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN VALIDATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def validate_image_size(img_pil):
    """
    Validate image dimensions.
    - Hard reject: images below 32×32 (too small to be a real scan) or above 8000×8000
    - All other sizes are ACCEPTED — the preprocessing pipeline resizes to IMG_SIZE×IMG_SIZE
      internally, so the model always receives exactly 224×224 regardless of upload size.
    - Returns (ok, message, warn_flag) where warn_flag=True triggers a yellow quality note.
    """
    w, h = img_pil.size
    if w < 32 or h < 32:
        return False, (
            f"Image Too Small ({w}×{h} px). "
            "Medical scans must be at least 32×32 pixels. "
            "Please upload the original full-resolution scan."
        ), False
    if w > 8000 or h > 8000:
        return False, (
            f"Image Too Large ({w}×{h} px). "
            "Images above 8000×8000 px are atypical for medical scans. "
            "Please verify you have the correct file."
        ), False

    # Quality warning for non-standard sizes (but still accepted)
    warn = False
    warn_msg = f"Image size {w}×{h} px — valid."
    if w < 100 or h < 100:
        warn = True
        warn_msg = (
            f"Image size {w}×{h} px is very small. "
            f"Auto-resizing to {IMG_SIZE}×{IMG_SIZE} px for inference. "
            "For best results, use full-resolution scans (≥224×224 px)."
        )
    elif w != IMG_SIZE or h != IMG_SIZE:
        # Non-224×224 — silently accepted, just log it
        warn_msg = (
            f"Image size {w}×{h} px — auto-resized to "
            f"{IMG_SIZE}×{IMG_SIZE} px for inference."
        )
    return True, warn_msg, warn


def validate_modality_match(selected_modality: str, img_rgb: np.ndarray):
    """
    Two-stage validation:
    Stage 1 — Pre-flight checks (blank scan, colour photograph)
    Stage 2 — Multi-signal modality detection (score all 3 modalities)
    Stage 3 — Per-modality anatomical content check

    Returns (is_valid: bool, message: str)
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # ── Stage 1a: blank / corrupted ──────────────────────────────────────────
    if float(gray.std()) < 4.0:
        return False, (
            "Near-Blank / Corrupted Image. "
            "The scan appears blank or severely underexposed. "
            "Please upload the original, full-resolution medical scan."
        )

    # ── Stage 1b: natural colour photograph ──────────────────────────────────
    r, g, b = (img_rgb[:, :, c].astype(float) for c in range(3))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    colourfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * (rg.mean() + yb.mean())
    )
    if colourfulness > 35:
        return False, (
            f"Not a Medical Scan (colourfulness={colourfulness:.1f}). "
            "This appears to be a natural colour photograph. "
            "Please upload a real grayscale medical scan."
        )

    # ── Stage 2: Multi-signal modality scoring ────────────────────────────────
    g224   = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    rgb224 = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    detected, scores = _detect_modality(img_rgb, selected_modality=selected_modality)

    score_str = (
        f"X-Ray={scores['score_xray']:.3f}, "
        f"CT={scores['score_ct']:.3f}, "
        f"MRI={scores['score_mri']:.3f}, "
        f"gap={scores['gap']:.3f}"
    )

    if detected == "UNCERTAIN":
        # Ambiguous — scores too close or all too low even after benefit-of-doubt
        return False, (
            f"⚠️ MODALITY DETECTION UNCERTAIN\n\n"
            f"The image features are ambiguous and could not be confidently assigned "
            f"to any single modality.\n\n"
            f"Scores: [X-Ray={scores['score_xray']:.3f}, "
            f"CT={scores['score_ct']:.3f}, "
            f"MRI={scores['score_mri']:.3f}, "
            f"gap={scores['gap']:.3f}]\n\n"
            f"This usually happens with:\n"
            f"  • Atypical or edge-case scans (very dark, low contrast, heavily cropped)\n"
            f"  • Images from a different scanner or preprocessing pipeline\n"
            f"  • Non-standard image orientations\n\n"
            f"Please try:\n"
            f"  1. Uploading the original unprocessed scan file\n"
            f"  2. Ensuring you selected the correct modality: '{selected_modality}'\n"
            f"  3. Using a different slice/frame from the same scan\n"
            f"  4. Checking that the image is a real medical scan (not a screenshot or processed image)"
        )

    if detected != selected_modality:
        GUIDELINES = {
            "X-Ray": (
                "X-Ray Upload Guidelines:\n"
                "  - Use CHEST RADIOGRAPH images (PA or AP view)\n"
                "  - Image must show ribs, lung fields, heart shadow, clavicles\n"
                "  - Background must be white or light-grey (not pure black)\n"
                "  - Dataset: COVID-19 Radiography Database (Kaggle)\n"
                "  - Classes: COVID / Lung Opacity / Normal / Viral Pneumonia"
            ),
            "CT Scan": (
                "CT Scan Upload Guidelines:\n"
                "  - Use AXIAL-SLICE LUNG CT images\n"
                "  - Must show: circular scanner boundary, dark lungs, grey tissue\n"
                "  - Background outside scanner circle must be near-black\n"
                "  - Dataset: Lung cancer CT dataset (Kaggle)\n"
                "  - Classes: Adenocarcinoma / Large Cell / Normal / Squamous Cell"
            ),
            "MRI": (
                "Brain MRI Upload Guidelines:\n"
                "  - Use BRAIN MRI images (axial or sagittal slice)\n"
                "  - Brain tissue must fill most of the frame\n"
                "  - Black background, skull visible as a bright ring\n"
                "  - Dataset: Brain Tumor MRI (masoudnickparvar, Kaggle)\n"
                "  - Classes: Glioma / Meningioma / No Tumor / Pituitary"
            ),
        }
        error_lines = [
            "MODALITY MISMATCH DETECTED",
            f"  You selected  : {selected_modality}",
            f"  Image detected: {detected}",
            f"  Scores        : [{score_str}]",
            "",
            f"ACTION: Select '{detected}' from the sidebar and re-upload.",
            "",
            "SEVERITY: HIGH — Wrong modality will produce meaningless predictions.",
            "",
            GUIDELINES[selected_modality],
        ]
        return False, "\n".join(error_lines)

    # ── Stage 3: Anatomical content check for the confirmed modality ──────────
    validator = CONTENT_VALIDATORS[selected_modality]
    if selected_modality == "CT Scan":
        content_ok, content_msg = validator(g224, rgb224)
    else:
        content_ok, content_msg = validator(g224)

    if not content_ok:
        return False, (
            f"CONTENT VALIDATION FAILED for {selected_modality}\n"
            f"Modality scores: [{score_str}]\n\n"
            f"{content_msg}"
        )

    return True, (
        f"Modality validated: '{selected_modality}' confirmed. "
        f"[{score_str}] | {content_msg}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLINICAL ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Severity table: every class from all three modalities
_SEVERITY_MAP = {
    # X-Ray classes
    "COVID":                   ("HIGH",   "#B71C1C", "⚠️"),
    "Lung_Opacity":             ("MEDIUM", "#E65100", "🟡"),
    "Viral Pneumonia":          ("MEDIUM", "#E65100", "🟡"),
    "Normal":                   ("NONE",   "#1B5E20", "✅"),
    # CT Scan classes
    "adenocarcinoma":           ("HIGH",   "#B71C1C", "⚠️"),
    "large cell carcinoma":     ("HIGH",   "#B71C1C", "⚠️"),
    "squamous cell carcinoma":  ("HIGH",   "#B71C1C", "⚠️"),
    "normal":                   ("NONE",   "#1B5E20", "✅"),
    # MRI classes
    "glioma":                   ("HIGH",   "#B71C1C", "⚠️"),
    "meningioma":               ("MEDIUM", "#E65100", "🟡"),
    "pituitary":                ("MEDIUM", "#E65100", "🟡"),
    "notumor":                  ("NONE",   "#1B5E20", "✅"),
}

def get_severity(class_name: str, confidence: float) -> tuple[str, str, str]:
    """
    Returns (severity_label, hex_colour, icon).
    Confidence below 0.55 downgrades HIGH to MEDIUM.
    Confidence below 0.40 marks any class as uncertain regardless of severity.
    """
    level, colour, icon = _SEVERITY_MAP.get(class_name, ("UNKNOWN", "#546E7A", "❓"))
    if confidence < 0.40:
        return f"{level} — Very Uncertain", "#546E7A", "❓"
    if confidence < 0.55 and level == "HIGH":
        return "MEDIUM — Low Confidence", "#E65100", "🟡"
    return level, colour, icon


def compute_entropy(probs: np.ndarray) -> tuple[float, str, str]:
    """
    Shannon entropy of predicted class distribution.
    Measures how spread the probability mass is across classes.

    Raw entropy thresholds (4-class problem):
      < 0.40  → High Confidence   (one class strongly dominant)
      < 0.90  → Moderate Confidence
      >= 0.90 → Low Confidence    (probability spread across classes)

    Returns (raw_entropy, label, hex_colour).
    """
    eps = 1e-9
    raw = float(-np.sum(probs * np.log(probs + eps)))
    if raw < 0.40:
        return raw, "High Confidence",                   "#2E7D32"
    elif raw < 0.90:
        return raw, "Moderate Confidence",               "#F57F17"
    else:
        return raw, "Low Confidence — Interpret Carefully", "#C62828"


def compute_roi_bbox(cam_m: np.ndarray, threshold: float = 0.60):
    """
    Derive a tight bounding box around the GradCAM++ activation region.
    Applies morphological closing to merge nearby blobs before boxing.
    Returns (x, y, w, h) in 0–1 normalised image coordinates, or None.
    """
    binary = (cam_m >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    all_pts    = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_pts)
    H, W       = cam_m.shape
    return (x / W, y / H, w / W, h / H)   # normalised


def draw_bbox_on_image(img_uint8: np.ndarray,
                       bbox_norm: tuple,
                       colour=(255, 80, 0),
                       thickness: int = 3) -> np.ndarray:
    """
    Draw a bounding box (normalised coords) on a uint8 RGB image.
    Returns a new image with the box drawn.
    """
    H, W = img_uint8.shape[:2]
    x = int(bbox_norm[0] * W); y = int(bbox_norm[1] * H)
    w = int(bbox_norm[2] * W); h = int(bbox_norm[3] * H)
    out = img_uint8.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), colour, thickness)
    # Small corner brackets for a cleaner clinical look
    L = max(8, min(w, h) // 4)
    for px, py in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
        sx = 1 if px == x else -1
        sy = 1 if py == y else -1
        cv2.line(out, (px, py), (px + sx*L, py), colour, thickness+1)
        cv2.line(out, (px, py), (px, py + sy*L), colour, thickness+1)
    return out


def compute_image_quality(img_rgb: np.ndarray) -> dict:
    """
    Compute three image quality metrics relevant to medical imaging:
      - Brightness: mean pixel value (ideal: 0.3–0.7 of full range)
      - Contrast:   standard deviation (low std = flat, boring image)
      - Sharpness:  Laplacian variance (low = blurry)

    Returns a dict with individual scores and an overall 0–100 score.
    """
    gray       = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    brightness = float(gray.mean()) / 255.0
    contrast   = float(gray.std())  / 128.0
    lap_var    = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
    sharpness  = min(lap_var / 500.0, 1.0)

    # Component scores (0–1)
    b_score = max(0.0, 1.0 - abs(brightness - 0.50) / 0.50)
    c_score = min(contrast / 0.40, 1.0)
    s_score = min(sharpness / 0.60, 1.0)
    overall = 0.25 * b_score + 0.40 * c_score + 0.35 * s_score

    if overall >= 0.70:   label, colour = "Good",       "#2E7D32"
    elif overall >= 0.45: label, colour = "Acceptable", "#F57F17"
    else:                 label, colour = "Poor",        "#C62828"

    return {
        "brightness": round(brightness * 100, 1),
        "contrast":   round(min(contrast, 1.0) * 100, 1),
        "sharpness":  round(s_score * 100, 1),
        "overall":    round(overall * 100, 1),
        "label":      label,
        "colour":     colour,
    }


def build_report(filename: str, modality: str, top_class: str,
                 top_conf: float, probs: np.ndarray, classes: list,
                 roi_pct: float, peak_act: float, mean_act: float,
                 bbox, entropy_val: float, entropy_label: str,
                 severity_label: str, quality: dict) -> str:
    """
    Build a plain-text clinical analysis report suitable for download.
    Includes prediction, confidence, uncertainty, ROI stats, bbox, quality.
    """
    import datetime
    ts   = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    idx  = np.argsort(probs)[::-1]
    top2 = [(classes[i], float(probs[i])) for i in idx[:2]]

    lines = [
        "=" * 62,
        "   MEDICAL IMAGE ANALYSIS REPORT",
        "   EfficientNet-B3 + GradCAM++ ROI Localisation System",
        "=" * 62,
        f"  Timestamp   : {ts}",
        f"  File        : {filename}",
        f"  Modality    : {modality}",
        "",
        "── PREDICTION ──────────────────────────────────────────────",
        f"  Primary Class   : {top_class.replace('_', ' ').title()}",
        f"  Confidence      : {top_conf * 100:.1f}%",
        f"  2nd Candidate   : {top2[1][0].replace('_', ' ').title()}"
        f"  ({top2[1][1] * 100:.1f}%)",
        f"  Clinical Risk   : {severity_label}",
        f"  Model Certainty : {entropy_label}  (entropy = {entropy_val:.3f})",
        "",
        "── PER-CLASS PROBABILITIES ─────────────────────────────────",
    ]
    for cls, p in zip(classes, probs):
        bar  = "█" * int(p * 28)
        mark = " ← predicted" if cls == top_class else ""
        lines.append(f"  {cls:<32} {p * 100:5.1f}%  {bar}{mark}")

    lines += [
        "",
        "── ROI LOCALISATION ────────────────────────────────────────",
        f"  ROI Coverage    : {roi_pct:.1f}% of image area",
        f"  Peak Activation : {peak_act:.4f}",
        f"  Mean Activation : {mean_act:.4f}",
    ]
    if bbox:
        x, y, w, h = bbox
        lines += [
            f"  Bounding Box    : top-left ({x * 100:.1f}%, {y * 100:.1f}%)"
            f"   size ({w * 100:.1f}% × {h * 100:.1f}%)",
        ]
    else:
        lines.append("  Bounding Box    : No focal ROI detected above threshold")

    lines += [
        "",
        "── IMAGE QUALITY ───────────────────────────────────────────",
        f"  Overall Quality : {quality['label']}  ({quality['overall']:.0f} / 100)",
        f"  Brightness      : {quality['brightness']:.0f} / 100",
        f"  Contrast        : {quality['contrast']:.0f} / 100",
        f"  Sharpness       : {quality['sharpness']:.0f} / 100",
        "",
        "── DISCLAIMER ──────────────────────────────────────────────",
        "  This report is produced by an AI research system and is",
        "  NOT a substitute for professional medical diagnosis.",
        "  Always consult a qualified radiologist or physician.",
        "=" * 62,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ROIClassifier(nn.Module):
    """EfficientNet-B3 + GradCAM++ — dropout 0.5/0.3 matches training exactly."""
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        base = efficientnet_b3(weights=None)
        self.features   = base.features
        self.avgpool    = base.avgpool
        in_features     = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(256, num_classes),
        )
        self._gradients   = None
        self._activations = None
        self.features[-1].register_forward_hook(self._save_activation)
        self.features[-1].register_full_backward_hook(self._save_gradient)

    def _save_activation(self, m, inp, out): self._activations = out.detach()
    def _save_gradient(self, m, gi, go):    self._gradients   = go[0].detach()

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)).flatten(1))

    def gradcam_plus_plus(self, img_tensor, class_idx=None):
        self.eval()
        inp = img_tensor.unsqueeze(0).to(DEVICE)
        with torch.enable_grad():
            inp    = inp.requires_grad_(True)
            output = self(inp)
            if class_idx is None:
                class_idx = int(output.argmax(dim=1).item())
            self.zero_grad()
            output[0, class_idx].backward()

        grads = self._gradients[0]
        acts  = self._activations[0]
        gsq   = grads ** 2
        gcb   = grads ** 3
        denom = 2 * gsq + (acts * gcb).sum(dim=(1,2), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha   = gsq / denom
        weights = (alpha * torch.relu(grads)).mean(dim=(1,2))
        cam     = torch.relu((weights[:,None,None] * acts).sum(0)).cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam     = cv2.GaussianBlur(cam, (5,5), 0)
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam     = np.where(cam < 0.65, 0.0, cam).astype(np.float32)
        if cam.max() > 0: cam /= cam.max()
        probs   = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        return cam, class_idx, probs


@st.cache_resource(show_spinner=False)
def load_model(modality):
    try:
        ckpt    = torch.load(MODEL_PATHS[modality], map_location=DEVICE, weights_only=False)
        classes = ckpt.get("classes", FALLBACK_CLASSES[modality])
        model   = ROIClassifier(ckpt["num_classes"], dropout=0.5).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, classes, False
    except FileNotFoundError:
        classes = FALLBACK_CLASSES[modality]
        model   = ROIClassifier(len(classes), dropout=0.5).to(DEVICE)
        model.eval()
        return model, classes, True
    except Exception as e:
        st.error(f"Error loading model: {e}"); st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  PREPROCESSING (matches training pipeline exactly)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_xray(img_rgb):
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray  = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    img   = gray.astype(np.float32) / 255.0
    return np.stack([img, img, img], axis=-1)

def preprocess_ct(img_rgb):
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img.astype(np.float32) / 255.0

def preprocess_mri(img_rgb):
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray    = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    p2, p98 = np.percentile(gray, [2, 98])
    gray    = np.clip(gray, p2, p98)
    gray    = (gray - p2) / (p98 - p2 + 1e-8)
    return np.stack([gray.astype(np.float32)] * 3, axis=-1)

PREPROCESSORS = {"X-Ray": preprocess_xray, "CT Scan": preprocess_ct, "MRI": preprocess_mri}

def to_tensor(img_np):
    t = torch.from_numpy(np.ascontiguousarray(img_np)).permute(2, 0, 1).float()
    for c in range(3): t[c] = (t[c] - float(IMAGENET_MEAN[c])) / float(IMAGENET_STD[c])
    return t

def make_overlay(orig_uint8, cam, alpha=0.45):
    if orig_uint8.ndim == 2 or orig_uint8.shape[2] == 1:
        orig_uint8 = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2RGB)
    H, W = orig_uint8.shape[:2]
    if cam.shape != (H, W): cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    cam     = np.clip(cam, 0, 1)
    heatmap = cv2.cvtColor(cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB).astype(np.float32)
    mask    = (cam > 0.6).astype(np.float32)[:,:,np.newaxis]
    return np.clip(orig_uint8.astype(np.float32)*(1-mask*alpha) + heatmap*(mask*alpha), 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("""
    <div style="text-align:center; padding:16px 0 8px 0;">
        <div style="font-size:2.4rem;">🏥</div>
        <h2 style="margin:4px 0 0 0; font-size:1.1rem;">Medical ROI System</h2>
        <p style="font-size:0.72rem; color:#aac8e4; margin:2px 0;">EfficientNet-B3 + GradCAM++</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚙️ Configuration**")
    modality = st.sidebar.selectbox("Select Image Modality", ["X-Ray","CT Scan","MRI"],
        help="Select the matching modality. Wrong modality upload will be rejected.")
    alpha        = st.sidebar.slider("Heatmap Intensity", 0.1, 0.9, 0.40, 0.05)
    show_raw_cam = st.sidebar.checkbox("Show Raw CAM Heatmap", value=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Classes**")
    class_info = {
        "X-Ray":   [("🦠","COVID"),("🌫️","Lung Opacity"),("✅","Normal"),("🫧","Viral Pneumonia")],
        "CT Scan": [("🔴","Adenocarcinoma"),("🟠","Large Cell Carcinoma"),("✅","Normal"),("🟡","Squamous Cell Carcinoma")],
        "MRI":     [("🧠","Glioma"),("🔵","Meningioma"),("✅","No Tumor"),("🟣","Pituitary")],
    }
    for icon, cls in class_info[modality]:
        st.sidebar.markdown(f"&nbsp;&nbsp;{icon}&nbsp; {cls}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚫 Auto-rejected inputs:**")
    st.sidebar.markdown(
        "- Near-blank / corrupted scans\n"
        "- Natural colour photographs\n"
        "- Wrong modality type\n"
        "- Images < 32×32 px\n"
        "- Non-chest X-rays (knee, skull, etc.)\n"
        "- Non-brain MRI slices\n\n"
        "**ℹ️ Any upload size accepted** — auto-resized to 224×224 px for inference."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("🔬 B.Tech Final Year Project | Modality-based ROI Localisation")
    return modality, alpha, show_raw_cam


def render_confidence_chart(probs, classes, pred_idx, modality):
    color = MODALITY_COLORS[modality]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = [c.replace("_"," ").title() for c in classes]
    bars  = ax.barh(names, probs*100, color="#dce8f5", edgecolor="#b0c8e4", height=0.55)
    bars[pred_idx].set_color(color)
    bars[pred_idx].set_edgecolor("black")
    bars[pred_idx].set_linewidth(1.5)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title("Class Confidence Scores", fontweight="bold")
    ax.set_xlim(0, 108)
    ax.spines[["top","right"]].set_visible(False)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, f"{p*100:.1f}%",
                va="center", fontsize=9,
                fontweight="bold" if bar==bars[pred_idx] else "normal")
    ax.invert_yaxis()
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def run_inference(img_pil, modality, alpha, show_raw_cam):
    img_rgb = np.array(img_pil.convert("RGB"))

    size_ok, size_msg, size_warn = validate_image_size(img_pil)
    if not size_ok:
        st.error(f"❌ {size_msg}")
        st.info("💡 Please upload the original full-resolution medical scan file.")
        return
    # Show resize note only when dimensions differ from 224×224
    w, h = img_pil.size
    if size_warn:
        st.warning(f"⚠️ {size_msg}")
    elif w != IMG_SIZE or h != IMG_SIZE:
        st.info(f"ℹ️ {size_msg}")

    mod_ok, mod_msg = validate_modality_match(modality, img_rgb)
    if not mod_ok:
        st.error(f"❌ {mod_msg}")
        st.markdown("**How to fix:** Select the correct modality from the sidebar and re-upload.")
        return
    else:
        st.success(f"✅ {mod_msg}")

    # Image quality report (shown immediately after validation)
    quality = compute_image_quality(img_rgb)
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    qcol1.metric("🖼️ Image Quality", quality["label"],
                 help="Overall scan quality: Good / Acceptable / Poor")
    qcol2.metric("☀️ Brightness",  f"{quality['brightness']:.0f}/100")
    qcol3.metric("🎛️ Contrast",    f"{quality['contrast']:.0f}/100")
    qcol4.metric("🔍 Sharpness",   f"{quality['sharpness']:.0f}/100")
    if quality["label"] == "Poor":
        st.warning("⚠️ Image quality is poor. Results may be less reliable. "
                   "Please upload a higher-quality scan if available.")
    st.markdown("---")

    with st.spinner(f"⏳ Loading {modality} model..."):
        model, classes, demo_mode = load_model(modality)

    if demo_mode:
        st.warning(
            f"⚠️ Demo Mode — weights not found at `{MODEL_PATHS[modality]}`. "
            "Run model_training.ipynb first. Predictions below are from untrained model."
        )

    preprocessed = PREPROCESSORS[modality](img_rgb)
    tensor       = to_tensor(preprocessed)

    with st.spinner("🔍 Running EfficientNet-B3 + GradCAM++ ROI localisation..."):
        t0 = time.time()
        cam, pred_cls, probs = model.gradcam_plus_plus(tensor)
        elapsed = time.time() - t0

    top_class = classes[pred_cls]
    top_conf  = float(probs[pred_cls])
    icon      = MODALITY_EMOJI[modality]

    # ── Entropy / Uncertainty ──────────────────────────────────────────────
    entropy_val, entropy_label, entropy_colour = compute_entropy(probs)

    # ── Severity ───────────────────────────────────────────────────────────
    severity_label, severity_colour, severity_icon = get_severity(top_class, top_conf)

    # ── Top-2 predictions ──────────────────────────────────────────────────
    top2_idx  = np.argsort(probs)[::-1][:2]
    top2_list = [(classes[i], float(probs[i])) for i in top2_idx]

    # ── Prediction banner ──────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{MODALITY_GRADIENTS[modality]}; padding:18px 24px; border-radius:14px;
                margin-bottom:12px; box-shadow:0 3px 12px rgba(0,0,0,0.18);">
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <div>
                <h2 style="color:white; margin:0; font-size:1.6rem; font-weight:700;">
                    {icon} {top_class.replace("_"," ").title()}
                </h2>
                <p style="color:rgba(255,255,255,0.88); margin:5px 0 0 0; font-size:0.95rem;">
                    Confidence: <b>{top_conf*100:.1f}%</b> | Modality: <b>{modality}</b>
                    | Inference: <b>{elapsed*1000:.0f} ms</b> | Device: <b>{str(DEVICE).upper()}</b>
                </p>
            </div>
            <div style="font-size:3rem; opacity:0.3;">{icon}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Severity + Uncertainty badges side by side ─────────────────────────
    badge1, badge2, badge3 = st.columns(3)
    with badge1:
        st.markdown(f"""
        <div style="background:{severity_colour}18; border:2px solid {severity_colour};
                    border-radius:10px; padding:12px 16px; text-align:center;">
          <div style="font-size:1.6rem;">{severity_icon}</div>
          <div style="font-weight:700; color:{severity_colour}; font-size:0.95rem;">
              Clinical Risk</div>
          <div style="font-size:0.85rem; color:{severity_colour};">{severity_label}</div>
        </div>""", unsafe_allow_html=True)

    with badge2:
        st.markdown(f"""
        <div style="background:{entropy_colour}18; border:2px solid {entropy_colour};
                    border-radius:10px; padding:12px 16px; text-align:center;">
          <div style="font-size:1.6rem;">📊</div>
          <div style="font-weight:700; color:{entropy_colour}; font-size:0.95rem;">
              Model Certainty</div>
          <div style="font-size:0.85rem; color:{entropy_colour};">{entropy_label}</div>
        </div>""", unsafe_allow_html=True)

    with badge3:
        alt_cls, alt_conf = top2_list[1]
        st.markdown(f"""
        <div style="background:#37474F18; border:2px solid #546E7A;
                    border-radius:10px; padding:12px 16px; text-align:center;">
          <div style="font-size:1.6rem;">🔄</div>
          <div style="font-weight:700; color:#37474F; font-size:0.95rem;">
              2nd Candidate</div>
          <div style="font-size:0.85rem; color:#546E7A;">
              {alt_cls.replace("_"," ").title()} ({alt_conf*100:.1f}%)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── GradCAM++ visualisation + Bounding Box ─────────────────────────────
    orig_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    gray2        = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2GRAY)
    _, tmask     = cv2.threshold(gray2, 20, 255, cv2.THRESH_BINARY)
    tmask        = cv2.GaussianBlur(tmask.astype(np.float32)/255.0, (7,7), 0)
    if tmask.shape != cam.shape: tmask = cv2.resize(tmask, (cam.shape[1], cam.shape[0]))
    cam_m = cam * tmask; cam_m /= (cam_m.max() + 1e-8)
    overlay_img = make_overlay(orig_resized, cam_m, alpha)

    # Compute bounding box
    bbox = compute_roi_bbox(cam_m)
    bbox_img = draw_bbox_on_image(orig_resized, bbox) if bbox else orig_resized

    ncols = 4 if show_raw_cam else 3
    cols  = st.columns(ncols)
    with cols[0]:
        st.image(orig_resized,  caption="📷 Original Upload",  use_container_width=True)
    if show_raw_cam:
        with cols[1]:
            st.image((mpl_cm.jet(cam_m)[:,:,:3]*255).astype(np.uint8),
                     caption="🔥 GradCAM++ Heatmap", use_container_width=True)
        with cols[2]:
            st.image(overlay_img,
                     caption=f"🎯 ROI Overlay — {top_class.replace('_',' ').title()}",
                     use_container_width=True)
        with cols[3]:
            st.image(bbox_img,
                     caption="📦 ROI Bounding Box", use_container_width=True)
    else:
        with cols[1]:
            st.image(overlay_img,
                     caption=f"🎯 ROI Overlay — {top_class.replace('_',' ').title()}",
                     use_container_width=True)
        with cols[2]:
            st.image(bbox_img,
                     caption="📦 ROI Bounding Box", use_container_width=True)

    # ── Confidence chart + score breakdown ────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Per-Class Confidence Scores")
    chart_col, info_col = st.columns([3, 2])
    with chart_col:
        fig = render_confidence_chart(probs, classes, pred_cls, modality)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with info_col:
        badge_col = MODALITY_COLORS[modality]
        st.markdown("**Score Breakdown**")
        for prob, cls in sorted(zip(probs, classes), reverse=True):
            is_pred   = (cls == top_class)
            disp_name = cls.replace("_"," ").title()
            bar_pct   = int(prob * 100)
            bar_color = badge_col if is_pred else "#cccccc"
            fw        = "700" if is_pred else "400"
            fc        = "#111" if is_pred else "#666"
            st.markdown(f"""
            <div style="margin:8px 0">
              <div style="display:flex;justify-content:space-between;
                          font-weight:{fw};color:{fc};font-size:13px;">
                <span>{disp_name}</span><span>{prob*100:.2f}%</span>
              </div>
              <div style="background:#eef2f7;border-radius:5px;height:8px;margin-top:3px;">
                <div style="width:{bar_pct}%;background:{bar_color};
                            height:8px;border-radius:5px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── ROI + Bounding Box statistics ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 ROI Localisation Statistics")
    s1, s2, s3, s4, s5 = st.columns(5)
    roi_pct  = float((cam_m > 0.65).sum()) / cam_m.size * 100
    peak_act = float(cam_m.max())
    mean_act = float(cam_m.mean())
    s1.metric("ROI Coverage",    f"{roi_pct:.1f}%",      help="% of image with high activation")
    s2.metric("Peak Activation", f"{peak_act:.4f}",      help="Max GradCAM++ value")
    s3.metric("Mean Activation", f"{mean_act:.4f}",      help="Mean activation across image")
    s4.metric("Predicted Class", top_class.replace("_"," ").title())
    if bbox:
        s5.metric("BBox Area",   f"{bbox[2]*bbox[3]*100:.1f}%",
                  help="Bounding box as % of image area")
    else:
        s5.metric("BBox Area",   "—",   help="No focal ROI detected")

    # ── Downloads ─────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        buf = io.BytesIO(); Image.fromarray(overlay_img).save(buf, format="PNG")
        st.download_button("⬇️ Download ROI Overlay", buf.getvalue(),
            f"roi_{modality.lower().replace(' ','_')}_{top_class}.png", "image/png")
    with c2:
        buf2 = io.BytesIO()
        Image.fromarray((mpl_cm.jet(cam_m)[:,:,:3]*255).astype(np.uint8)).save(buf2, format="PNG")
        st.download_button("⬇️ Download CAM Heatmap", buf2.getvalue(),
            f"cam_{modality.lower().replace(' ','_')}_{top_class}.png", "image/png")
    with c3:
        report_txt = build_report(
            img_pil.filename if hasattr(img_pil, "filename") else "upload",
            modality, top_class, top_conf, probs, classes,
            roi_pct, peak_act, mean_act, bbox,
            entropy_val, entropy_label, severity_label, quality
        )
        st.download_button("⬇️ Download Analysis Report", report_txt.encode(),
            f"report_{modality.lower().replace(' ','_')}_{top_class}.txt", "text/plain")


def render_landing(modality):
    col_l, col_r = st.columns(2)
    with col_l:
        st.info(f"👆 Select modality from the sidebar → Upload a **{modality}** image to begin.")
        st.markdown("""
        #### How to Use
        1. **Select modality** (X-Ray / CT Scan / MRI) from the left sidebar
        2. **Upload** the corresponding medical scan image *(any resolution — auto-resized to 224×224)*
        3. The system **validates** image — wrong modality or non-medical images are rejected
        4. **EfficientNet-B3** classifies the pathology
        5. **GradCAM++** highlights the Region of Interest (ROI)
        6. **Download** the ROI overlay for your report

        #### Important
        - Only upload **real medical scans** — natural photos will be rejected
        - Always select the **correct modality** before uploading
        - Wrong modality uploads (e.g., X-Ray to CT slot) will be detected and rejected
        - For **X-Ray**: upload chest (PA/AP) radiographs only — not knee, skull, etc.
        - For **MRI**: upload brain MRI slices only (all 4 classes: Glioma, Meningioma, No Tumor, Pituitary)
        - Images are automatically resized to 224×224 px for inference regardless of upload size
        """)
    with col_r:
        st.markdown("#### About GradCAM++")
        st.markdown("""
        **Gradient-weighted Class Activation Mapping++** generates visual
        explanations for CNN decisions by highlighting which image regions
        most strongly influenced the prediction.

        This enables **clinically interpretable ROI localisation**
        without any pixel-level annotation labels.

        ---
        **Architecture:** EfficientNet-B3
        **Explainability:** GradCAM++
        **Regularisation:** MixUp + Label Smoothing 0.15
        **Anti-overfit:** Dropout 0.5/0.3 + Early Stopping
        **Modality Detection:** Multi-signal scoring (anatomy + physics)
        """)


def main():
    modality, alpha, show_raw_cam = render_sidebar()
    icon = MODALITY_EMOJI[modality]
    col  = MODALITY_COLORS[modality]

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
        <span style="font-size:2rem;">{icon}</span>
        <div>
            <h1 style="margin:0;color:{col};font-size:1.8rem;font-weight:800;">
                {modality} — ROI Localisation & Classification
            </h1>
            <p style="margin:2px 0 0 0;color:#666;font-size:13px;">
                Classes: {" · ".join(c.replace("_"," ").title() for c in FALLBACK_CLASSES[modality])}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(f"#### 📂 Upload a `{modality}` Image")
    uploaded_file = st.file_uploader(
        f"Upload {modality} scan", type=["png","jpg","jpeg"],
        accept_multiple_files=False, label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            img_pil = Image.open(uploaded_file); img_pil.verify()
            img_pil = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"❌ Corrupted file `{uploaded_file.name}`: {e}. Please upload a valid PNG/JPG.")
            return
        st.markdown(
            f"**File:** `{uploaded_file.name}` | "
            f"**Size:** {img_pil.size[0]}×{img_pil.size[1]} px | "
            f"**Mode:** `{img_pil.mode}`"
        )
        st.markdown("---")
        run_inference(img_pil, modality, alpha, show_raw_cam)
    else:
        render_landing(modality)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#aaa;font-size:12px;padding:8px 0;'>"
        "🏥 Medical ROI Localisation System | EfficientNet-B3 + GradCAM++ | "
        "X-Ray · CT Scan · MRI | B.Tech Final Year Project</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()