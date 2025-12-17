from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import pillow_heif

# Enable HEIC/HEIF support in Pillow
pillow_heif.register_heif_opener()


# ----------------------------
# Streamlit setup
# ----------------------------
st.set_page_config(page_title="TC ID Card OCR", page_icon="🆔", layout="wide")
st.title("TC ID Card OCR Scanner")
st.markdown("Carica un'immagine (JPG/PNG/HEIC) e avvia l'estrazione dei dati.")


# ----------------------------
# Optional repo imports (graceful)
# ----------------------------
_FACE_FACTORY_AVAILABLE = False
_UTILS_AVAILABLE = False

try:
    # detect_face.py exposes: face_factory(face_model="ssd") -> FaceFactory
    from detect_face import face_factory  # type: ignore
    _FACE_FACTORY_AVAILABLE = True
except Exception:
    _FACE_FACTORY_AVAILABLE = False

try:
    # utlis.py exposes: correctPerspective(image)
    from utlis import correctPerspective  # type: ignore
    _UTILS_AVAILABLE = True
except Exception:
    _UTILS_AVAILABLE = False


# ----------------------------
# Helpers
# ----------------------------
def load_image(uploaded_file) -> Optional[Image.Image]:
    """Load any uploaded image as a PIL.Image (handles HEIC thanks to pillow-heif)."""
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        st.error(f"Errore nella lettura del file: {e}")
        return None


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR."""
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def normalize_text(s: str) -> str:
    """Uppercase + remove diacritics + compact spaces."""
    if s is None:
        return ""
    s = str(s).strip().replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def safe_regex_find(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text)
    return m.group(0) if m else None


@dataclass
class OcrItem:
    text: str
    text_norm: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


# ----------------------------
# OCR backends (SIMPLE mode)
# ----------------------------
@st.cache_resource(show_spinner=False)
def _get_easyocr_reader():
    import easyocr  # local import
    return easyocr.Reader(["tr", "en"], gpu=False)


def run_easyocr(img_rgb: np.ndarray) -> List[OcrItem]:
    reader = _get_easyocr_reader()
    results = reader.readtext(img_rgb)  # [(bbox4pts, text, conf), ...]
    items: List[OcrItem] = []
    for bbox, text, conf in results:
        pts = np.array(bbox).astype(int)
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        x2 = int(pts[:, 0].max())
        y2 = int(pts[:, 1].max())
        items.append(
            OcrItem(
                text=str(text),
                text_norm=normalize_text(str(text)),
                conf=float(conf),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return items


def run_tesseract(img_rgb: np.ndarray) -> List[OcrItem]:
    import pytesseract  # local import

    # Try Turkish + English; fall back to English if needed
    langs = ["tur+eng", "tur", "eng"]
    data = None
    last_err = None
    for lang in langs:
        try:
            data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT, lang=lang)
            last_err = None
            break
        except Exception as e:
            last_err = e
            data = None

    if data is None:
        raise RuntimeError(f"Tesseract non disponibile o errore di esecuzione: {last_err}")

    items: List[OcrItem] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = data["text"][i]
        if not txt or not str(txt).strip():
            continue
        conf_raw = data.get("conf", [0] * n)[i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = 0.0
        x1 = int(data["left"][i])
        y1 = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        items.append(
            OcrItem(
                text=str(txt),
                text_norm=normalize_text(str(txt)),
                conf=conf,
                x1=x1,
                y1=y1,
                x2=x1 + w,
                y2=y1 + h,
            )
        )
    return items


def find_label(items: List[OcrItem], label_variants: List[str]) -> Optional[OcrItem]:
    variants = [normalize_text(v) for v in label_variants]
    for it in items:
        if it.text_norm in variants:
            return it
    for it in items:
        for v in variants:
            if v and v in it.text_norm and len(it.text_norm) <= len(v) + 3:
                return it
    return None


def pick_line_words(items: List[OcrItem], y_center: float, y_tol: float, min_x: float) -> List[OcrItem]:
    same_line = [it for it in items if abs(it.cy - y_center) <= y_tol and it.x1 >= int(min_x)]
    same_line.sort(key=lambda it: it.x1)
    return same_line


def join_words(words: List[OcrItem]) -> str:
    if not words:
        return ""
    s = " ".join(w.text_norm for w in words)
    return re.sub(r"\s+", " ", s).strip()


def extract_fields_simple(items: List[OcrItem], img_shape_hw: Tuple[int, int]) -> Tuple[Dict[str, str], Dict[str, Tuple[int, int, int, int]]]:
    """Layout-aware extraction from OCR items."""
    h, _w = img_shape_hw
    y_tol = max(12.0, 0.02 * h)
    highlights: Dict[str, Tuple[int, int, int, int]] = {}

    full_text = " ".join(it.text_norm for it in items)
    full_text = re.sub(r"\s+", " ", full_text).strip()

    # Tc: 11 digits
    tc = ""
    for it in items:
        digits = re.sub(r"\D", "", it.text_norm)
        if len(digits) == 11:
            tc = digits
            highlights["Tc"] = (it.x1, it.y1, it.x2, it.y2)
            break
    if not tc:
        tc = safe_regex_find(r"\b\d{11}\b", full_text) or ""

    # Date of birth
    dob_pat = r"\b\d{2}[./-]\d{2}[./-]\d{4}\b"
    dob = ""
    for it in items:
        m = safe_regex_find(dob_pat, it.text_norm)
        if m:
            dob = m.replace("-", ".").replace("/", ".")
            highlights["DateofBirth"] = (it.x1, it.y1, it.x2, it.y2)
            break
    if not dob:
        m = safe_regex_find(dob_pat, full_text)
        dob = (m.replace("-", ".").replace("/", ".") if m else "")

    surname = ""
    name = ""

    lbl_surname = find_label(items, ["SOYADI", "SOYAD"])
    if lbl_surname:
        words = pick_line_words(items, lbl_surname.cy, y_tol, lbl_surname.x2 + 5)
        surname = join_words(words)
        if words:
            x1 = min(w.x1 for w in words)
            y1 = min(w.y1 for w in words)
            x2 = max(w.x2 for w in words)
            y2 = max(w.y2 for w in words)
            highlights["Surname"] = (x1, y1, x2, y2)

    lbl_name = find_label(items, ["ADI", "AD"])
    if lbl_name:
        words = pick_line_words(items, lbl_name.cy, y_tol, lbl_name.x2 + 5)
        name = join_words(words)
        if words:
            x1 = min(w.x1 for w in words)
            y1 = min(w.y1 for w in words)
            x2 = max(w.x2 for w in words)
            y2 = max(w.y2 for w in words)
            highlights["Name"] = (x1, y1, x2, y2)

    out = {"Tc": tc, "Surname": surname, "Name": name, "DateofBirth": dob}
    return out, highlights


def annotate(img_bgr: np.ndarray, items: List[OcrItem], highlights: Dict[str, Tuple[int, int, int, int]], draw_all: bool) -> np.ndarray:
    out = img_bgr.copy()
    if draw_all:
        for it in items:
            cv2.rectangle(out, (it.x1, it.y1), (it.x2, it.y2), (0, 255, 0), 1)

    for field, (x1, y1, x2, y2) in highlights.items():
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, field, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


# ----------------------------
# Repo "PIPELINE" mode (optional)
# ----------------------------
def pipeline_extract(img_bgr: np.ndarray, img_name: str, ocr_method: str, neighbor_dist: int) -> Tuple[Dict[str, str], np.ndarray]:
    """
    Attempts to run a pipeline compatible with the provided repo structure:
    - CRAFT (text heatmap)
    - UNet mask (if importable)
    - NearestBox refinement
    - OCR via extract_words.ocr_factory()

    This will run only if optional dependencies and model files are present.
    """
    # Imports are inside to keep SIMPLE mode fast and robust.
    from craft_text_detector import Craft  # type: ignore

    try:
        from find_nearest_box import NearestBox  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Impossibile importare find_nearest_box.py: {e}")

    try:
        # NOTE: unet_predict.py may require optional deps (segmentation_models_pytorch).
        from unet_predict import UnetModel  # type: ignore
        _unet_available = True
    except Exception:
        _unet_available = False

    try:
        from extract_words import ocr_factory  # type: ignore
        _ocr_factory_available = True
    except Exception as e:
        _ocr_factory_available = False
        raise RuntimeError(f"Impossibile importare extract_words.py: {e}")

    if not _unet_available:
        raise RuntimeError(
            "unet_predict.py non importabile (manca una dipendenza o un modello). "
            "Verifica che 'segmentation_models_pytorch' sia installato oppure usa la modalità SIMPLE."
        )

    # 1) CRAFT heatmap + boxes (run on CPU for compatibility)
    craft = Craft(output_dir="outputs", crop_type="poly", cuda=False)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    prediction = craft.detect_text(img_rgb)
    heatmap = prediction["heatmaps"]["text_score_heatmap"]
    boxes = prediction["boxes"]

    # 2) UNet mask from heatmap
    heatmap_3 = np.stack([heatmap, heatmap, heatmap], axis=-1).astype(np.float32)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    unet = UnetModel(device=device)
    mask = unet.predict(heatmap_3)

    # 3) Extract 4 line-like regions from mask (largest components)
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.max() <= 1 else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > 200:  # small filter
            rects.append((x, y, w, h, area))

    if len(rects) < 4:
        raise RuntimeError("Maschere UNet insufficienti (<4 regioni). Usa la modalità SIMPLE o controlla il modello.")

    # Keep 4 largest by area, sorted by y
    rects = sorted(rects, key=lambda r: r[4], reverse=True)[:4]
    rects = sorted(rects, key=lambda r: r[1])

    # 4) Convert CRAFT polygons to (x,w,y,h)
    box_coords = []
    for b in boxes:
        pts = np.array(b).astype(int)
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        x2 = int(pts[:, 0].max())
        y2 = int(pts[:, 1].max())
        box_coords.append([x1, x2 - x1, y1, y2 - y1])
    box_coords = np.array(box_coords, dtype=np.int32)

    # 5) Pick one representative box index per region (closest center)
    box_centers = np.column_stack([box_coords[:, 0] + box_coords[:, 1] / 2.0, box_coords[:, 2] + box_coords[:, 3] / 2.0])
    region_centers = np.array([[x + w / 2.0, y + h / 2.0] for x, y, w, h, _a in rects], dtype=np.float32)

    chosen_indexes = []
    for rc in region_centers:
        d = np.linalg.norm(box_centers - rc[None, :], axis=1)
        chosen_indexes.append(int(np.argmin(d)))
    box_indexes = tuple(chosen_indexes[:4])

    # 6) Refine boxes with NearestBox (optional extension)
    nearest = NearestBox(distance_thresh=neighbor_dist, draw_line=False)
    refined_boxes = nearest.searchNearestBoundingBoxes(box_coords, box_indexes, img_bgr)

    # 7) OCR using the repo factory
    ocr = ocr_factory(ocr_method=ocr_method, border_thresh=3, denoise=False)
    text_output = ocr.ocrOutput(img_name, img_bgr, refined_boxes)

    # 8) Draw refined boxes
    out = img_bgr.copy()
    labels = ["Tc", "Surname", "Name", "DateofBirth"]
    for lab, (x, w, y, h) in zip(labels, refined_boxes):
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        cv2.putText(out, lab, (int(x), max(0, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return text_output, out


# ----------------------------
# Face alignment / perspective (optional)
# ----------------------------
def align_by_face(img_bgr: np.ndarray, method: str, rot_interval: int) -> Tuple[np.ndarray, str]:
    if not _FACE_FACTORY_AVAILABLE:
        return img_bgr, "Rilevamento volto non disponibile (detect_face.py non importabile)."

    try:
        detector = face_factory(method).get_face_detector()
        aligned = detector.changeOrientationUntilFaceFound(img_bgr, rot_interval)
        if aligned is None:
            return img_bgr, "Nessun volto rilevato: uso l'immagine originale."
        return aligned, "Allineamento tramite volto completato."
    except Exception as e:
        return img_bgr, f"Allineamento volto non riuscito: {e}. Uso l'immagine originale."


def apply_perspective(img_bgr: np.ndarray) -> Tuple[np.ndarray, str]:
    if not _UTILS_AVAILABLE:
        return img_bgr, "Correzione prospettiva non disponibile (utlis.py non importabile)."
    try:
        return correctPerspective(img_bgr), "Correzione prospettiva completata."
    except Exception as e:
        return img_bgr, f"Correzione prospettiva non riuscita: {e}. Uso l'immagine originale."


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Impostazioni")

mode = st.sidebar.radio("Modalità", ["SIMPLE (consigliata)", "PIPELINE (sperimentale)"], index=0)

with st.sidebar.expander("Opzioni avanzate", expanded=True):
    face_method = st.selectbox("Metodo volto (rotazione)", ["ssd", "haar", "dlib"], index=0)
    rot_interval = st.slider("Passo rotazione (gradi)", 5, 30, 15)
    do_perspective = st.checkbox("Correzione prospettiva ID (se disponibile)", value=False)
    ocr_method = st.selectbox("Metodo OCR", ["TesseractOcr", "EasyOcr"], index=0)
    neighbor_dist = st.slider("Distanza NearestBox (solo PIPELINE)", 10, 100, 60)
    show_all_boxes = st.checkbox("Mostra tutti i box OCR (solo SIMPLE)", value=True)


uploaded_file = st.file_uploader(
    "Trascina qui la tua immagine o clicca per cercare",
    type=["jpg", "png", "jpeg", "heic", "bmp", "webp"],
    accept_multiple_files=False,
)


# ----------------------------
# Main
# ----------------------------
if uploaded_file is not None:
    pil_image = load_image(uploaded_file)
    if pil_image is None:
        st.stop()

    img_bgr = pil_to_bgr(pil_image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_image, caption="Immagine caricata", use_container_width=True)

    if st.button("Avvia OCR", type="primary"):
        with st.spinner("Elaborazione in corso..."):
            # 0) Face alignment
            img_bgr, msg_face = align_by_face(img_bgr, face_method, rot_interval)
            st.info(msg_face)

            # 1) Perspective correction (optional)
            if do_perspective:
                img_bgr, msg_p = apply_perspective(img_bgr)
                st.info(msg_p)

            # 2) Run selected mode
            try:
                if mode.startswith("PIPELINE"):
                    data, vis_bgr = pipeline_extract(
                        img_bgr=img_bgr,
                        img_name="streamlit_scan",
                        ocr_method=ocr_method,
                        neighbor_dist=int(neighbor_dist),
                    )
                else:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    # SIMPLE: pick OCR backend
                    if ocr_method == "EasyOcr":
                        try:
                            items = run_easyocr(img_rgb)
                        except Exception as e:
                            st.warning(f"EasyOCR non disponibile ({e}). Passo a Tesseract.")
                            items = run_tesseract(img_rgb)
                    else:
                        items = run_tesseract(img_rgb)

                    data, highlights = extract_fields_simple(items, img_rgb.shape[:2])
                    vis_bgr = annotate(img_bgr, items, highlights, draw_all=show_all_boxes)

                # 3) Show result
                with col2:
                    st.image(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), caption="Risultato / Debug", use_container_width=True)

                st.divider()
                st.subheader("Dati estratti")
                st.json(data, expanded=True)

                json_str = json.dumps(data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Scarica JSON",
                    data=json_str,
                    file_name="risultato_ocr.json",
                    mime="application/json",
                )

            except Exception as e:
                st.error(f"Errore durante l'elaborazione: {e}")
