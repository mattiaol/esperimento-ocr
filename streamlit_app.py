import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import pillow_heif  # Nuova libreria per HEIC

# Registra il supporto HEIC per Pillow
pillow_heif.register_heif_opener()

try:
    # CORREZIONE 1: Usa il nome corretto della funzione presente in detect_face.py
    from detect_face import face_factory 
    
    # CORREZIONE 2: Importiamo le factory disponibili, dato che ExtractWords non c'è
    from extract_words import ocr_factory, OcrFactory
    
    # Importiamo anche utility necessarie se dobbiamo ricostruire la logica
    from utlis import createHeatMapAndBoxCoordinates
except ImportError as e:
    st.error(f"Errore di importazione: {e}. Controlla i nomi nei file detect_face.py e extract_words.py")
    st.stop()

st.set_page_config(page_title="TC ID Card OCR Pro", page_icon="🆔", layout="wide")

st.title("🆔 TC ID Card OCR Scanner")
st.markdown("Carica qualsiasi immagine (JPG, PNG, HEIC) per estrarre i dati.")

# --- Funzione Helper per caricare qualsiasi formato ---
def load_image(uploaded_file):
    """Converte qualsiasi file uploadato in un'immagine PIL e poi in array OpenCV"""
    try:
        # Apre l'immagine (Pillow gestisce automaticamente HEIC grazie a register_heif_opener)
        image = Image.open(uploaded_file)
        
        # Corregge l'orientamento (es. foto fatte col cellulare)
        try:
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass # Se non ci sono exif, pazienza

        # Converte in RGB (rimuove trasparenze alpha se presenti)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        st.error(f"Errore nella lettura del file: {e}")
        return None

# Sidebar Impostazioni
st.sidebar.header("🔧 Impostazioni Avanzate")
with st.sidebar.expander("Configurazione Modelli"):
    face_method = st.selectbox("Metodo Volto", ["ssd", "haar"], index=0)
    ocr_method = st.selectbox("Metodo OCR", ["EasyOcr", "Tesseract"], index=0)
    neighbor_dist = st.slider("Distanza Box", 10, 100, 60)

# --- AREA DRAG & DROP ---
# Streamlit supporta nativamente il drag & drop nel file_uploader
uploaded_file = st.file_uploader(
    "Trascina qui la tua immagine o clicca per cercare", 
    type=["jpg", "png", "jpeg", "heic", "bmp", "webp"], # Supporto esteso
    accept_multiple_files=False
)

if uploaded_file is not None:
    # 1. Caricamento e Conversione Immagine
    pil_image = load_image(uploaded_file)
    
    if pil_image:
        # Layout a due colonne
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(pil_image, caption="Immagine Caricata", use_container_width=True)

        if st.button("🚀 Avvia Analisi OCR", type="primary"):
            with st.spinner("Elaborazione in corso..."):
                # Salvataggio temporaneo per compatibilità con la vecchia repo
                # La repo originale spesso rilegge il file da disco con cv2.imread
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tfile:
                    pil_image.save(tfile.name, format="JPEG")
                    temp_path = tfile.name

                try:
                    # --- FASE 1: Preparazione OpenCV ---
                    img_cv = cv2.imread(temp_path)
                    
                    # --- FASE 2: Rilevamento Volto ---
                    face_detector = FaceDetectionFactory(face_method)
                    processed_img = face_detector.detect_face(img_cv)
                    
                    if processed_img is None:
                        st.warning("⚠️ Nessun volto rilevato per l'allineamento. Uso l'immagine originale.")
                        processed_img = img_cv
                    
                    # --- FASE 3: Estrazione Parole ---
                    extractor = ExtractWords(
                        ocr_method=ocr_method,
                        neighbor_box_distance=neighbor_dist
                    )
                    
                    # ExtractWords vuole (image, name)
                    result_data, result_img = extractor.extract(processed_img, "scan_task")

                    # --- Visualizzazione Risultati ---
                    with col2:
                        # Converti da BGR (OpenCV) a RGB per mostrare a video
                        if isinstance(result_img, np.ndarray):
                            res_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                            st.image(res_rgb, caption="Aree Rilevate", use_column_width=True)
                        else:
                            st.warning("Nessuna immagine risultante generata.")

                    st.divider()
                    st.subheader("📄 Dati Estratti")
                    
                    # Visualizzazione JSON bella
                    st.json(result_data, expanded=True)

                    # Export dei dati
                    json_str = json.dumps(result_data, indent=2)
                    st.download_button(
                        label="📥 Scarica JSON",
                        data=json_str,
                        file_name="risultato_ocr.json",
                        mime="application/json"
                    )

                except Exception as e:
                    st.error(f"❌ Errore critico durante l'elaborazione: {e}")
                    st.write(e) # Stack trace per debug
                
                finally:
                    # Pulizia
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
