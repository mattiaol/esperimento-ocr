import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import json

# Importiamo i moduli della repository originale
# Assicurati che detect_face.py, extract_words.py, ecc. siano nella stessa cartella
try:
    from detect_face import FaceDetectionFactory
    from extract_words import ExtractWords
except ImportError as e:
    st.error(f"Errore di importazione: {e}. Assicurati di aver caricato TUTTI i file della repo originale.")
    st.stop()

st.set_page_config(page_title="TC ID Card OCR", page_icon="🆔", layout="wide")

st.title("🆔 TC ID Card OCR Scanner")
st.markdown("Replica web della repo **musimab/Tc_ID_Card_OCR**. Estrae dati dalle carte d'identità Turche.")

# Configurazione Sidebar (Simula gli argomenti CLI originali)
st.sidebar.header("Impostazioni Modello")
face_method = st.sidebar.selectbox("Metodo Rilevamento Volto", ["ssd", "haar"], index=0, help="SSD è consigliato per Streamlit Cloud (Dlib è stato rimosso per compatibilità).")
ocr_method = st.sidebar.selectbox("Metodo OCR", ["EasyOcr", "Tesseract"], index=0)
neighbor_dist = st.sidebar.slider("Distanza Box Vicini", 10, 100, 60)
rotation_int = st.sidebar.slider("Intervallo Rotazione", 0, 360, 60)

uploaded_file = st.file_uploader("Carica una carta d'identità (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Salva il file temporaneamente (la repo originale lavora con path di file spesso)
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    image_path = tfile.name

    # Carica immagine per display
    original_image = Image.open(image_path)
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Immagine Originale", use_column_width=True)

    if st.button("Avvia Analisi Completa"):
        with st.spinner("Inizializzazione Pipeline (Face Detect -> CRAFT -> UNET -> OCR)..."):
            try:
                # --- FASE 1: Rilevamento Volto ---
                # Usiamo la factory della repo originale
                face_detector = FaceDetectionFactory(face_method)
                
                # Lettura con OpenCV
                img_cv = cv2.imread(image_path)
                
                # La repo si aspetta spesso un path o un'immagine cv2. 
                # Adattiamo la logica di main.py qui:
                
                # Face Detection & Alignment
                # Nota: Ispezionando il codice originale, detect_face restituisce coordinate o immagine ruotata
                # Qui simuliamo la logica principale di main.py riga per riga
                
                processed_img = face_detector.detect_face(img_cv)
                if processed_img is None:
                    st.warning("Nessun volto rilevato o allineamento fallito. Procedo con l'immagine originale.")
                    processed_img = img_cv
                
                # --- FASE 2: Estrazione Parole (CRAFT + UNET + OCR) ---
                # Inizializza la classe ExtractWords dalla repo
                extractor = ExtractWords(
                    ocr_method=ocr_method,
                    neighbor_box_distance=neighbor_dist
                )
                
                # Esegui l'estrazione
                # NOTA: ExtractWords.extract vuole (image, image_name)
                result_data, result_img = extractor.extract(processed_img, "uploaded_image")

                # --- Visualizzazione Risultati ---
                with col2:
                    # Converti BGR (OpenCV) a RGB per Streamlit
                    if isinstance(result_img, np.ndarray):
                        res_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        st.image(res_rgb, caption="Risultato Elaborato", use_column_width=True)

                st.divider()
                st.subheader("📄 Dati Estratti")
                
                # Formattiamo il JSON in modo leggibile
                st.json(result_data)
                
                # Opzionale: Mostra campi specifici se la struttura è nota
                if 'tc' in result_data:
                    st.success(f"TC NO: {result_data['tc']}")
                if 'name' in result_data:
                    st.info(f"Nome: {result_data['name']}")
                if 'surname' in result_data:
                    st.info(f"Cognome: {result_data['surname']}")

            except Exception as e:
                st.error(f"Errore durante l'elaborazione: {e}")
                st.write("Dettagli errore (debug):", e)
    
    # Pulizia file temporaneo
    os.unlink(image_path)
