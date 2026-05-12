import gradio as gr
import numpy as np
import joblib
import os
from inference_utils import load_models, extract_ecapa_from_file, extract_bert_from_file

# 1. Betöltjük a betanított modellt és a küszöböt a szerver indulásakor
MODEL_PATH = "./saved_models/multimodal_xgboost.pkl"
THRESHOLD_PATH = "./saved_models/optimal_threshold.txt"

print("Betanított XGBoost modell betöltése...")
xgb_model = joblib.load(MODEL_PATH)
with open(THRESHOLD_PATH, 'r') as f:
    CLINICAL_THRESHOLD = float(f.read().strip())

# Betöltjük a Deep Learning modelleket a GPU-ra
load_models()

def evaluate_patient(audio_file, transcript_file):
    if audio_file is None or transcript_file is None:
        return "Kérlek, töltsd fel mindkét fájlt (Audio + Transcript)!", ""

    try:
        # 1. Jellemzőkinyerés
        print("Extracting ECAPA...")
        ecapa_emb = extract_ecapa_from_file(audio_file)
        print("Extracting BERT...")
        bert_emb = extract_bert_from_file(transcript_file)

        # 2. Fúzió: Összefűzzük a két vektort (192 + 768 = 960 dimenzió)
        # Fontos: Pontosan abban a sorrendben kell lenniük, ahogy a betanításnál voltak! (ecapa majd bert)
        fused_vector = np.concatenate([ecapa_emb, bert_emb]).reshape(1, -1)

        # 3. Predikció az XGBoost-tal
        dementia_probability = xgb_model.predict_proba(fused_vector)[0][1]

        # 4. Kiértékelés az optimális orvosi küszöb alapján
        if dementia_probability >= CLINICAL_THRESHOLD:
            diagnosis = "🔴 Magas Kockázat: Demencia jelei detektálva"
            explanation = f"A modell {dementia_probability*100:.1f}%-os valószínűséggel demenciát jelzett. (A klinikai küszöb: {CLINICAL_THRESHOLD*100:.1f}%)."
        else:
            diagnosis = "🟢 Alacsony Kockázat: Egészséges"
            explanation = f"A modell {dementia_probability*100:.1f}%-os valószínűséggel jelzett demenciát, ami a {CLINICAL_THRESHOLD*100:.1f}%-os klinikai küszöb alatt van."

        return diagnosis, explanation

    except Exception as e:
        return f"Hiba történt a feldolgozás során: {str(e)}", ""

# --- GRADIO FELÜLET ---
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Beszéd alapú demencia diagnosztika (Cookie Theft Picture)")
    gr.Markdown("Töltsd fel a páciens hangfelvételét és a leíratát, hogy a multimodális (Audio + Szöveg) AI modell kiértékelje az eredményt.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.File(label="Hangfájl feltöltése (.mp3 / .wav)")
            transcript_input = gr.File(label="Transcript fájl feltöltése (.cha)")
            eval_btn = gr.Button("Kiértékelés", variant="primary")
            
        with gr.Column():
            result_diagnosis = gr.Textbox(label="Diagnózis", lines=2)
            result_explanation = gr.Textbox(label="Részletes Magyarázat", lines=3)

    eval_btn.click(
        fn=evaluate_patient,
        inputs=[audio_input, transcript_input],
        outputs=[result_diagnosis, result_explanation]
    )

if __name__ == "__main__":
    # Megnyitjuk az összes hálózati interfészt (0.0.0.0), hogy a Dockerből elérhető legyen kívülről is
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)