import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoTokenizer, AutoModel
import subprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Globális változók a modellek gyors memóriában tartásához
_ecapa_classifier = None
_bert_tokenizer = None
_bert_model = None

def load_models():
    global _ecapa_classifier, _bert_tokenizer, _bert_model
    if _ecapa_classifier is None:
        print("Loading ECAPA model...")
        _ecapa_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="./tmp_model",
            run_opts={"device": DEVICE}
        )
    if _bert_model is None:
        print("Loading BERT model...")
        _bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        _bert_model = AutoModel.from_pretrained("bert-base-uncased")
        _bert_model.to(DEVICE)
        _bert_model.eval()

def extract_ecapa_from_file(audio_path):
    # Ha mp3, átalakítjuk wav-ba az ffmpeg segítségével
    if audio_path.lower().endswith(".mp3"):
        wav_path = audio_path.replace(".mp3", ".wav")
        try:
            # FFmpeg futtatása a háttérben némítva (stdout/stderr elrejtése)
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            audio_path = wav_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg konverziós hiba: Nem sikerült átalakítani az mp3 fájlt. Részletek: {e}")

    # Hangfájl betöltése a Torchaudio segítségével
    signal, fs = torchaudio.load(audio_path)
    
    # Ha sztereó, monósítjuk
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
        
    # Ha nem 16kHz, átmintavételezzük (A SpeechBrain ECAPA 16kHz-et vár)
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)

    # ECAPA jellemzők kinyerése
    with torch.no_grad():
        embeddings = _ecapa_classifier.encode_batch(signal.to(DEVICE))
        emb_flat = embeddings.squeeze().cpu().numpy()
        
    return emb_flat

def extract_bert_from_file(transcript_path):
    # Szöveg kinyerése a .cha fájlból
    utterances = []
    with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("*PAR"):
                text = line.split(":", 1)[-1]
                utterances.append(text)
    full_text = " ".join(utterances)

    if len(full_text.strip()) == 0:
        raise ValueError("A transcript fájl nem tartalmaz páciens (*PAR) szöveget!")

    # BERT Embedding generálása
    inputs = _bert_tokenizer(full_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    
    return cls_embedding