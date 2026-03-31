# Kód célja: Ellenőrizni, hogy a hangfájlok helyes mappákban vannak-e a klinikai diagnózis alapján.
# A fájlok helyes besorolása kritikus a modell tanításához, ezért ez egy fontos lépés az adat-előkészítés során.
import os
import glob
import pandas as pd

DATA_DIR = "./data"
EXCEL_PATH = os.path.join(DATA_DIR, "Pitt-data.xlsx")

def check_folder_vs_metadata():
    print("1. Audiófájlok és mappák beolvasása...")
    mp3_files = glob.glob(os.path.join(DATA_DIR, "**/*.mp3"), recursive=True)
    
    audio_records = []
    for f in mp3_files:
        if "WLS" in f: continue
        
        filename = os.path.basename(f)
        # Melyik mappában van a fájl?
        if "dementia-audio" in f:
            folder_label = "Dementia Mappa"
        elif "control-audio" in f:
            folder_label = "Control Mappa"
        else:
            folder_label = "Ismeretlen Mappa"
            
        try:
            # Beteg ID kinyerése a fájlnévből (pl: 001-0.mp3 -> 1)
            patient_id = int(filename.split('-')[0])
            audio_records.append({
                "Filename": filename, 
                "id": patient_id, 
                "Folder_Location": folder_label
            })
        except Exception as e:
            pass

    df_audio = pd.DataFrame(audio_records)
    print(f"Beolvasott audiófájlok: {len(df_audio)} db")

    print("\n2. Klinikai metaadatok beolvasása...")
    df_meta = pd.read_excel(EXCEL_PATH, sheet_name="data", skiprows=2)
    df_meta.columns = df_meta.columns.str.strip()

    def map_diagnosis(code):
        if pd.isna(code): return "Unknown"
        code = int(code)
        if code in [1, 100, 2, 200, 3, 300, 4, 420, 430]: return "Klinikai Demencia"
        elif code in [8, 800, 851]: return "Klinikai Kontroll"
        elif code in [6, 7, 610, 611, 720, 740, 730, 770]: return "Klinikai MCI / Other"
        else: return "Klinikai Other"

    df_meta['Clinical_Diagnosis'] = df_meta['basedx'].apply(map_diagnosis)

    print("\n3. Mappa vs. Metaadat összehasonlítása...")
    # Összekötjük a hangfájlokat a metaadattal az ID alapján
    df_merged = pd.merge(df_audio, df_meta[['id', 'Clinical_Diagnosis']], on='id', how='left')

    # Kereszttábla (Crosstab) generálása
    crosstab = pd.crosstab(df_merged['Folder_Location'], df_merged['Clinical_Diagnosis'], margins=True)
    
    print("\n" + "="*60)
    print("TÉVESZTÉSI MÁTRIX: Mappa elhelyezkedés VS Valós diagnózis")
    print("="*60)
    print(crosstab)
    print("="*60)

    # Hibásan besorolt fájlok kigyűjtése
    mismatches = df_merged[
        ((df_merged['Folder_Location'] == 'Control Mappa') & (df_merged['Clinical_Diagnosis'] != 'Klinikai Kontroll')) |
        ((df_merged['Folder_Location'] == 'Dementia Mappa') & (df_merged['Clinical_Diagnosis'] != 'Klinikai Demencia'))
    ]
    
    if len(mismatches) > 0:
        print(f"\nFIGYELEM! {len(mismatches)} db fájl nem a megfelelő mappában van, vagy MCI kategóriás!")
        print(mismatches.head(15).to_string(index=False))
    else:
        print("\nTökéletes egyezés! A mappák 100%-ban megfelelnek a klinikai diagnózisnak.")

if __name__ == "__main__":
    check_folder_vs_metadata()