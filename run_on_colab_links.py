import os
import pandas as pd
import requests

# ================= CONFIGURATION =================
# Définissez ici les liens de vos différents datasets (URLs directes vers des fichiers CSV)
DATASETS_LINKS = [
    {
        "url": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv",
        "nom": "nsl_kdd_train"
    },
    {
        "url": "https://raw.githubusercontent.com/yolandalal/Edge-IIoTset-Dataset/main/Edge-IIoTset%20dataset/Selected%20dataset%20for%20ML%20and%20DL/DNN-EdgeIIoT-dataset.csv",
        "nom": "edge_iiotset_sample"
    }
    # Ajoutez autant de liens que nécessaire :
    # { "url": "https://votre-lien-vers-fichier.csv", "nom": "nom_du_dataset" }
]

# Chemin où est supposé être le projet dans Google Colab
COLAB_DIR = "/content/FEAE/"
# =================================================

def run_in_colab():
    # 1. Vérification et déplacement dans le répertoire du projet
    if os.path.exists(COLAB_DIR):
        os.chdir(COLAB_DIR)
        print(f"[*] Déplacé dans le répertoire du projet : {COLAB_DIR}")
    else:
        print(f"[!] Attention : Le répertoire {COLAB_DIR} est introuvable.")
        print(f"[!] Assurez-vous d'avoir cloné ou déposé vos fichiers dans {COLAB_DIR}.")
        print(f"[*] Le script continue dans le dossier actuel : {os.getcwd()}")
        
    os.makedirs("data", exist_ok=True)

    for data_info in DATASETS_LINKS:
        url = data_info["url"]
        nom = data_info["nom"]
        path_csv = f"data/{nom}.csv"
        
        print(f"\n=========================================================")
        print(f"[*] Début du traitement pour : {nom}")
        print(f"[*] Lien : {url}")
        
        # 2. Téléchargement
        try:
            print("[*] Téléchargement en cours...")
            r = requests.get(url, timeout=30) # Ajout d'un timeout
            r.raise_for_status() # Lève une erreur si le statut de la requête est mauvais (ex: 404)
            with open(path_csv, 'wb') as f:
                f.write(r.content)
            print(f"[*] Téléchargement réussi -> {path_csv}")
        except Exception as e:
            print(f"[!] Erreur de téléchargement pour {nom} : {e}")
            continue # Passe au dataset suivant
            
        # 3. Standardisation (Adaptation au format attendu par LineGCN)
        try:
            print("[*] Lecture et standardisation des colonnes...")
            df = pd.read_csv(path_csv)
            mapping = {
                'src_ip': 'Source IP',
                'dst_ip': 'Destination IP',
                'src_port': 'Source Port',
                'dst_port': 'Destination Port',
                'protocol': 'Protocol',
                'label': 'Label'
            }
            # Renomme les colonnes selon le dictionnaire de mapping
            df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}, inplace=True)
            df.to_csv(path_csv, index=False)
            print("[*] Fichier standardisé avec succès.")
        except Exception as e:
            print(f"[!] Erreur de lecture/standardisation pour {nom} : {e}")
            continue
            
        # 4. Entraînement FEAE
        print(f"[*] Lancement de l'entraînement du modèle FEAE pour : {nom}")
        
        # NOTE : Dans Colab, l'environnement par défaut est souvent suffisant sans 'conda'
        # Si vous avez explicitement réinstallé Conda dans Colab, vous pouvez préfixer avec 'conda run -n feae'
        # Ici on utilise un appel Python standard approprié pour une exécution Colab classique
        cmd = f"env MPLBACKEND=Agg python feae/line_graph_bench.py --encoder=LineGCN --dataset={nom}"
        
        print(f"[*] Exécution : {cmd}")
        return_code = os.system(cmd)
        
        if return_code == 0:
            print(f"[*] [+] Entraînement terminé avec SUCCÈS pour {nom}")
        else:
            print(f"[!] [-] L'entraînement a retourné une erreur (code {return_code}) pour {nom}")

if __name__ == "__main__":
    run_in_colab()
