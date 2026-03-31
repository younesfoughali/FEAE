import os
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# ================= CONFIGURATION =================
# Liste des datasets Kaggle à utiliser pour l'entraînement
DATASETS = [
    {
        "kaggle_handle": "ziya07/network-security-dataset", # Le handle Kaggle du repo
        "file_path": "", # Le nom exact du fichier CSV dans le repo (laisser vide si fichier unique)
        "nom": "ziya_network_security"
    },
    {
        "kaggle_handle": "cicdataset/cicids2017",
        "file_path": "MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", 
        "nom": "cicids2017_portscan"
    },
    {
        "kaggle_handle": "mrwellsdavid/unsw-nb15",
        "file_path": "UNSW_NB15_testing-set.csv", 
        "nom": "unsw_nb15_test"
    }
]
# ============================================================

def prepare_and_run(kaggle_handle, file_path, nom_dataset):
    # 1. Aller dans le bon répertoire
    try:
        os.chdir('/content/FEAE/')
    except FileNotFoundError:
        pass # Reste dans le répertoire courant si exécuté en local
    
    os.makedirs("data", exist_ok=True)
    path_csv = f"data/{nom_dataset}.csv"
    
    print(f"\n[*] ======================================================")
    print(f"[*] Début du traitement pour : {nom_dataset}")
    print(f"[*] Téléchargement du dataset Kaggle : {kaggle_handle}")
    
    # 2. Téléchargement via kagglehub
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            kaggle_handle,
            file_path,
        )
    except Exception as e:
        print(f"[!] Erreur lors du téléchargement de {nom_dataset} via Kaggle : {e}")
        return

    # 3. Standardisation des colonnes
    print("[*] Vérification du format des données et standardisation...")
    try:
        # Mapping générique : à adapter si ton CSV utilise des noms très différents (ex: src_ip -> Source IP)
        mapping = {
            'src_ip': 'Source IP',
            'dst_ip': 'Destination IP',
            'src_port': 'Source Port',
            'dst_port': 'Destination Port',
            'protocol': 'Protocol',
            'label': 'Label',
            # Ajoutez tout autre format de nommage propre aux différents Datasets Kaggle
        }
        
        # Renommage
        df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}, inplace=True)
        
        # Sauvegarde en format CSV attendu par la solution
        df.to_csv(path_csv, index=False)
        print(f"[*] Dataset prêt et sauvegardé dans : {path_csv}")
    except Exception as e:
        print(f"[!] Erreur lors du traitement des colonnes de {nom_dataset} : {e}")
        return

    # 4. Lancement de la commande originale
    print(f"[*] Lancement de l'entraînement sur {nom_dataset}...")
    cmd = f"env MPLBACKEND=Agg conda run -n feae python feae/line_graph_bench.py --encoder=LineGCN --dataset={nom_dataset}"
    os.system(cmd)
    print(f"[*] Fin du traitement pour {nom_dataset}")

# Exécuter le processus pour chaque dataset
for dataset in DATASETS:
    prepare_and_run(dataset["kaggle_handle"], dataset["file_path"], dataset["nom"])