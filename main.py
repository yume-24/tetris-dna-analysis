# main.py
# Model: Use simple layers: Conv1D, GlobalAveragePooling, and small MLP with up to ~100K parameters total.
import requests
import os

# from train_model import train_model, infer
# from train_rl_new import train_model, infer

#Bad
# from train_rl_model import train_model_rl, infer_rl
# from train_rl_model_backup import train_model_rl, infer_rl

from analyze import *



def download_jaspar_motif(motif_id: str, save_dir: str, version: str = "latest") -> str:
    """
    Download a JASPAR motif by ID in MEME format from JASPAR REST API
    and save it to save_dir/motif_id.meme.

    Args:
        motif_id: JASPAR motif ID, e.g. 'MA0065.2'
        save_dir: Directory to save the motif file
        version: Version or 'latest' (default)

    Returns:
        Path to the saved motif file
    """
    os.makedirs(save_dir, exist_ok=True)
    url = f"https://jaspar.genereg.net/api/v1/matrix/{motif_id}/?format=meme"
    resp = requests.get(url)
    resp.raise_for_status()
    path = os.path.join(save_dir, f"{motif_id}.meme")
    with open(path, "wb") as f:
        f.write(resp.content)
    return path




from tetris import build_tetris_data
if __name__ == '__main__':
    #Note: current dataset has 15k exactly
    #Next one will have #30k

    # build_tetris_data(num_games=10000, rows=20, cols=10, output_file='tetris_test.json')

    # download_jaspar_motif("MA0065.2", "./motifs") #MA0065.2 — PPARγ::RXRA heterodimer motif (PPARG) -- promotes metabolism
    # download_jaspar_motif("MA0105.4", "./motifs") #MA0105.4 — NFKB1 motif (a core NF-κB subunit) -- triggers inflamation
    

    # train_model()
    # infer()


    # train_model_rl()
    # infer_rl()


    # plot_validation_loss()
    generate_heatmap()