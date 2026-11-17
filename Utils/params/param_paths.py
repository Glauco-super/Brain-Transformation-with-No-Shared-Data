# Percorsi dei dati esterni usati dal codice.
# Nel codice originale erano scritti fissi e puntavano al server degli autori
# (/net/mraid11/...). Qui sono raccolti in un unico posto, con i percorsi
# sul cluster HPC ReGAInS.

_SCRATCH = '/scratch_share/islab/glauco.rinaldi/datasets'

# Cartella delle immagini "esterne" (senza fMRI) usate nella parte
# auto-supervisionata dell'addestramento. Struttura attesa, come nell'originale:
# una sottocartella per classe ImageNet, con nome che inizia per 'n'.
# Creata con preproc/extract_imagenet_subset.py.
EXT_IMAGES_DIR = _SCRATCH + '/imagenet_ext_50pc'

# Lista delle immagini di test di GOD (nel codice viene solo letta, non usata).
GOD_TEST_CSV = _SCRATCH + '/GOD_Ho2022/images/image_test_id.csv'

# File in cui il generatore salva l'elenco delle immagini esterne scelte.
EXT_FILES_CACHE = _SCRATCH + '/ext_images_files.pkl'

# Fattori di normalizzazione delle immagini (img_norm.npz) usati solo dalle
# loss mae()/mse() su immagini in Utils/image_loss.py. Il file originale non
# e' distribuito dagli autori: None = non disponibile.
IMG_NORM_FILE = None
