# Seed per la riproducibilita' degli addestramenti.
#
# SEED = None  -> nessun seed: ogni esecuzione e' diversa (comportamento originale del codice).
# SEED = intero -> stesso seed per Python (random), NumPy e TensorFlow.
# Il valore puo' essere sovrascritto per una singola esecuzione con la variabile
# d'ambiente B2B_SEED (es. B2B_SEED=1 python train_encoder.py ...), comodo nei job array.
SEED = None

# True -> chiede a TensorFlow di usare solo operazioni GPU deterministiche
# (TF_DETERMINISTIC_OPS=1, supportato da nvidia-tensorflow 1.15). Senza, anche con lo
# stesso seed alcune operazioni su GPU possono dare risultati leggermente diversi.
# Rende l'addestramento piu' lento. Sovrascrivibile con B2B_DETERMINISTIC=1.
DETERMINISTIC_OPS = False
