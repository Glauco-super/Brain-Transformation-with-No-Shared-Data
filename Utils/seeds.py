"""
Gestione centralizzata dei seed (Python, NumPy, TensorFlow).

I valori predefiniti sono in Utils/params/param_seeds.py; per una singola esecuzione
si possono sovrascrivere con le variabili d'ambiente B2B_SEED e B2B_DETERMINISTIC.
set_seeds() va chiamata all'inizio dello script, PRIMA di costruire qualunque modello
(il seed di TensorFlow vale per il grafo costruito dopo la chiamata).
"""
import os
import random

import numpy as np

from Utils.params.param_seeds import SEED, DETERMINISTIC_OPS


def get_seed():
    """Seed da usare: B2B_SEED se impostata, altrimenti SEED di param_seeds.py (o None)."""
    env = os.environ.get('B2B_SEED', '').strip()
    if env != '':
        return int(env)
    return SEED


def set_seeds(seed=None):
    """Imposta i seed di Python, NumPy e TensorFlow e restituisce il seed usato (o None)."""
    if seed is None:
        seed = get_seed()

    deterministic = DETERMINISTIC_OPS or os.environ.get('B2B_DETERMINISTIC', '').strip() == '1'
    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    if seed is None:
        print('seed: nessuno (esecuzione non riproducibile) | operazioni deterministiche: %s' % deterministic, flush=True)
        return None

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
        tf.compat.v1.set_random_seed(seed)
    else:
        tf.set_random_seed(seed)
    print('seed: %d | operazioni deterministiche: %s' % (seed, deterministic), flush=True)
    return seed
