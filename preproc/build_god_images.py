"""
Ricostruisce data/Processed_data/GOD/images_112.npz dalle immagini stimolo GOD
(ricevute da Kamitani Lab) e, se indicato, lo confronta con il file della demo.

Ordine: riga k del CSV (image_training_id.csv / image_test_id.csv) = label k del file
h5 di Ho et al. (vedi l'analisi dei dati della fase 1), quindi train_images[k-1] e'
l'immagine con label k.
Preprocessing: Utils.image_functions.image_prepare(img, 112), la stessa funzione che il
codice usa a runtime per le immagini esterne (crop quadrato centrale, resize bicubico
a 112x112, grigi -> 3 canali, divisione per 255).

Uso (dalla cartella del repository):
    python preproc/build_god_images.py --images-dir <cartella con training/, test/ e i CSV> \
        --out <file .npz da scrivere> [--ref data/Processed_data/GOD/images_112.npz]
"""
import argparse
import csv
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from Utils.misc_compat import imread
from Utils.image_functions import image_prepare

SIZE = 112


def read_csv_names(path):
    """Nomi dei file nell'ordine delle righe del CSV (seconda colonna)."""
    with open(path, newline='') as f:
        return [row[1] for row in csv.reader(f) if row]


def build(images_dir, split, csv_name):
    names = read_csv_names(os.path.join(images_dir, csv_name))
    out = np.zeros((len(names), SIZE, SIZE, 3), dtype=np.float64)
    for i, name in enumerate(names):
        out[i] = image_prepare(imread(os.path.join(images_dir, split, name)), SIZE)
        if (i + 1) % 200 == 0:
            print('  %s: %d/%d' % (split, i + 1, len(names)), flush=True)
    return out


def compare(name, mine, ref):
    """Confronto globale e immagine per immagine (anche per scoprire un ordine diverso)."""
    print('\n== %s  mio %s  ref %s' % (name, mine.shape, ref.shape))
    if mine.shape != ref.shape:
        print('   FORME DIVERSE'); return
    d = np.abs(mine - ref)
    print('   identici: %s | diff max %.6f | diff media %.6f | pixel uguali %.4f%%'
          % (np.array_equal(mine, ref), d.max(), d.mean(), 100 * (d == 0).mean()))
    # Per ogni mia immagine, l'immagine della ref piu' vicina (MSE): se l'ordine e' giusto
    # deve essere quella con lo stesso indice.
    m = mine.reshape(len(mine), -1); r = ref.reshape(len(ref), -1)
    mse = ((m ** 2).sum(1)[:, None] + (r ** 2).sum(1)[None, :] - 2 * m @ r.T) / m.shape[1]
    best = mse.argmin(1)
    same = (best == np.arange(len(mine))).mean()
    print('   immagine piu vicina con lo stesso indice: %.2f%%' % (100 * same))
    per_img = d.reshape(len(d), -1).max(1)
    print('   diff max per immagine: mediana %.4f, peggiore %.4f (indice %d)'
          % (np.median(per_img), per_img.max(), per_img.argmax()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ref', default=None, help='npz della demo da confrontare')
    a = ap.parse_args()

    train = build(a.images_dir, 'training', 'image_training_id.csv')
    test = build(a.images_dir, 'test', 'image_test_id.csv')
    np.savez(a.out, train_images=train, test_images=test)
    print('salvato ' + a.out)

    if a.ref:
        ref = np.load(a.ref)
        compare('train_images', train, ref['train_images'])
        compare('test_images', test, ref['test_images'])


if __name__ == '__main__':
    main()
