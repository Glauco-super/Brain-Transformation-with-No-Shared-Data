"""
Verifica da quale versione delle immagini GOD gli autori hanno costruito images_112.npz.

Le immagini GOD ricevute da Kamitani Lab sono tutte 500x500. Ricostruendo images_112.npz
da queste con image_prepare l'ordine coincide al 100% con la demo, ma i pixel differiscono
in media di ~2 livelli su 255, con un errore che cresce dall'angolo in alto a sinistra
verso quello in basso a destra (tipico di una scala leggermente diversa).
Ipotesi: gli autori sono partiti dalle immagini ImageNet ORIGINALI (dimensioni variabili),
a cui image_prepare applica il crop quadrato centrale prima del resize.

Lo script cerca le immagini GOD (stesso identificativo nXXXXXXXX_NNNN) negli shard Arrow
di ImageNet-1k presenti sul cluster, le prepara con image_prepare partendo dall'originale
e dalla versione 500x500 di Kamitani, e confronta entrambe con la demo.

Uso (dalla cartella del repository, ambiente crcns o b2b_nvtf con pyarrow):
    python preproc/check_god_source_imagenet.py [--ref data/Processed_data/GOD/images_112.npz]
"""
import argparse
import csv
import glob
import io
import os
import sys

import numpy as np
import pyarrow as pa
from PIL import Image
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from Utils.misc_compat import imread, fromimage
from Utils.image_functions import image_prepare

SRC_DIR = '/scratch_share/datasets/ILSVRC___imagenet-1k'
GOD_IMAGES_DIR = '/scratch_share/islab/glauco.rinaldi/datasets/GOD_Ho2022/images'


# Identificativo ImageNet: classe (n + 8 cifre) _ numero, es. n01443537_22563.
# Nei file Arrow di Hugging Face il nome ha un suffisso (es. n01440764_10026_n01440764.JPEG),
# quindi si confronta solo questa parte iniziale (come in extract_imagenet_subset.py).
ID_PATTERN = re.compile(r'n\d{8}_\d+')


def imagenet_id(name):
    if not name:
        return None
    m = ID_PATTERN.match(os.path.basename(name))
    return m.group() if m else None


def csv_names(path):
    with open(path, newline='') as f:
        return [row[1] for row in csv.reader(f) if row]


def open_table(path):
    src = pa.memory_map(path, 'r')
    try:
        return pa.ipc.open_stream(src).read_all()
    except pa.ArrowInvalid:
        return pa.ipc.open_file(pa.memory_map(path, 'r')).read_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC_DIR)
    ap.add_argument('--god-images', default=GOD_IMAGES_DIR)
    ap.add_argument('--ref', default='data/Processed_data/GOD/images_112.npz')
    a = ap.parse_args()

    ref = np.load(a.ref)
    # identificativo (senza estensione) -> (split, indice nel CSV)
    wanted = {}
    for split, csv_name, key in [('training', 'image_training_id.csv', 'train_images'),
                                 ('test', 'image_test_id.csv', 'test_images')]:
        for k, name in enumerate(csv_names(os.path.join(a.god_images, csv_name))):
            wanted[imagenet_id(name)] = (split, key, k, name)
    print('immagini GOD cercate:', len(wanted), flush=True)

    shards = sorted(glob.glob(os.path.join(a.src, '*.arrow')))   # train, validation e test
    print('shard:', len(shards), flush=True)
    rows = []
    for s, path in enumerate(shards):
        t = open_table(path)
        # Leggo solo il campo 'path' della colonna immagine (niente decodifica dei byte)
        col = t.column('image').combine_chunks()
        paths = col.field('path').to_pylist()
        for r, p in enumerate(paths):
            iid = imagenet_id(p)
            if iid in wanted:
                split, key, k, name = wanted[iid]
                b = col.field('bytes')[r].as_py()
                orig = fromimage(Image.open(io.BytesIO(b)))
                from_orig = image_prepare(orig, 112)
                from_500 = image_prepare(imread(os.path.join(a.god_images, split, name)), 112)
                demo = ref[key][k]
                rows.append((iid, split, orig.shape, np.abs(from_orig - demo).mean(),
                             np.abs(from_500 - demo).mean(), (from_orig == demo).mean()))
                print('%-22s %-8s originale %-15s  diff da originale %.5f | da 500x500 %.5f | pixel uguali (orig) %.3f'
                      % rows[-1], flush=True)
        if s == 0:
            print('esempi di nomi nello shard:', paths[:3], flush=True)
        if s % 20 == 0:
            print('shard %d/%d, trovate finora %d' % (s + 1, len(shards), len(rows)), flush=True)

    print('\nFINE: trovate %d immagini GOD su %d in ImageNet-1k' % (len(rows), len(wanted)))
    if rows:
        d_o = np.array([r[3] for r in rows]); d_5 = np.array([r[4] for r in rows])
        print('diff media dalla demo: partendo dall\'originale %.5f | dalla 500x500 %.5f'
              % (d_o.mean(), d_5.mean()))
        print('immagini identiche partendo dall\'originale: %d' % int((d_o == 0).sum()))


if __name__ == '__main__':
    main()
