"""
NSD, primo passo della ricostruzione dei .pt della demo: righe, label e immagini.

Ipotesi da verificare (dall'analisi dei .pt della demo di subj01):
  - le righe seguono l'ordine di subjectim (nsd_expdesign.mat): prima i 1000 shared
    (ordinati per id 73k), poi i 9000 propri del soggetto;
  - si tengono solo le immagini viste almeno una volta nelle sessioni disponibili agli
    autori (iculding_ind_mask = posizioni 0..9999 in subjectim; per subj01 ne restano 9841:
    ipotesi = sessioni 1-37, perche' le ultime 3 erano state trattenute da NSD);
  - Images_ordered_10k_labels = subjectim[subj, posizione], cioe' id 73k in base 1;
  - shared_mask_ordered_10k = posizione < 1000;
  - Images_ordered_10k = immagini di nsd_stimuli.hdf5 (425x425) ridotte a 112x112, uint8.

Uso (dalla cartella del repository, ambiente b2b_nvtf: torch, h5py, scipy, Pillow):
    python preproc/check_nsd_rows_images.py [--subj 1] [--nsd-dir ...] [--demo-dir ...]
"""
import argparse
import glob
import os
import sys

import numpy as np
import h5py
import scipy.io
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from Utils.image_functions import image_prepare

TAG = '_fsaverage_betas_fithrf_GLMdenoise_RR'
TRIALS_PER_SESSION = 750


def find(root, name):
    hits = glob.glob(os.path.join(root, '**', name), recursive=True)
    if not hits:
        raise SystemExit('non trovato %s sotto %s' % (name, root))
    print('trovato', hits[0])
    return hits[0]


def tload(path):
    x = torch.load(path, map_location='cpu')
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


def same(name, a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = a.shape == b.shape and np.array_equal(a, b)
    extra = '' if ok or a.shape != b.shape else ' (diversi in %d posizioni)' % int((a != b).sum())
    print('  %-45s %s%s' % (name, 'IDENTICO' if ok else 'DIVERSO %s vs %s' % (a.shape, b.shape), extra))
    return ok


def resize_variants(img):
    """img: uint8 425x425x3 -> dizionario di varianti uint8 112x112x3."""
    pil = Image.fromarray(img)
    out = {'image_prepare (bicubic, repo)': np.round(image_prepare(img, 112) * 255).astype(np.uint8)}
    for name, f in [('PIL bilinear', Image.BILINEAR), ('PIL lanczos', Image.LANCZOS),
                    ('PIL box', Image.BOX), ('PIL nearest', Image.NEAREST)]:
        out[name] = np.asarray(pil.resize((112, 112), f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subj', type=int, default=1)
    ap.add_argument('--nsd-dir', default='/scratch_share/islab/glauco.rinaldi/datasets/NSD')
    ap.add_argument('--demo-dir', default='/scratch_share/islab/glauco.rinaldi/demo_data/Processed_data/NSD')
    ap.add_argument('--n-variants', type=int, default=300, help='immagini usate per scegliere il resize')
    a = ap.parse_args()
    s = a.subj
    D = a.demo_dir + '/'

    # --- file della demo
    lab_demo = tload(D + 'Images_ordered_10k_labels_subj0%d%s.pt' % (s, TAG)).ravel().astype(np.int64)
    inc_demo = tload(D + 'iculding_ind_mask_ordered_10k_subj0%d%s.pt' % (s, TAG)).ravel()
    shm_demo = tload(D + 'shared_mask_ordered_10k_subj0%d%s.pt' % (s, TAG)).ravel()
    print('demo: %d righe, label %d..%d' % (len(lab_demo), lab_demo.min(), lab_demo.max()))

    # --- disegno sperimentale NSD
    exp = scipy.io.loadmat(find(a.nsd_dir, 'nsd_expdesign.mat'))
    subjectim = exp['subjectim'].astype(np.int64)          # 8 x 10000, id 73k in base 1
    master = exp['masterordering'].ravel().astype(np.int64)  # 30000, posizioni 10k in base 1
    print('subjectim', subjectim.shape, '| masterordering', master.shape)

    # --- 1) quali sessioni spiegano le righe tenute?
    print('\n1) Immagini tenute (iculding_ind_mask) in funzione del numero di sessioni:')
    best = None
    for n_sess in range(30, 41):
        seen = np.unique(master[:n_sess * TRIALS_PER_SESSION] - 1)    # posizioni 0..9999
        match = len(seen) == len(inc_demo) and np.array_equal(seen, inc_demo)
        print('   sessioni 1-%d: %5d immagini viste %s' % (n_sess, len(seen), '<- IDENTICO alla demo' if match else ''))
        if match:
            best = n_sess
    if best is None:
        print('   nessun numero di sessioni riproduce esattamente le righe della demo')

    # --- 2) label e maschera shared
    print('\n2) Label e maschera shared (righe = iculding_ind_mask della demo):')
    lab_mine = subjectim[s - 1, inc_demo]
    same('Images_ordered_10k_labels = subjectim[pos]', lab_mine, lab_demo)
    same('shared_mask_ordered_10k = pos < 1000', (inc_demo < 1000).astype(np.float32), shm_demo)

    # --- 3) immagini
    print('\n3) Immagini 112x112:')
    img_demo = tload(D + 'Images_ordered_10k_subj0%d%s.pt' % (s, TAG))
    print('   demo', img_demo.shape, img_demo.dtype)
    with h5py.File(find(a.nsd_dir, 'nsd_stimuli.hdf5'), 'r') as f:
        brick = f['imgBrick']
        print('   imgBrick', brick.shape, brick.dtype)
        # 3a) scelta della variante di resize su un sottoinsieme
        rows = np.linspace(0, len(lab_demo) - 1, a.n_variants).astype(int)
        err = {}
        for r in rows:
            img = brick[lab_demo[r] - 1]                    # id in base 1 -> indice in base 0
            for name, v in resize_variants(img).items():
                d = np.abs(v.astype(np.int16) - img_demo[r].astype(np.int16))
                e = err.setdefault(name, [0.0, 0.0, 0])
                e[0] += d.mean(); e[1] += (d == 0).mean(); e[2] += 1
        print('   varianti su %d immagini (diff media in livelli 0-255, pixel uguali):' % len(rows))
        for name, (dm, eq, n) in sorted(err.items(), key=lambda kv: kv[1][0]):
            print('     %-32s diff %.4f  uguali %.2f%%' % (name, dm / n, 100 * eq / n))
        # 3b) controllo dell'ordine sulle prime 200 righe: l'immagine giusta e' la piu' vicina?
        k = 200
        mine = np.stack([np.round(image_prepare(brick[lab_demo[r] - 1], 112) * 255) for r in range(k)])
        dm = mine.reshape(k, -1).astype(np.float32); dd = img_demo[:k].reshape(k, -1).astype(np.float32)
        mse = (dm ** 2).sum(1)[:, None] + (dd ** 2).sum(1)[None] - 2 * dm @ dd.T
        print('   ordine: immagine piu vicina con lo stesso indice nelle prime %d righe: %.1f%%'
              % (k, 100 * (mse.argmin(1) == np.arange(k)).mean()))


if __name__ == '__main__':
    main()
