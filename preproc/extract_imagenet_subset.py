"""
Estrae un sottoinsieme di ImageNet-1k (train) dai file Arrow di Hugging Face
presenti sul cluster e lo salva come JPEG in una cartella per classe:

    <out_dir>/<wnid o nNNNN>/<nome_file>.JPEG

E' la struttura che si aspetta Utils/batch_generator.py
(batch_generator_external_images), che sceglie le immagini dalle
sottocartelle il cui nome inizia per 'n'.

Uso:
    python preproc/extract_imagenet_subset.py --inspect
    python preproc/extract_imagenet_subset.py --per-class 50 --out <cartella>
"""
import argparse
import glob
import io
import os
import random
from collections import defaultdict

import numpy as np
import pyarrow as pa
from PIL import Image

SRC_DIR = '/scratch_share/datasets/ILSVRC___imagenet-1k'


def open_table(path):
    """Apre un file .arrow senza caricarlo in RAM (memory map).
    I file della cache di Hugging Face sono in formato 'stream';
    per sicurezza si prova anche il formato 'file'."""
    src = pa.memory_map(path, 'r')
    try:
        return pa.ipc.open_stream(src).read_all()
    except pa.ArrowInvalid:
        return pa.ipc.open_file(pa.memory_map(path, 'r')).read_all()


def get_image_field(table, i):
    """Restituisce (bytes, path) dell'immagine alla riga i."""
    cell = table.column('image')[i].as_py()
    if isinstance(cell, dict):
        return cell.get('bytes'), cell.get('path')
    return cell, None


def class_folder(label, path):
    """Nome della sottocartella: il WordNet ID (es. n01440764) se ricavabile
    dal nome del file originale, altrimenti n + indice della classe."""
    if path:
        name = os.path.basename(path)
        if name.startswith('n') and '_' in name:
            return name.split('_')[0]
    return 'n%04d' % label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC_DIR)
    ap.add_argument('--out', default='/scratch_share/islab/glauco.rinaldi/datasets/imagenet_ext_50pc')
    ap.add_argument('--per-class', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--inspect', action='store_true')
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.src, '*-train-*.arrow')))
    print('shard di train trovati:', len(shards), flush=True)

    if args.inspect:
        t = open_table(shards[0])
        print(t.schema)
        print('righe nel primo shard:', t.num_rows)
        b, p = get_image_field(t, 0)
        img = Image.open(io.BytesIO(b))
        print('esempio: label', t.column('label')[0].as_py(), '| path', p,
              '| formato', img.format, img.mode, img.size)
        return

    # 1) Legge solo le etichette di tutti gli shard -> elenco (shard, riga) per classe
    by_class = defaultdict(list)
    for s, path in enumerate(shards):
        labels = open_table(path).column('label').to_numpy()
        for r, lab in enumerate(labels):
            by_class[int(lab)].append((s, r))
        if s % 20 == 0:
            print('etichette lette: shard %d/%d' % (s + 1, len(shards)), flush=True)
    print('classi trovate:', len(by_class), flush=True)

    # 2) Sceglie a caso (seed fisso, quindi riproducibile) per_class immagini per classe.
    #    Ne prende qualcuna in piu' per poter scartare le immagini non RGB/grigie.
    rng = random.Random(args.seed)
    chosen = defaultdict(list)  # shard -> [(riga, label)]
    for lab, items in by_class.items():
        k = min(len(items), args.per_class + 10)
        for s, r in rng.sample(items, k):
            chosen[s].append((r, lab))

    # 3) Scrive i JPEG, fermandosi a per_class immagini valide per classe
    written = defaultdict(int)
    skipped = 0
    for s in sorted(chosen):
        t = open_table(shards[s])
        for r, lab in sorted(chosen[s]):
            if written[lab] >= args.per_class:
                continue
            b, p = get_image_field(t, r)
            img = Image.open(io.BytesIO(b))
            # Il codice originale gestisce solo immagini RGB o in scala di grigi:
            # le poche immagini CMYK di ImageNet darebbero 4 canali e un errore.
            if img.mode not in ('RGB', 'L'):
                skipped += 1
                continue
            folder = os.path.join(args.out, class_folder(lab, p))
            os.makedirs(folder, exist_ok=True)
            fname = os.path.basename(p) if p else 's%03d_r%06d.JPEG' % (s, r)
            if not fname.endswith('.JPEG'):
                fname = os.path.splitext(fname)[0] + '.JPEG'
            out_path = os.path.join(folder, fname)
            if img.format == 'JPEG':
                with open(out_path, 'wb') as fh:   # bytes originali, nessuna ricompressione
                    fh.write(b)
            else:
                img.save(out_path, 'JPEG', quality=95)
            written[lab] += 1
        if s % 20 == 0:
            print('immagini scritte: shard %d/%d, totale %d' % (s + 1, len(shards), sum(written.values())), flush=True)

    counts = np.array([written[l] for l in by_class])
    print('FINE: %d immagini, %d classi, min/max per classe %d/%d, scartate (non RGB/L) %d'
          % (counts.sum(), len(counts), counts.min(), counts.max(), skipped), flush=True)


if __name__ == '__main__':
    main()
