"""
Estrae un sottoinsieme di ImageNet-1k (train) dai file Arrow di Hugging Face
presenti sul cluster e lo salva come JPEG in una cartella per classe:

    <out_dir>/<wnid o nNNNN>/<nome_file>.JPEG

E' la struttura che si aspetta Utils/batch_generator.py
(batch_generator_external_images), che sceglie le immagini dalle
sottocartelle il cui nome inizia per 'n'.

Le immagini di test di GOD (che vengono da ImageNet) sono sempre escluse:
se finissero tra le immagini esterne, l'insegnante darebbe allo studente una
"risposta" proprio sulle immagini su cui viene valutato. Con --exclude-god-train
si escludono anche le immagini di training di GOD (facoltativo: lo studente le
vede gia' con la sua fMRI, quindi la sovrapposizione non falsa la valutazione).
L'esclusione avviene al momento della scrittura, usando le immagini di riserva:
a parita' di seed, se nessuna immagine scelta e' da escludere, l'insieme
estratto e' identico a quello ottenuto senza il controllo.

Uso:
    python preproc/extract_imagenet_subset.py --inspect
    python preproc/extract_imagenet_subset.py --per-class 50 --out <cartella>
    python preproc/extract_imagenet_subset.py --per-class 50 --exclude-god-train
"""
import argparse
import glob
import io
import os
import random
import re
from collections import defaultdict

import numpy as np
import pyarrow as pa
from PIL import Image

SRC_DIR = '/scratch_share/datasets/ILSVRC___imagenet-1k'
GOD_IMAGES_DIR = '/scratch_share/islab/glauco.rinaldi/datasets/GOD_Ho2022/images'

# Identificativo ImageNet: classe (n + 8 cifre) _ numero, es. n01443537_22563.
# E' lo stesso nei nomi dei file di GOD e in quelli dei file Arrow di ImageNet.
ID_PATTERN = re.compile(r'n\d{8}_\d+')


def imagenet_id(name):
    """Estrae l'identificativo ImageNet dall'inizio di un nome di file (o None)."""
    if not name:
        return None
    m = ID_PATTERN.match(os.path.basename(name))
    return m.group() if m else None


def ids_in_folder(folder):
    """Insieme degli identificativi ImageNet dei file in una cartella (e sottocartelle)."""
    out = set()
    for root, _, files in os.walk(folder):
        for f in files:
            i = imagenet_id(f)
            if i:
                out.add(i)
    return out


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
    ap.add_argument('--god-images', default=GOD_IMAGES_DIR,
                    help='cartella delle immagini GOD, con le sottocartelle training/ e test/')
    ap.add_argument('--exclude-god-train', action='store_true',
                    help='esclude anche le immagini di training di GOD (il test e\' sempre escluso)')
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

    # 0) Identificativi da escludere: test di GOD sempre, training solo se richiesto
    exclude = ids_in_folder(os.path.join(args.god_images, 'test'))
    if len(exclude) == 0:
        raise SystemExit('Nessuna immagine di test di GOD trovata in ' + args.god_images +
                         '/test: controlla --god-images (l\'esclusione del test e\' obbligatoria).')
    print('immagini di test GOD da escludere:', len(exclude), flush=True)
    if args.exclude_god_train:
        train_ids = ids_in_folder(os.path.join(args.god_images, 'training'))
        print('immagini di training GOD da escludere:', len(train_ids), flush=True)
        exclude |= train_ids

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
    excluded = 0
    no_id = 0
    for s in sorted(chosen):
        t = open_table(shards[s])
        for r, lab in sorted(chosen[s]):
            if written[lab] >= args.per_class:
                continue
            b, p = get_image_field(t, r)
            iid = imagenet_id(p)
            if iid is None:
                no_id += 1          # senza nome non si puo' controllare la sovrapposizione con GOD
            elif iid in exclude:
                excluded += 1
                continue
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
    print('FINE: %d immagini, %d classi, min/max per classe %d/%d, scartate (non RGB/L) %d, '
          'escluse perche\' in GOD %d'
          % (counts.sum(), len(counts), counts.min(), counts.max(), skipped, excluded), flush=True)
    if no_id:
        print('ATTENZIONE: %d immagini senza identificativo ImageNet nel nome: '
              'non e\' stato possibile verificarne la sovrapposizione con GOD' % no_id, flush=True)
    short = int((counts < args.per_class).sum())
    if short:
        print('ATTENZIONE: %d classi con meno di %d immagini (riserve esaurite)' % (short, args.per_class), flush=True)


if __name__ == '__main__':
    main()
