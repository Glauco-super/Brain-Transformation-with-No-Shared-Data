"""
NSD, secondo passo: da dove vengono le 10000 colonne di FMRI_ordered_10k della demo.

Gia' verificato (preproc/check_nsd_rows_images.py): righe = immagini di subjectim viste
almeno una volta nelle sessioni 1-37, label = subjectim[pos], immagini = image_prepare.

Qui, su un sottoinsieme di righe (immagini), si calcolano dai beta fsaverage (.mgh) delle
sessioni 1-37 le risposte medie sulle ripetizioni per TUTTI i vertici (lh poi rh, 327684), con
piu' normalizzazioni, e si cerca per ogni colonna della demo il vertice piu' correlato.
Poi si verifica quale normalizzazione riproduce i valori, e con quale criterio e in quale
ordine sono stati scelti i 10000 vertici.

Normalizzazioni confrontate (media sulle ripetizioni di ogni immagine):
  A  z-score per vertice all'interno di ogni sessione (750 trial), poi media
  B  z-score per vertice su tutti i trial delle 37 sessioni, poi media
  C  beta grezzi (e /300), poi media

Uso (dalla cartella del repository, ambiente b2b_nvtf; CPU, ~32 GB, qualche ora):
    python preproc/check_nsd_fmri.py [--subj 1] [--n-rows 500] [--out <file .npz>]
"""
import argparse
import glob
import os
import time

import numpy as np
import nibabel as nib
import scipy.io
import torch

TAG = '_fsaverage_betas_fithrf_GLMdenoise_RR'
TPS = 750          # trial per sessione
N_SESS = 37


def find_one(root, pattern):
    hits = sorted(glob.glob(os.path.join(root, '**', pattern), recursive=True))
    return hits


def tload(path):
    x = torch.load(path, map_location='cpu')
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


def load_session(beta_dir, sess):
    """Beta di una sessione: (750 trial, 327684 vertici), lh poi rh, float32."""
    parts = []
    for hemi in ('lh', 'rh'):
        img = nib.load(os.path.join(beta_dir, '%s.betas_session%02d.mgh' % (hemi, sess)))
        if sess == 1 and hemi == 'lh':
            print('  %s: forma %s, tipo su disco %s' % (os.path.basename(img.get_filename()),
                  img.shape, img.header.get_data_dtype()), flush=True)
        a = np.asarray(img.dataobj, dtype=np.float32).reshape(img.shape[0], -1)   # vertici x trial
        parts.append(a.T)
    return np.concatenate(parts, axis=1)


def colnorm(x):
    """Colonne a media 0 e norma 1 (per correlazioni con un prodotto matriciale)."""
    x = x - x.mean(0, keepdims=True)
    n = np.linalg.norm(x, axis=0, keepdims=True)
    n[n == 0] = 1
    return x / n


def compare(name, mine, demo):
    d = np.abs(mine - demo)
    slope = float((mine * demo).sum() / max((mine * mine).sum(), 1e-12))   # demo ~ slope * mine
    r = np.corrcoef(mine.ravel(), demo.ravel())[0, 1]
    print('  %-34s diff max %.5f  diff media %.6f  corr %.6f  pendenza demo/mio %.5f'
          % (name, d.max(), d.mean(), r, slope), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subj', type=int, default=1)
    ap.add_argument('--nsd-dir', default='/scratch_share/islab/glauco.rinaldi/datasets/NSD')
    ap.add_argument('--demo-dir', default='/scratch_share/islab/glauco.rinaldi/demo_data/Processed_data/NSD')
    ap.add_argument('--n-rows', type=int, default=500)
    ap.add_argument('--out', default='/scratch_share/islab/glauco.rinaldi/processed/NSD/check_nsd_fmri_subj01.npz')
    a = ap.parse_args()
    s = a.subj
    t0 = time.time()

    # --- demo e disegno sperimentale
    D = a.demo_dir + '/'
    fmri_demo = tload(D + 'FMRI_ordered_10k_subj0%d%s.pt' % (s, TAG))
    inc = tload(D + 'iculding_ind_mask_ordered_10k_subj0%d%s.pt' % (s, TAG)).ravel().astype(np.int64)
    print('demo fMRI', fmri_demo.shape, flush=True)
    exp = scipy.io.loadmat(find_one(a.nsd_dir, 'nsd_expdesign.mat')[0])
    master = exp['masterordering'].ravel().astype(np.int64) - 1          # posizione 10k (base 0) di ogni trial

    hits = [h for h in find_one(a.nsd_dir, 'lh.betas_session01.mgh')
            if 'subj%02d' % s in h and 'fsaverage' in h and 'betas_fithrf_GLMdenoise_RR' in h]
    if not hits:
        raise SystemExit('beta fsaverage di subj%02d non trovati sotto %s' % (s, a.nsd_dir))
    beta_dir = os.path.dirname(hits[0])
    print('cartella beta:', beta_dir, flush=True)

    # --- sottoinsieme di righe (immagini) e trial corrispondenti nelle sessioni 1-37
    rows = np.linspace(0, len(inc) - 1, a.n_rows).astype(int)
    pos2row = {p: j for j, p in enumerate(inc[rows])}
    n_tr = N_SESS * TPS
    nv = None
    sum_z = sum_raw = None
    cnt = np.zeros(len(rows))
    g_sum = g_sq = None

    print('\nlettura delle sessioni 1-%d...' % N_SESS, flush=True)
    for sess in range(1, N_SESS + 1):
        x = load_session(beta_dir, sess)                                  # 750 x V
        if nv is None:
            nv = x.shape[1]
            sum_z = np.zeros((len(rows), nv), np.float64); sum_raw = np.zeros_like(sum_z)
            g_sum = np.zeros(nv, np.float64); g_sq = np.zeros(nv, np.float64)
            print('  vertici totali (lh+rh):', nv, '| valori sessione 1: min %.1f max %.1f media %.3f'
                  % (x.min(), x.max(), x.mean()), flush=True)
        g_sum += x.sum(0, dtype=np.float64); g_sq += (x.astype(np.float64) ** 2).sum(0)
        mu = x.mean(0); sd = x.std(0); sd[sd == 0] = 1
        trials = np.arange((sess - 1) * TPS, sess * TPS)
        for k, t in enumerate(trials):
            j = pos2row.get(master[t])
            if j is not None:
                sum_raw[j] += x[k]
                sum_z[j] += (x[k] - mu) / sd
                cnt[j] += 1
        print('  sessione %2d letta (%.0f s)' % (sess, time.time() - t0), flush=True)
        del x

    avg_raw = sum_raw / cnt[:, None]
    A = (sum_z / cnt[:, None]).astype(np.float32)
    g_mu = g_sum / n_tr; g_sd = np.sqrt(np.maximum(g_sq / n_tr - g_mu ** 2, 0)); g_sd[g_sd == 0] = 1
    B = ((avg_raw - g_mu) / g_sd).astype(np.float32)
    print('ripetizioni per immagine nel sottoinsieme: min %d max %d' % (cnt.min(), cnt.max()), flush=True)

    # --- 1) per ogni colonna della demo, il vertice piu' correlato (su A)
    print('\n1) Ricerca dei vertici (correlazione sulle %d immagini del sottoinsieme)...' % len(rows), flush=True)
    dz = colnorm(fmri_demo[rows].astype(np.float32))
    az = colnorm(A)
    best = np.full(dz.shape[1], -1); best_r = np.full(dz.shape[1], -2.0, np.float32)
    second_r = np.full(dz.shape[1], -2.0, np.float32)
    step = 20000
    for v0 in range(0, nv, step):
        c = dz.T @ az[:, v0:v0 + step]                                    # 10000 x step
        top = c.argmax(1); top_r = c[np.arange(len(top)), top]
        c[np.arange(len(top)), top] = -2
        sec = c.max(1)
        upd = top_r > best_r
        second_r = np.where(upd, np.maximum(best_r, sec), np.maximum(second_r, top_r))
        best[upd] = v0 + top[upd]; best_r[upd] = top_r[upd]
    print('  corr del vertice migliore: min %.4f mediana %.4f | seconda migliore: mediana %.4f'
          % (best_r.min(), np.median(best_r), np.median(second_r)), flush=True)
    print('  vertici distinti: %d su %d | in lh: %d, in rh: %d'
          % (len(np.unique(best)), len(best), (best < nv // 2).sum(), (best >= nv // 2).sum()), flush=True)
    print('  ordine delle colonne crescente per indice di vertice? %s' % bool(np.all(np.diff(best) > 0)), flush=True)

    # --- 2) quale normalizzazione riproduce i valori
    print('\n2) Valori della demo sulle colonne trovate:', flush=True)
    dm = fmri_demo[rows]
    compare('A z-score per sessione, media', A[:, best], dm)
    compare('B z-score su 37 sessioni, media', B[:, best], dm)
    compare('C grezzi, media', avg_raw[:, best].astype(np.float32), dm)
    compare('C grezzi/300, media', (avg_raw[:, best] / 300).astype(np.float32), dm)

    # --- 3) criterio di scelta: file ncsnr di NSD, se presenti
    print('\n3) Criterio di scelta dei vertici:', flush=True)
    nc_files = [os.path.join(beta_dir, '%s.ncsnr.mgh' % h) for h in ('lh', 'rh')]
    if all(os.path.exists(f) for f in nc_files):
        ncsnr = np.concatenate([np.asarray(nib.load(f).dataobj, np.float32).ravel() for f in nc_files])
        top = np.argsort(ncsnr)[::-1][:len(best)]
        print('  ncsnr dei vertici scelti: min %.4f mediana %.4f | ncsnr del 10000-esimo vertice NSD: %.4f'
              % (ncsnr[best].min(), np.median(ncsnr[best]), ncsnr[top[-1]]), flush=True)
        print('  sovrapposizione con i 10000 vertici a ncsnr piu alto (file NSD): %d' % len(np.intersect1d(top, best)), flush=True)
        print('  colonne in ordine di ncsnr decrescente? %s' % bool(np.all(np.diff(ncsnr[best]) <= 0)), flush=True)
    else:
        ncsnr = None
        print('  file ncsnr di NSD non presenti in', beta_dir, flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, rows=rows, best=best, best_r=best_r, second_r=second_r,
             g_mu=g_mu.astype(np.float32), g_sd=g_sd.astype(np.float32),
             **({'ncsnr': ncsnr} if ncsnr is not None else {}))
    print('\nsalvato', a.out, '| tempo totale %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
