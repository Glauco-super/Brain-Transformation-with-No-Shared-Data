"""
Ispeziona i file .pt preprocessati del demo (NSD subj01) per capire
ESATTAMENTE il formato che Load_NSD_data() si aspetta.
Uso (sull'HPC, env Brain2Brain, dalla root del repo):
    python preproc/inspect_demo_nsd.py /percorso/a/nsd_expdesign.mat
"""
import sys
import numpy as np
import torch
import scipy.io

D = 'data/Processed_data/NSD/'
TAG = '_fsaverage_betas_fithrf_GLMdenoise_RR'

def describe(name, x):
    if torch.is_tensor(x):
        x = x.numpy()
    x = np.asarray(x)
    print('\n== %s' % name)
    print('   shape=%s dtype=%s' % (x.shape, x.dtype))
    if x.size and np.issubdtype(x.dtype, np.number) or x.dtype == bool:
        xf = x.astype(np.float64)
        print('   min=%.4f max=%.4f mean=%.4f std=%.4f  n_nonzero=%d'
              % (xf.min(), xf.max(), xf.mean(), xf.std(), np.count_nonzero(xf)))
    return x

# 1) i 3 file per-soggetto davvero usati da train_encoder
fmri = describe('FMRI_ordered_10k', torch.load(D + 'FMRI_ordered_10k_subj01' + TAG + '.pt'))
imgs = describe('Images_ordered_10k', torch.load(D + 'Images_ordered_10k_subj01' + TAG + '.pt'))
labs = describe('Images_ordered_10k_labels', torch.load(D + 'Images_ordered_10k_labels_subj01' + TAG + '.pt'))

# z-score? guardo media/std per voxel
vm, vs = fmri.mean(0), fmri.std(0)
print('\n   FMRI per-voxel mean: media %.4f (range %.4f..%.4f)' % (vm.mean(), vm.min(), vm.max()))
print('   FMRI per-voxel std : media %.4f (range %.4f..%.4f)' % (vs.mean(), vs.min(), vs.max()))
print('   voxel costanti (std=0): %d' % (vs == 0).sum())

lab = labs.reshape(-1).astype(np.int64)
print('\n   labels primi 15: %s' % lab[:15])
print('   labels min/max: %d / %d   unici: %d   ordinati crescenti? %s'
      % (lab.min(), lab.max(), len(np.unique(lab)), np.all(np.diff(lab) > 0)))

# 2) confronto con il design sperimentale NSD
exp = scipy.io.loadmat(sys.argv[1])
subjectim = exp['subjectim'][0]          # 10000 id 73k (1-based) del soggetto 1
master = exp['masterordering'][0]        # 30000 trial, indici 1-based nei 10k
shared = exp['sharedix'][0]              # 1000 id 73k (1-based) condivisi
for off in (0, 1):
    print('   labels+%d dentro subjectim: %d / %d, dentro sharedix: %d'
          % (off, np.isin(lab + off, subjectim).sum(), len(lab), np.isin(lab + off, shared).sum()))
# quante immagini uniche si vedono nelle prime N sessioni (750 trial/sessione)
for nses in (37, 40):
    seen = np.unique(subjectim[master[:nses * 750] - 1])
    print('   immagini uniche viste in %d sessioni: %d' % (nses, len(seen)))

# 3) le maschere globali (indicizzate sui 73k id, uguali per tutti i soggetti)
for n in ['train_mask', 'val_mask', 'test_mask', 'shared_mask',
          'train_mask_shared', 'val_mask_shared', 'test_mask_shared']:
    describe(n, torch.load(D + n + '.pt'))

# 4) file per-soggetto caricati ma NON usati per l'encoder (devono solo esistere)
for n in ['shared_mask_subj01.pt', 'shared_mask_ordered_10k_subj01' + TAG + '.pt',
          'iculding_ind_mask_ordered_10k_subj01' + TAG + '.pt', 'shared_all_indexes_subj01.pt']:
    describe(n, torch.load(D + n))
