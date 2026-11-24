"""
Diagnostica dei dati NSD usati in validazione.

Nella demo della trasformazione le loss di validazione che coinvolgono l'encoder NSD
(encoding_2, rec_FMRI_2_from_enc_1) erano ~1000 volte quelle di training, pur con
l'encoder NSD congelato. Questo script controlla, per il soggetto NSD indicato:
  1) forma e scala di immagini e fMRI nelle tre parti (train / val / test), divise con
     le stesse maschere e la stessa funzione (Split_data_new) usate in addestramento;
  2) l'errore (MSE, come l2_loss) e la correlazione dell'encoder NSD pre-addestrato
     su train e su test, con il modello in modalita' inferenza come in validazione.

Uso (dalla cartella del repository, ambiente b2b_nvtf):
    python preproc/check_nsd_split.py [soggetto] [file_pesi_encoder]
Predefiniti: soggetto 1, pesi data/NSD_encoders/encoder_weights_subj1_NSD_exp_s_700_exp_ns_6400_ref.hdf5
"""
import os
import sys
# Lo script sta in preproc/: aggiungo la cartella del repository al percorso di ricerca dei
# moduli, cosi' 'Utils' e 'Models' si importano anche lanciandolo come preproc/check_nsd_split.py.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

from Utils.image_loss import vgg_loss
from Utils.train_utils import *   # Load_NSD_data, Split_data_new, encoder_param, encoder_ml_seperable

subj = int(sys.argv[1]) if len(sys.argv) > 1 else 1
weights = sys.argv[2] if len(sys.argv) > 2 else \
    'data/NSD_encoders/encoder_weights_subj%d_NSD_exp_s_700_exp_ns_6400_ref.hdf5' % subj


def stats(name, a):
    a = np.asarray(a)
    af = a.astype(np.float64)
    print('%-14s shape=%-22s dtype=%-8s min=%10.4f max=%10.4f mean=%9.4f std=%9.4f nan=%d'
          % (name, str(a.shape), a.dtype, np.nanmin(af), np.nanmax(af), np.nanmean(af), np.nanstd(af),
             int(np.isnan(af).sum())), flush=True)


def evaluate(name, model, images, fmri, n=None):
    if n is not None:
        images, fmri = images[:n], fmri[:n]
    pred = model.predict(images.astype(np.float32), batch_size=16)
    mse_per_sample = ((pred - fmri) ** 2).mean(axis=1)   # come l2_loss: MSE sui voxel
    corr = [np.corrcoef(pred[:, i], fmri[:, i])[0, 1] for i in range(fmri.shape[1])]
    print('%-6s n=%4d | MSE media %.4f (max per esempio %.4f) | corr media per voxel %.4f'
          % (name, len(images), mse_per_sample.mean(), mse_per_sample.max(), np.nanmean(corr)), flush=True)
    stats('  predizione', pred)
    return mse_per_sample


print('== Caricamento NSD soggetto %d' % subj, flush=True)
FMRI, Images, Labels, train_mask_s, val_mask_s, test_mask_s, train_mask, val_mask, test_mask, \
    shared_mask, shared_all_indexes = Load_NSD_data(subj)
tr_im, va_im, te_im, tr_f, va_f, te_f, tr_l, va_l, te_l = Split_data_new(
    Images, FMRI, Labels[0], train_mask, val_mask, test_mask)

print('\n== 1) Scala dei dati per parte', flush=True)
for part, im, f in (('train', tr_im, tr_f), ('val', va_im, va_f), ('test', te_im, te_f)):
    stats(part + ' immagini', im)
    stats(part + ' fMRI', f)

print('\n== 2) Encoder NSD pre-addestrato (modalita inferenza)', flush=True)
print('pesi: ' + weights, flush=True)
num_voxels = FMRI.shape[1]
scale = 0.1   # come in calc_data_type_param per i soggetti NSD
param = encoder_param(num_voxels, c2f_l1=scale * 1e-7, c2f_gl=scale * 1e-7, lc_l1=scale * 1e-7,
                      lc_l1_out=scale * 1e-7, conv_l1_reg=scale * 1e-5, conv_l2_reg=scale * 0.001)
encoder = encoder_ml_seperable(param, vgg_loss, ch_mult=1, name='encoder_2')
encoder.load_weights(weights)
mse_tr = evaluate('train', encoder, tr_im, tr_f, n=300)
mse_te = evaluate('test', encoder, te_im, te_f)

print('\n== 3) Esempi di test con l\'errore piu alto', flush=True)
worst = np.argsort(mse_te)[::-1][:5]
for i in worst:
    print('  test #%4d label %6d  MSE %.4f  | immagine min/max %.3f/%.3f | fMRI min/max %.3f/%.3f'
          % (i, int(te_l[i]), mse_te[i], te_im[i].min(), te_im[i].max(), te_f[i].min(), te_f[i].max()), flush=True)

print('\n== 4) Stesso encoder con le immagini NSD divise per 255 (scala 0-1)', flush=True)
# Il primo livello dell'encoder (encoder_ml_seperable) moltiplica l'input per 255:
# si aspetta immagini in [0, 1]. Se con /255 l'MSE torna ~1, le immagini NSD vanno riscalate.
if tr_im.max() > 1.5:
    evaluate('train/255', encoder, tr_im / 255.0, tr_f, n=300)
    evaluate('test/255', encoder, te_im / 255.0, te_f)
else:
    print('immagini gia in [0, 1] (Load_NSD_data le riscala): controllo non necessario')

print('\n== 5) Scala delle immagini GOD (per confronto)', flush=True)
god = np.load('data/Processed_data/GOD/images_112.npz')
stats('GOD train img', god['train_images'])
stats('GOD test img', god['test_images'])
