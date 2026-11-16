"""
Reimplementazione di scipy.misc.imread / imresize / imsave.

Queste funzioni sono state rimosse da SciPy (imread/imsave in 1.2,
imresize in 1.3) e non esistono nelle versioni di SciPy disponibili per
Python 3.8. Il codice originale le usava con SciPy 1.2.1: qui sono
riscritte con Pillow seguendo il comportamento di scipy 1.1/1.2,
in particolare la conversione a uint8 tramite `bytescale` (riscalatura
min-max a 0-255 per gli array non uint8).
"""
import numpy as np
from PIL import Image

__all__ = ['bytescale', 'toimage', 'fromimage', 'imread', 'imresize', 'imsave']

# Stessi codici di filtro usati da scipy.misc.imresize (costanti di PIL)
_INTERP = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """Converte un array in uint8, riscalando [cmin, cmax] -> [low, high].
    Gli array gia' uint8 sono restituiti invariati (come in scipy)."""
    data = np.asarray(data)
    if data.dtype == np.uint8:
        return data
    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()
    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1
    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, channel_axis=None):
    """Array numpy -> immagine PIL (solo i casi usati nel repository:
    scala di grigi 2D, oppure 3D con 3 o 4 canali)."""
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    if data.ndim == 2:
        return Image.fromarray(bytescale(data), mode='L')
    if data.ndim == 3:
        if channel_axis is None:
            shape = np.asarray(data.shape)
            ca = np.flatnonzero(shape == 3)
            if len(ca) == 0:
                ca = np.flatnonzero(shape == 4)
            if len(ca) == 0:
                raise ValueError("Could not find channel dimension.")
            ca = ca[0]
        else:
            ca = channel_axis
        numch = data.shape[ca]
        if numch not in (3, 4):
            raise ValueError("Channel axis dimension is not valid.")
        bytedata = np.ascontiguousarray(np.moveaxis(bytescale(data), ca, -1))
        return Image.fromarray(bytedata, mode='RGB' if numch == 3 else 'RGBA')
    raise ValueError("'arr' does not have a suitable array shape for any mode.")


def fromimage(im, flatten=False):
    """Immagine PIL -> array numpy, con le stesse conversioni di scipy."""
    im.load()
    if im.mode == 'P':
        im = im.convert('RGBA' if 'transparency' in im.info else 'RGB')
    if flatten:
        im = im.convert('F')
    elif im.mode == '1':
        im = im.convert('L')
    return np.array(im)


def imread(name, flatten=False):
    """Legge un file immagine e lo restituisce come array (di solito uint8)."""
    return fromimage(Image.open(name), flatten=flatten)


def imresize(arr, size, interp='bilinear'):
    """Ridimensiona un array-immagine. Restituisce SEMPRE uint8, come scipy.
    size: int = percentuale, float = frazione, tupla/lista = (righe, colonne)."""
    im = toimage(arr)
    if isinstance(size, (int, np.integer)):
        size = tuple((np.array(im.size) * (size / 100.0)).astype(int))
    elif isinstance(size, (float, np.floating)):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])  # PIL vuole (larghezza, altezza)
    return fromimage(im.resize(size, resample=_INTERP[interp]))


def imsave(name, arr, format=None):
    """Salva un array come immagine (gli array non uint8 vengono riscalati
    min-max a 0-255, come in scipy)."""
    im = toimage(arr, channel_axis=2 if np.ndim(arr) == 3 else None)
    if format is None:
        im.save(name)
    else:
        im.save(name, format)
