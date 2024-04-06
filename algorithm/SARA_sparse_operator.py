import numpy as np
import pywt

def coef2vec(coef, Nx, Ny):
    """
    Convert wavelet coefficients to an array-type vector, inverse operation of
    vec2coef.

    The initial wavelet coefficients are stocked in a list as follows:
        [cAn, (cHm, cVn, cDn), ..., (cH1, cV1, cD1)],
    and each element is a 2D array.

    After the conversion, the returned vector is as follows:
    [cAn.flatten(), cHn.flatten(), cVn.flatten(), cDn.flatten(), ...,
    cH1.flatten(), cV1.flatten(), cD1.flatten()].
    """
    vec = []
    bookkeeping = []
    for ele in coef:
        if type(ele) == tuple:
            bookkeeping.append((np.shape(ele[0])))
            for wavcoef in ele:
                vec = np.append(vec, np.transpose(wavcoef).flatten())
        else:
            bookkeeping.append((np.shape(ele)))
            vec = np.append(vec, np.transpose(ele).flatten())
    bookkeeping.append((Nx, Ny))
    return vec, bookkeeping

def vec2coef(vec, bookkeeping):
    """
    Convert an array-type vector to wavelet coefficients, inverse operation of
    coef2vec.

    The initial vector is stocked in a 1D array as follows:
    [cAn.flatten(), cHn.flatten(), cVn.flatten(), cDn.flatten(), ...,
    cH1.flatten(), cV1.flatten(), cD1.flatten()].

    After the conversion, the returned wavelet coefficient is in the form of
    the list as follows:
        [cAn, (cHm, cVn, cDn), ..., (cH1, cV1, cD1)],
    and each element is a 2D array. This list can be passed as the argument in
    pywt.waverec2.
    """
    ind = bookkeeping[0][0] * bookkeeping[0][1]
    coef = [np.transpose(np.reshape(vec[:ind], bookkeeping[0]))]
    for ele in bookkeeping[1:-1]:
        indnext = ele[0] * ele[1]
        coef.append((
            np.transpose(np.reshape(vec[ind:ind+indnext], ele)),
            np.transpose(np.reshape(vec[ind+indnext:ind+2*indnext], ele)),
            np.transpose(np.reshape(vec[ind+2*indnext:ind+3*indnext], ele))))
        ind += 3 * indnext
    return coef

def wavedec_asarray(im, wv, level=2):
    wd = pywt.wavedec2(im, wv, level=level, mode='symmetric')
    wd, book = coef2vec(wd, im.shape[0], im.shape[1])
    return wd, book

def waverec_asarray(wd, book, wv='db8'):
    wc = vec2coef(wd, book)
    im = pywt.waverec2(wc, wv, mode='symmetric')
    return im


def SARA_sparse_operator(im, level=2):
    c, b = wavedec_asarray(im, 'db8', level=level)
    ncoef = len(c)
    c_1, b_1 = wavedec_asarray(im, 'db1', level=level)
    ncoef1 = len(c_1)
    c_2, b_2 = wavedec_asarray(im, 'db2', level=level)
    ncoef2 = len(c_2)
    c_3, b_3 = wavedec_asarray(im, 'db3', level=level)
    ncoef3 = len(c_3)
    c_4, b_4 = wavedec_asarray(im, 'db4', level=level)
    ncoef4 = len(c_4)
    c_5, b_5 = wavedec_asarray(im, 'db5', level=level)
    ncoef5 = len(c_5)
    c_6, b_6 = wavedec_asarray(im, 'db6', level=level)
    ncoef6 = len(c_6)
    c_7, b_7 = wavedec_asarray(im, 'db7', level=level)
    ncoef7 = len(c_7)

    def Psit(x):
        out = np.concatenate((
            wavedec_asarray(x, 'db1', level=level)[0],
            wavedec_asarray(x, 'db2', level=level)[0],
            wavedec_asarray(x, 'db3', level=level)[0],
            wavedec_asarray(x, 'db4', level=level)[0],
            wavedec_asarray(x, 'db5', level=level)[0],
            wavedec_asarray(x, 'db6', level=level)[0],
            wavedec_asarray(x, 'db7', level=level)[0],
            wavedec_asarray(x, 'db8', level=level)[0],
            np.transpose(x).flatten(),
        ))
        return out/np.sqrt(9)

    def Psi(y):
        out = waverec_asarray(y[0 : ncoef1], b_1, wv='db1')[:-1, :-1]
        out = out+waverec_asarray(y[ncoef1: ncoef1+ncoef2], b_2, wv='db2')[:-1, :-1]
        out = out+waverec_asarray(y[ncoef1+ncoef2 : ncoef1+ncoef2+ncoef3], b_3, wv='db3')[:-1, :-1]
        out = out+waverec_asarray(y[ncoef1+ncoef2+ncoef3 : ncoef1+ncoef2+ncoef3+ncoef4], b_4, wv='db4')[:-1, :-1]
        out = out+waverec_asarray(y[ncoef1+ncoef2+ncoef3+ncoef4 : ncoef1+ncoef2+ncoef3+ncoef4+ncoef5], b_5, wv='db5')[:-1, :-1]
        out = out+waverec_asarray(y[ncoef1+ncoef2+ncoef3+ncoef4+ncoef5 : ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6], b_6, wv='db6')[:-1, :-1]
        out = out+waverec_asarray(y[ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6 : ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7], b_7, wv='db7')[:-1, :-1]
        out = out+waverec_asarray(y[ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7 : ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7+ncoef], b, wv='db8')[:-1, :-1]
        out = out+np.transpose(np.reshape(y[ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7+ncoef:], im.shape))
        return out/np.sqrt(9)
    return Psi, Psit