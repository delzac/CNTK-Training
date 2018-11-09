import cntk as C
from cntk.layers import Convolution2D, MaxPooling


def UpSampling2D(x):
    xr = C.reshape(x, (x.shape[0], x.shape[1], 1, x.shape[2], 1))
    xx = C.splice(xr, xr, axis=-1)  # axis=-1 refers to the last axis
    xy = C.splice(xx, xx, axis=-3)  # axis=-3 refers to the middle axis
    r = C.reshape(xy, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2))
    return r
