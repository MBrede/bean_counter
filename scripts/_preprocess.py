import cv2
from scipy.signal import savgol_filter


def _prep_sav_filter(self, w_size=21, grad=9):
    """Savitzky-Golay-Filter: Apply Savitzky-Golay-Filter to smooth image.

    A filter that smoothes the picture by exchanging the grey-values of an
    image with the results of a polynomial regression.
    Args:
        w_size (int): Window to be smoothed at a time. Even even w_size is
                      given, w_size + 1 is used.
        grad (int): Polynom to use for smoothing. If grad > w_size, w_size-1
                    is used.
    """
    if w_size % 2 == 0:
        w_size += 1
    if grad >= w_size:
        grad = w_size - 1
    self.modImg = savgol_filter(self.modImg, w_size, grad, axis=0)
    self.modImg = savgol_filter(self.modImg, w_size, grad, axis=1)
    self.modImg = self.modImg.astype("uint8")


def _prep_hist_equalization(self):
    """Histogram-Equalizer: Equalize the histogram to enhance contrast.

    A method that stretches histogram of intesities to use
    as much of the grey-spectrum as possible. Increases an
    images contrast.
    """
    self.modImg = cv2.equalizeHist(self.modImg)


def _prep_gauss_blurr(self, k_size=(5, 5)):
    """Gauss-Filter: Blurr the image using a Gaussian filter.

    Args:
        k_size (tuple): Kernel-size in width and height as tuple. If input
                        is even, gets changed by +1

    """
    if k_size[0] % 2 == 0:
        k_size = (k_size[0] + 1, k_size[1])
    if k_size[1] % 2 == 0:
        k_size = (k_size[0], k_size[1] + 1)
    self.modImg = cv2.GaussianBlur(self.modImg, k_size, 0)


def _prep_nlMeans_denoising(self, h=3, temp_winSize=7, search_winSize=21):
    """Non-Local-Mean-Denoiser: Denoise the image by using non-local means

    Removes gaussian white noise.
    """
    self.modImg = cv2.fastNlMeansDenoising(
        self.modImg,
        dst=None,
        h=h,
        templateWindowSize=temp_winSize,
        searchWindowSize=search_winSize,
    )


def preprocess(self, type, *args, **kwargs):
    """Wrapper to ease the preprocess-interface and make batching easier.

    Args:
        type (type): Preprocessing-filter to apply.
        **kwargs (type): Settings for filter.

    """
    if type == "gauss":
        _prep_gauss_blurr(self, *args, **kwargs)
    elif type == "sav":
        _prep_sav_filter(self, *args, **kwargs)
    elif type == "hist":
        _prep_hist_equalization(self, *args, **kwargs)
    elif type == "nlMeans":
        _prep_nlMeans_denoising(self, *args, **kwargs)
    else:
        raise ValueError("%s is not implemented yet" % type)
