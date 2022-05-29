# -*- coding: utf-8 -*-
from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
import cv2

def _analyse_dbscan(self, eps=1, min_samples=10, color_weight=2):
    """DBSCAN: Generate DBSCAN-Cluster for image

    Args:
        eps (float): Maximum distance to be seen as neighbours.
        min_samples (float): Minimum number of pixels in Cluster.
        color_weight (float): Max-weight for color-values after scaling.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples,
                   n_jobs=-2)
    _analysis_wrapper(self, model, color_weight)

class SLIC:
    """Wrapper-Class to perform SLIC-Superpixel extraction with scikit-learn interface.

    Attributes:
        compactness (float): Ratio of spatial to color-weighting. Higher values indicate higher spatial weights.
    """

    def __init__(self, compactness=0.25):
        """Prep model"""
        self.compactness=compactness
        self.start_label=0

    def fit_predict(self, img, flatten=True):
        """scikit-learn-like API to fit SLIC"""
        out = (
            img.iloc[:, [0,1,2]]
            .pivot(index="y", columns="x", values="value")
            .to_numpy()
        )
        out = segmentation.slic(out,
                                compactness=self.compactness,
                                start_label=self.start_label)
        if flatten:
            return(out.flatten())
        else:
            return(out)


class NCuts:
    """Wrapper-Class to perform Normalized Cuts-Superpixel extraction with scikit-learn interface.

    Attributes:
        compactness (float): Ratio of spatial to color-weighing. Higher values indicate higher spatial weights.
        thresh (float): Stop-criterion for cuts on teh sub-graph.
        num_cuts (int): Number of cuts to test to determine optimal one.
        sigma(float): Maximum distance of two colors to be treated as similar.
    """

    def __init__(self, compactness=0.25, thresh=.001, num_cuts=10, sigma=100):
        """Prep model"""
        self.compactness=compactness
        self.start_label=0
        self.thresh=thresh
        self.num_cuts=num_cuts
        self.sigma=sigma

    def fit_predict(self, img):
        """scikit-learn-like API to fit NCut"""
        out = (
            img.iloc[:, [0,1,2]]
            .pivot(index="y", columns="x", values="value")
            .to_numpy()
        )
        slic = SLIC(self.compactness)
        slic_results = slic.fit_predict(img, flatten=False)
        g = graph.rag_mean_color(out, slic_results, mode='similarity', sigma=self.sigma)
        out = graph.cut_normalized(slic_results, g,
                                   thresh=self.thresh, num_cuts=self.num_cuts)
        return(out.flatten())

if __name__=='__main__':
    import sys
    import matplotlib.pyplot as plt
    import pandas as pd

    sys.path.append("scripts")
    from model import GrainImage

    beans = GrainImage("../data/17 0 650 cbs 01_002.tif")
    img = beans.origImage
    model = NCuts()
    img = pd.read_csv('test.csv').iloc[:,1:4]
    self = model
    model.fit_predict(img)

    ax[0].imshow(labels1)
    ax[1].imshow(out2)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
