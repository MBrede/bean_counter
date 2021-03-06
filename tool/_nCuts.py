# -*- coding: utf-8 -*-
from skimage import segmentation
from skimage.future import graph

class SLIC:
    """Wrapper-Class to perform SLIC-Superpixel extraction with scikit-learn interface.

    Attributes:
        compactness (float): Ratio of spatial to color-weighting. Higher values indicate higher spatial weights.
        n_segments (int): The (approximate) number of labels in the segmented output image.
    """

    def __init__(self, compactness=0.25, n_segments=1000):
        """Prep model"""
        self.compactness=compactness
        self.n_segments=n_segments
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
                                start_label=self.start_label,
                                n_segments=self.n_segments)
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

    def __init__(self, compactness=0.25, n_segments=1000, thresh=.001, num_cuts=10, sigma=100):
        """Prep model"""
        self.compactness=compactness
        self.start_label=0
        self.n_segments=n_segments
        self.thresh=thresh
        self.num_cuts=num_cuts
        self.sigma=sigma

    def fit_predict(self, img, flatten = True):
        """scikit-learn-like API to fit NCut"""
        out = (
            img.iloc[:, [0,1,2]]
            .pivot(index="y", columns="x", values="value")
            .to_numpy()
        )
        slic = SLIC(self.compactness, self.n_segments)
        slic_results = slic.fit_predict(img, flatten=False)
        g = graph.rag_mean_color(out, slic_results, mode='similarity', sigma=self.sigma)
        out = graph.cut_normalized(slic_results, g,
                                   thresh=self.thresh, num_cuts=self.num_cuts)
        if flatten:
            return(out.flatten())
        else:
            return(out)

