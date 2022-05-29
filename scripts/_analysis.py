from sklearn.cluster import DBSCAN  # , OPTICS
import _nCuts as nc
import pandas as pd
import numpy as np
import inspect
from skimage.measure import EllipseModel
import regex as re


def min_max_scale(x, min=0, max=1):
    """Rescale values to lie between limits.

    Args:
        x (numpy.array): One-dimensional array to scale between min and max.
        min (float): New minimum.
        max (float): New maximum.

    Returns:
        numpy.array: Rescaled values

    """
    return (x - np.min(x)) / (np.max(x) - np.min(x)) * max + min


def cluster_summary(cluster, pixel_side, calcElipses):
    """Calculate cluster statistics.

    A Helper to calculate summary statistics for a given cluster.

    Args:
        cluster (pandas.DataFrame): x,y coordinates and color values of found cluster.

    Returns:
        pandas.DataFrame: sumamry statistics for given cluster.

    """
    out = {'angle': 0, 'major': 0, 'minor': 0}
    out['pixel'] = [len(cluster.index)]
    out['size'] = len(cluster.index) * pixel_side ** 2
    out['diameter'] = 2 * np.sqrt(out['size']/np.pi)
    out['center_x'] = np.mean(cluster['x']) * pixel_side
    out['center_y'] = np.mean(cluster['y']) * pixel_side
    out['color_variance'] = np.var(cluster['value'])
    if calcElipses:
        points = cluster.loc[:, ['x', 'y']].to_numpy()
        ell = EllipseModel()
        if ell.estimate(points):
            out['angle'] = ell.params[4]
            out['major'] = np.max(ell.params[2:4]) * pixel_side
            out['minor'] = np.min(ell.params[2:4]) * pixel_side
    return(pd.DataFrame(out))


def _analysis_wrapper(self, model, color_weight):
    """Analysis Helper

    Function that wraps analysis-steps that are repeatedly taken.

    Args:
        df (pandas.DataFrame): Data to run the analysis on.
        model (sklearn.cluster): Model to use.

    """
    df = pd.DataFrame(
        {
            "value": np.multiply(min_max_scale(self.modImg.flatten(), 0, 100), color_weight),
            "x": min_max_scale(
                np.tile(range(1, self.modImg.shape[1] + 1, 1), self.modImg.shape[0]), 0, 100
            ),
            "y": min_max_scale(
                np.repeat(range(1, self.modImg.shape[0] + 1, 1), self.modImg.shape[1]), 0, 100
            ),
        }
    )
    ret_df = pd.DataFrame(
        {
            "value": self.modImg.flatten(),
            "x": np.tile(range(1, self.modImg.shape[1] + 1, 1), self.modImg.shape[0]),
            "y": np.repeat(range(1, self.modImg.shape[0] + 1, 1), self.modImg.shape[1]),
            "cluster": model.fit_predict(df)
        }
    )
    self.modImg = (
        ret_df.iloc[:, [1, 2, 3]]
        .pivot(index="y", columns="x", values="cluster")
        .to_numpy()
    )
    try:
        def ratio(x,y,max_ratio):
            if np.max(x) == np.min(x) or np.max(y) == np.min(y):
                return False
            ratio = (np.max(x) - np.min(x)) / (np.max(y) - np.min(y))
            if ratio < 1:
                ratio = 1/ratio
            return ratio <= max_ratio

        ratios = ret_df.groupby('cluster')\
                       .apply(lambda row: ratio(row['x'],row['y'],self.max_ratio))\
                       .reset_index()

        self.generated_areas = ret_df\
                                     .loc[ret_df.loc[:, 'cluster'].gt(-1)]\
                                     .loc[ret_df.loc[:, 'cluster']\
                                                .isin(ratios.loc[ratios.iloc[:, 1],
                                                                 'cluster'])]\
                                     .groupby('cluster')\
                                     .apply(lambda cluster: cluster_summary(cluster, self.len_pro_pixel, self.calcElipses))\
                                     .reset_index()\
                                     .drop(columns='level_1')\
                                     .to_dict(orient='list')
    except ValueError:
        raise ValueError('not enough clusters found!')


def build_cdf(data, pixel_len, range_end = None):
    """
    Build cdf with grain-size resolution

    Calculates cdf for all possible grain sizes.

    Args:
        data (numpy.array): array of values to calculate cdf for
        pixel_len (numpy.float64): length of pixel, used as resolution

    Returns:
        tuple: grain sizes and frequency of values less then/equal of given size

    """
    if range_end is None:
        range_end = np.max(data)
    bins = int(range_end/pixel_len**2)
    hist, range = np.histogram(data, bins=bins, range=[0, range_end*1.1])
    value = np.cumsum(hist)
    value = [0] + list(value)
    return (range, np.divide(value, len(data)))


def fit_distro(dist, to_fit, pixel_len):
    """Wrapper to fit scikit continuos functions.

    Uses the maximum likelihood estimation for distributions as implemented in
    scikit stats to fit a density function to the given data.

    Args:
        dist (scipy.stats._continuous_distns): continuous distribution to fit
        to_fit (numpy.array): values to fit

    Returns:
        dict: Dictionary containing the fitted pdf as a lambda function,
              the parameters of the fitted function and the scikit distribution.

    Raises:
        Exception: description

    """
    arg_names = [re.sub('=.+', '', arg)
                 for arg
                 in re.search(r'(?<=pdf\()[^\)]+',
                              inspect.getdoc(dist))[0].split(', ')
                 if arg != 'x']
    args = dist.fit(to_fit)
    args = {a: args[i] for i, a in enumerate(arg_names)}
    return({'function': lambda x: dist.pdf(x, **args),
            'pars': args,
            'dist': dist,
            'emp cdf': build_cdf(to_fit, pixel_len)})


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


def _analyse_SLIC(self, compactness=0.25):
    """SLIC: Generate SLIC-Segmentation for image

    Args:
        compactness (float): Ratio of spatial to color-weighting. Higher values indicate higher spatial weights.
    """
    model = nc.SLIC(compactness=compactness)
    _analysis_wrapper(self, model, color_weight=1)



def _analyse_ncuts(self, compactness=0.25, thresh=.001, num_cuts=10, sigma=100):
    """NCuts: Generate NCuts-Segmentation for image

    Args:
    compactness (float): Ratio of spatial to color-weighing. Higher values indicate higher spatial weights.
    thresh (float): Stop-criterion for cuts on teh sub-graph.
    num_cuts (int): Number of cuts to test to determine optimal one.
    """
    model = nc.NCuts(compactness=compactness, thresh=thresh, num_cuts=num_cuts, sigma=sigma)
    _analysis_wrapper(self, model, color_weight=1)


# def _analyse_optics(self, max_eps=2, min_samples=10, color_weight=2):
#     pass
#     """OPTICS: Generate OPTICS-Cluster for image
#
#     Args:
#         max_eps (float): Maximum of distance range to be seen as neighbours.
#         min_samples (float):  Minimum number of pixels in Cluster.
#         color_weight (float): Max-weight for color-values after scaling.
#
#     """
#     model = OPTICS(max_eps=max_eps, min_samples=min_samples,
#                    n_jobs=-2, metric=self.metric)
#     _analysis_wrapper(self, model, color_weight)


def analyse(self, type, *args, **kwargs):
    """Wrapper to ease the analysis-interface and make batching easier.

    Args:
        type (type): Analysis to apply.
        **kwargs (type): Settings for analysis.

    """
    if type == "dbscan":
        _analyse_dbscan(self, *args, **kwargs)
    elif type == "SLIC":
        _analyse_SLIC(self, *args, **kwargs)
    elif type == "ncuts":
        _analyse_ncuts(self, *args, **kwargs)
    else:
        raise ValueError("%s is not implemented yet" % type)
    if self.generated_areas[self.toDisplay]:
        self.distros['generated'] = fit_distro(self.distro,
                                               to_fit=self.generated_areas[self.toDisplay],
                                               pixel_len=self.len_pro_pixel)


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    sys.path.append("scripts")
    from model import GrainImage

    self = GrainImage("../data/17 0 650 cbs 01_002.tif")
    analyse(self, 'ncuts')
    data = self.generated_areas['size']
    dists = self.distros
