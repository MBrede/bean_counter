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
        compactness (float): 
    """



path = '/home/brede/MEGA/Data_Science/bean_counter/data/17 2 650 cbs_04.tif'
img = cv2.imread(path, 0)

labels1 = segmentation.slic(img, compactness=0.25,start_label=1)
out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)

g = graph.rag_mean_color(img, labels1, mode='similarity')
labels2 = graph.cut_normalized(labels1, g)
out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].imshow(out1)
ax[1].imshow(out2)

for a in ax:
    a.axis('off')

plt.tight_layout()
