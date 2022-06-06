import cv2
from scipy.signal import savgol_filter

path = "../work/imgs/geometric.png"
img = cv2.imread(path, 0)
img = cv2.GaussianBlur(img, (15,15), 0)
cv2.imwrite("../work/imgs/gauss15.png",img)
img = cv2.imread(path, 0)
img = cv2.GaussianBlur(img, (151,151), 0)
cv2.imwrite("../work/imgs/gauss150.png",img)


img = cv2.imread(path, 0)
img = cv2.equalizeHist(img)
cv2.imwrite("../work/imgs/hist_equal.png",img)


img = cv2.imread(path, 0)
img = savgol_filter(img, 51, 1, axis=0)
img = savgol_filter(img, 51, 1, axis=1)
cv2.imwrite("../work/imgs/savgol51_1.png",img)

img = cv2.imread(path, 0)
img = savgol_filter(img, 51, 5, axis=0)
img = savgol_filter(img, 51, 5, axis=1)
cv2.imwrite("../work/imgs/savgol51_5.png",img)

img = cv2.imread(path, 0)
img = savgol_filter(img, 51, 11, axis=0)
img = savgol_filter(img, 51, 11, axis=1)
cv2.imwrite("../work/imgs/savgol51_11.png",img)

img = cv2.imread(path, 0)
img = cv2.fastNlMeansDenoising(
        img,
        dst=None,
        h=30,
        templateWindowSize=7,
        searchWindowSize=21,
    )
cv2.imwrite("../work/imgs/nlMean_30_7_21.png",img)

import _nCuts as nc
import numpy as np
import pandas as pd
path = "../work/imgs/geometric.png"
img = cv2.imread(path, 0)
df = pd.DataFrame(
    {
        "value": img.flatten(),
        "x": np.tile(range(1, img.shape[1] + 1, 1), img.shape[0]),
        "y": np.repeat(range(1, img.shape[0] + 1, 1), img.shape[1])
    }
)
def fit_export_slic(df, compactness, n_segments):
    model = nc.SLIC(compactness, n_segments)
    img = model.fit_predict(df, flatten=False)
    img = img/n_segments * 256
    cv2.imwrite(f"../work/imgs/slic_{int(compactness*100)}_{n_segments}.png",img)
    
fit_export_slic(df, 0.05, 250)
fit_export_slic(df, 0.05, 500)
fit_export_slic(df, 2, 250)
fit_export_slic(df, 2, 500)


from skimage.future import graph

df['label'] = df['x'] + df['y']-1
out = (
    df.iloc[:, 1:4]
    .pivot(index="y", columns="x", values="label")
    .to_numpy()
)
g = graph.rag_mean_color(img, out, mode='similarity', sigma=0.1)
import networkx as nx
from matplotlib import pyplot as plt
def display(g, title):
    """Displays a graph with the given title."""
    pos = nx.circular_layout(g)
    plt.figure()
    plt.title(title)
    nx.draw(g, pos)
    nx.draw_networkx_labels(g, pos)
    nx.draw_networkx_edge_labels(g, pos, font_size=20)

display(g, 'test')

out = graph.cut_normalized(out, g,thresh=.01, num_cuts=10)
cv2.imwrite("../work/imgs/ncuts.png",out)
