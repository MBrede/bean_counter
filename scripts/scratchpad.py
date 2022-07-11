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

img = cv2.imread(path, 0)
img = cv2.equalizeHist(img)
img = cv2.GaussianBlur(img, (151,151), 0)
cv2.imwrite("../work/imgs/hist_gauss.png",img)

img = cv2.imread(path, 0)
img = cv2.GaussianBlur(img, (151,151), 0)
img = cv2.equalizeHist(img)
cv2.imwrite("../work/imgs/gauss_hist.png",img)

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
def fit_export_nCuts(df, compactness=0.25, n_segments=1000, thresh=.001, num_cuts=10, sigma=100):
    model = nc.NCuts(compactness, n_segments,thresh,num_cuts,sigma)
    img = model.fit_predict(df, flatten = False)
    img = img/n_segments * 256
    cv2.imwrite(f"../work/imgs/ncuts_{int(compactness*100)}_{n_segments}_{thresh*1000}_{num_cuts}_{sigma}.png",img)
    
fit_export_nCuts(df, 0.05, 2000, num_cuts = 10, thresh = .5)
fit_export_nCuts(df, 0.05, 2000, num_cuts = 100)
fit_export_nCuts(df, 0.05, 2000, num_cuts = 10, sigma = 20)
fit_export_nCuts(df, 0.05, 2000, num_cuts = 100, sigma = 20, thresh = .5)

from skimage.future import graph

df['label'] = df['x'] + df['y']-1
out = (
    df.iloc[:, 1:4]
    .pivot(index="y", columns="x", values="label")
    .to_numpy()
)
g = graph.rag_mean_color(img, out, mode='similarity', sigma=0.1)
import networkx as nx
# from matplotlib import pyplot as plt
# import math
# from tqdm import tqdm
# import multiprocessing
import numpy as np
import pandas as pd
import cv2

path = "../work/imgs/geom_small.png"
img = cv2.imread(path, 0)

df = pd.DataFrame(
    {
        "value": img.flatten(),
        "x": np.tile(range(1, img.shape[1] + 1, 1), img.shape[0]),
        "y": np.repeat(range(1, img.shape[0] + 1, 1), img.shape[1])
    }
)
df = df.to_numpy()

def mat_dist(df, r, Sx, Si):
    X = np.array([df.T[1]])
    X = np.repeat(X, len(df), axis = 0)
    X = (X - X.T)**2
    
    Y = np.array([df.T[2]])
    Y = np.repeat(Y, len(df), axis = 0)
    Y = (Y - Y.T)**2
    
    Xij = np.sqrt(X + Y)
    del X,Y
    
    F = np.array([df.T[0]])
    F = np.repeat(F, len(df), axis = 0)
    F = np.abs(F - F.T)
    
    out = np.exp(-F**2/Si) * np.exp(-Xij**2/Sx)
    out[Xij>r] = 0
    del F
    return out


g = nx.Graph()
dists = mat_dist(df,5,4,4)
for i in range(len(df)):
    for j in range(len(df)):
        if i>j:
            g.add_edge(i, j, weight = dists[i][j])

out = graph.cut_normalized(out, g,thresh=.01, num_cuts=10)
cv2.imwrite("../work/imgs/ncuts.png",out)

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

import cv2
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
path = "../work/imgs/geometric.png"
img = cv2.imread(path, 0)
for i in range(2):
    img = cv2.pyrDown(img)
df = pd.DataFrame(
    {
        "value": np.multiply(min_max_scale(img.flatten(), 0, 100), 1),
        "x": min_max_scale(
            np.tile(range(1, img.shape[1] + 1, 1), img.shape[0]), 0, 100
        ),
        "y": min_max_scale(
            np.repeat(range(1, img.shape[0] + 1, 1),img.shape[1]), 0, 100
        ),
    }
)

for i in [1,2]:
    for j in [5,10]:
        model = DBSCAN(eps=i, min_samples=j,
                       n_jobs=-2)
        ret_df = pd.DataFrame(
            {
                "value": img.flatten(),
                "x": np.tile(range(1, img.shape[1] + 1, 1), img.shape[0]),
                "y": np.repeat(range(1, img.shape[0] + 1, 1), img.shape[1]),
                "cluster": model.fit_predict(df)
            }
        )
        img_out = (
            ret_df.iloc[:, [1, 2, 3]]
            .pivot(index="y", columns="x", values="cluster")
            .to_numpy()
        )
        cv2.imwrite(f"../work/imgs/dbscan_{i}_{j}.png", img_out)


# path = "../work/imgs/geom_small.png"
# img = cv2.imread(path, 0)

# df = pd.DataFrame(
#     {
#         "value": img.flatten(),
#         "x": np.tile(range(1, img.shape[1] + 1, 1), img.shape[0]),
#         "y": np.repeat(range(1, img.shape[0] + 1, 1), img.shape[1])
#     }
# )
# global df
# global i
# i = 0
# df = df.to_numpy()

# def calc_wij(j):
#     global pars
#     global df
#     global i
#     if i > j:
#         I = df[i]
#         J = df[j]
#         xij = np.sqrt(np.sum((I[1:] - J[1:])**2))
#         if xij < pars[2]:
#             wij = math.exp(-math.sqrt((I[0] - J[0])**2)**2/pars[0])
#             wij = wij * math.exp(-xij**2/pars[1])
#             return(i,j,wij)

# def calc_edge_weights(df, sI, sX, r):
#     mat = []
#     global pars
#     pars = (sI, sX, r)
#     pool_obj = multiprocessing.Pool()
#     global i
#     for i in tqdm(range(len(df))):
#         mat += pool_obj.map(calc_wij,range(len(df)))
#         mat = [m for m in mat if mat]
#     return(mat)

# calc_edge_weights(df,0.1, 4.0, 5)
