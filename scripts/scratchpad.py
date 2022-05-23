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
