import torch
import numpy as np
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
import os
import cv2
import glob

# import utils as ul

# If there's a GPU Integable...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) Available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU Available, using the CPU instead.')
    device = torch.device("cpu")

import warnings
warnings.filterwarnings("ignore")


# gallery_data = [cv2.imread(file) for file in glob.glob("./ROI/session1/1/*.bmp")]

# for file in glob.glob("./ROI/session2/2/*.bmp"):
#     probe_data.extend(cv2.imread(file))
#     probe = np.array(probe_data)
# probe_data = [cv2.imread(file) for file in glob.glob("./ROI/session2/2/*.bmp")]
img_shape = (128, 128)
list = os.listdir('./new_session/new_session1')
list_ = os.listdir('./new_session2/2')
gallery_data = np.zeros((len(list)*5,img_shape[0]*img_shape[1]))
probe_data = np.zeros((len(list_)*5,img_shape[0]*img_shape[1]))
index = 0
index_=0


for file in glob.glob("./new_session/new_session1/*.bmp"):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img[img < 128] = 0
    img[img >= 128] = 1
    # print('img.shape:',img.shape)
    img = img.reshape((1, img_shape[0] * img_shape[1]))
    gallery_data[index] = img
    index = index + 1
   # gallery = np.array(gallery_data)
for file in glob.glob("./new_session2/*.bmp"):
    img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    img[img < 128] = 0
    img[img >= 128] = 1
    # print('img.shape:',img.shape)
    img = img.reshape((1, img_shape[0] * img_shape[1]))
    probe_data[index_]=img
    index_ = index_ + 1
print("gallery data's shape is", gallery_data.shape)
print("probe data's shape is", gallery_data.shape)

# the feature vector's dimension, here is 144
DIM = gallery_data.shape[1]

# subject number, here is 100
SUB_NUM = gallery_data.shape[0] // 5

'''
Calculate Distance Matrix
'''


def cal_hamming(vec1, vec2, img_shape):
    Mat1 = vec1.copy()
    Mat2 = vec2.copy()
    Mat1 = Mat1.reshape(img_shape)
    Mat2 = Mat2.reshape(img_shape)
    hamming_distance = np.logical_xor(Mat1, Mat2).sum() / (img_shape[0] * img_shape[1])
    return hamming_distance


def cal_dist(arr1, arr2):
    '''Calculate hamming distance matrix
    '''
    arr1_size = arr1.shape[0]
    arr2_size = arr2.shape[0]
    dist_mat = np.empty((arr1_size, arr2_size), dtype=np.float32)
    for i in range(arr1_size):
        for j in range(arr2_size):
            dist_mat[i, j] = cal_hamming(arr1[i], arr2[j], img_shape)
    return dist_mat


hamming_mat = cal_dist(probe_data, gallery_data)
print("distance matrix's shape =", hamming_mat.shape)

'''
There are 100 subjects, with 5 images in each subject, and for every image in probe set, 
we got 500 matching scores (Hamming distance). We then need to calculate the best 
matching score from the 5 gallery images of every subject, and get 100 scores.
'''
hamming_mat = hamming_mat.reshape(-1, SUB_NUM, 5)
print(hamming_mat.shape)
# When using Hamming distance, we should choose the minimum distance
hamming_mat = hamming_mat.min(axis=2)
print(hamming_mat.shape)

'''
Split into genuine & imposter part
We can see that the distance matrix is a 500 * 100 array.
'''


def split(mat):
    mat = mat.reshape(SUB_NUM, -1, SUB_NUM)
    mask = np.zeros_like(mat, dtype=np.int32)
    for i in range(SUB_NUM):
        mask[i, :, i] = 1
    return mat[mask == 1], mat[mask == 0]


# split matrix into genuine and imposter array
hamming_garr, hamming_iarr = split(hamming_mat)

'''
Plot Distribution
To plot distribution map, we use sns.distplot function
'''
plt.title("Hamming distance distribution")
plt.xlabel("Hamming distance")
sns.distplot(hamming_garr, kde_kws={"label": "Genuine"})
sns.distplot(hamming_iarr, kde_kws={"label": "Imposter"})
plt.grid()
plt.show()

resolu = 1000


def cal_far_frr(garr, iarr, resolu=1000):
    d_max = max(garr.max(), iarr.max())
    d_min = min(garr.min(), iarr.min())
    far = np.empty(resolu)
    frr = np.empty(resolu)
    for i, d in enumerate(np.linspace(d_min, d_max, resolu)):
        far[i] = np.sum(iarr < d) / iarr.size
        frr[i] = np.sum(garr > d) / garr.size
    return far, frr


hamming_far, hamming_frr = cal_far_frr(hamming_garr, hamming_iarr, resolu)
# To plot FAR & FRR, we use matplotlib plot function, it is widely used and can meet most of your need
# If you have any question about plt.plot function, you can check matplotlib documentation
plt.title("Hamming distance FAR & FRR")
plt.xlim([0, 1000])
plt.ylim([0, 1])
plt.plot(np.arange(resolu), hamming_far, label="FAR")
plt.plot(np.arange(resolu), hamming_frr, label="FRR")
plt.legend()
plt.grid()
plt.show()

'''
Plot ROC
'''
plt.title("ROC Curve")
plt.xlabel("False Accept Rate")
plt.ylabel("Genuine Accept Rate")
# Here we use log scale for x-axis rather than normal scale
plt.xscale('log')
plt.xlim([1e-5, 1])
plt.ylim([0.4, 1.0])
plt.plot(hamming_far, 1 - hamming_frr, label="Hamming")
plt.legend()
plt.grid()
plt.show()

'''
Calculate EER and decidability index
'''


def cal_eer(far, frr):
    return far[np.argmin(np.abs(far - frr))]


hamming_eer = cal_eer(hamming_far, hamming_frr)
print("Hamming EER = %.2f%%" % (hamming_eer * 100))


def cal_di(garr, iarr):
    u_g = garr.mean()
    u_i = iarr.mean()
    sigma_g = garr.std()
    sigma_i = iarr.std()
    return abs(u_g - u_i) / np.sqrt((sigma_g ** 2 + sigma_i ** 2) / 2)


hamming_di = cal_di(hamming_garr, hamming_iarr)
print("Hamming Decidability Index = %.2f" % (hamming_di))

'''
Calculate CMC
'''
print("Distance matrix shape:", hamming_mat.shape)
mat = hamming_mat.reshape(SUB_NUM, -1, SUB_NUM)

rank_arr = np.zeros(SUB_NUM, dtype=np.int32)
for sub_i in range(mat.shape[0]):
    for img_i in range(mat.shape[1]):
        rank_arr[np.sum(mat[sub_i, img_i] < mat[sub_i, img_i, sub_i])] += 1

hamming_cmc = np.cumsum(rank_arr) / hamming_mat.shape[0]
print(hamming_cmc)

plt.plot(range(1, 11), hamming_cmc[:10], label="Hamming")
plt.title("CMC Curve")
plt.xlabel("Rank")
plt.ylabel("Recognition Accuracy")
plt.xlim((1, 10))
plt.ylim((.9, 1))
plt.grid()
plt.legend()
plt.show()

'''
Calculate FNIR & FPIR
Here we suppose that the first 70 subjects are registered in the system and the rest subjects are not registered.
'''
REG_NUM = 70
mat = hamming_mat[:, :REG_NUM]
reg_dists = mat[:REG_NUM * 5]
unreg_dists = mat[REG_NUM * 5:]

t_min = mat.min()
t_max = mat.max()

resolu = 1000
hamming_fpir = np.empty(resolu)
hamming_fnir = np.empty(resolu)

for i, t in enumerate(np.linspace(t_min, t_max, resolu)):
    hamming_fnir[i] = np.sum(np.sum(reg_dists <= t, axis=1) == 0) / reg_dists.shape[0]
    hamming_fpir[i] = np.sum(np.sum(unreg_dists <= t, axis=1) >= 1) / unreg_dists.shape[0]

plt.title("Hamming distance FPIR & FNIR")
plt.xlabel("FPIR")
plt.ylabel("FNIR")
plt.plot(hamming_fpir, hamming_fnir, label="Hamming")
plt.grid()
plt.legend()
plt.show()
