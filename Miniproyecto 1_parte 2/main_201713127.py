import os
import glob
import requests
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy.io import loadmat
import nibabel as nb

dt = loadmat('groundtruth.mat')
print(dt.shape)
dfgdgfhgh