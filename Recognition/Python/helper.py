import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb


def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')

#best one (handles worst cases (img 6,7,8))
# Gamma Correction

def Gamma_Correction(img,c,gamma):
    
    g_img=np.array(c*(img / 255) ** gamma)
    return g_img

#remove impulsive/random noise using median filter
def median_filter_algorithm(img, filter_size=(3,3)):
    if((filter_size[0] != filter_size[1]) or (filter_size[0]%2 == 0)):
        print('Please enter an odd square shape')
        return
    
    copied_img = img.copy()
    img_size = img.shape
    half_filter_size = filter_size[0]//2
    
    for row in range(img_size[0]):
        row_start = max(0, row - half_filter_size)
        row_end = min(img_size[0], row + half_filter_size)
        
        for col in range(img_size[1]):
            col_start = max(0, col - half_filter_size)
            col_end = min(img_size[1], col + half_filter_size)
            
            pixels_list = list(img[row_start:row_end+1, col_start:col_end+1].flatten())
            padded_pixels_list = pixels_list + [0]*(len(pixels_list) - filter_size[0]**2)
            padded_pixels_list.sort()
            copied_img[row, col] = padded_pixels_list[len(pixels_list)//2]
    
    return copied_img

# #----------------------------------------------------------------------------------

# Equalization
def histogramEqualization(img,nbins=256):
#     img_eq = equalize_hist(img)
#     return img_eq
    img=img*255
    img=img.astype('uint8')
    values, keys= histogram(img)
    H_arr= np.zeros(nbins)
    for i in range (0, len(keys)):
        H_arr[keys[i]] = values[i]
    H_c= np.zeros(nbins)
    T_arr= np.zeros(nbins)

    for i in range (0, nbins):
        if(i==0):
            H_c[i]= H_arr[0]
        else:
            H_c[i]= H_c[i-1] + H_arr[i]
        T_arr[i]= round((nbins-1)*H_c[i]/float(img.shape[0]*img.shape[1]))
    filtered_img= np.copy(img)
    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            filtered_img[row,col]= T_arr[filtered_img[row,col]]
    return filtered_img
  

