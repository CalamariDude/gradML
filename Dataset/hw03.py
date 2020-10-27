# -*- coding: utf-8 -*-
"""
File:   hw03C.py
"""

"""
====================================================
================ Import Packages ===================
====================================================
"""
import sys

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import skimage.filters as filt

"""
====================================================
================ Define Functions ==================
====================================================
"""

def process_image(in_fname,out_fname,debug=False):

    # load image
    x_in = np.array(Image.open(in_fname))

    # convert to grayscale
    x_gray = 1.0-rgb2gray(x_in)

    if debug:
        plt.figure(1)
        plt.imshow(x_gray)
        plt.title('original grayscale image')
        plt.show()

    # threshold to convert to binary
    thresh = filt.threshold_minimum(x_gray)
    fg = x_gray > thresh

    if debug:
        plt.figure(2)
        plt.imshow(fg)
        plt.title('binarized image')
        plt.show()

    # find bounds
    nz_r,nz_c = fg.nonzero()
    n_r,n_c = fg.shape
    l,r = max(0,min(nz_c)-1),min(n_c-1,max(nz_c)+1)+1
    t,b = max(0,min(nz_r)-1),min(n_r-1,max(nz_r)+1)+1

    # extract window
    win = fg[t:b,l:r]

    if debug:
        plt.figure(3)
        plt.imshow(win)
        plt.title('windowed image')
        plt.show()

    # resize so largest dim is 48 pixels 
    max_dim = max(win.shape)
    new_r = int(round(win.shape[0]/max_dim*48))
    new_c = int(round(win.shape[1]/max_dim*48))

    win_img = Image.fromarray(win.astype(np.uint8)*255)
    resize_img = win_img.resize((new_c,new_r))
    resize_win = np.array(resize_img).astype(bool)

    # embed into output array with 1 pixel border
    out_win = np.zeros((resize_win.shape[0]+2,resize_win.shape[1]+2),dtype=bool)
    out_win[1:-1,1:-1] = resize_win

    if debug:
        plt.figure(4)
        plt.imshow(out_win,cmap='Greys')
        plt.title('resized windowed image')
        plt.show()

    #save out result as numpy array
    np.save(out_fname,out_win)

"""
====================================================
========= Generate Features and Labels =============
====================================================
"""

# if __name__ == '__main__':

#     # To not call from command line, comment the following code block and use example below 
#     # to use command line, call: python hw03.py K.jpg output

#     if len(sys.argv) != 3 and len(sys.argv) != 4:
#         print('usage: {} <in_filename> <out_filename> (--debug)'.format(sys.argv[0]))
#         sys.exit(0)
    
#     in_fname = sys.argv[1]
#     out_fname = sys.argv[2]

#     if len(sys.argv) == 4:
#         debug = sys.argv[3] == '--debug'
#     else:
#         debug = False


#    #e.g. use'

import os

alpha = ['a','b','c','d','h','i','j','k']
for letter in alpha:
    path = './'
    path = path + letter + '/'
    items = os.listdir(path)
    i = 0
    for item in items:
        if item.find('.JPG') == -1:
            continue
        out_name = letter + str(i) + '.npy'
        process_image(path + item,out_name,debug=False)
        i+=1
# process_image('./a.JPG','./output.npy',debug=False)
    # process_image(in_fname,out_fname)
