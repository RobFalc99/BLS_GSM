# -*- coding: utf-8 -*-
"""
==========================================
=== Corso di Elaborazione  di Immagini ===
===          Final Project             ===
===         Image Denoising            ===
===             BLS-GSM                ===
===         Roberto Falcone            ===
==========================================
"""

import numpy as np
from numpy.linalg import inv, det
import cv2
import skimage as sk
import pywt
import math
from math import pi, e
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

def evaluate_metrics(org_image, rec_image, print_metrics=False):
    """
    Evaluate mean squared error, peak signal noise ratio and structural
    similarity between two images
     
    :param ndarray org_image: The original image
    :param ndarray rec_image: The reconstructed image
    :return: In order: 
        • Mean squared error
        • Peak signal noise ratio
        • Structural similarity
    """
    
    assert org_image.shape == rec_image.shape
    
    mse = mean_squared_error(org_image, rec_image)
    psnr = peak_signal_noise_ratio(org_image, rec_image)
    if len(org_image.shape)>2:
        ss = structural_similarity(org_image, rec_image, channel_axis=2)
    else:
        ss = structural_similarity(org_image, rec_image)
        
    if print_metrics:
        print('MSE: ', mse)
        print('PSNR: ', psnr)
        print('SS: ', ss)
    
    return mse, psnr, ss
                   
#%% Parameters
tras = "sym3"
wavelet = pywt.Wavelet(tras)

window_size = (3,3)
N = window_size[0] * window_size[1]

z_min = 1.25*(10**(-9))
z_max = 33.1155

z_step = 2

z_lin = np.arange(z_min, z_max+z_step, z_step)
    
p_z = {}
for z in z_lin:
    p_z[z] = 1/z    

mean_g = 0

def img_as_ubyte(f):
    return sk.util.img_as_ubyte(f)

def img_as_float(f):
    return f/255

def pad(f, new_image_width, new_image_height):
    """
    Adds padding to an image up to a desired width and height
     
    :param ndarray(N,M) f: The image on which apply the padding
    :param int new_image_width: The new width of image
    :param int new_image_height: The new height of image
    :return: The padded image
    """
    if len(f.shape)>2:
        old_image_height, old_image_width, channels = f.shape
        # create new image of desired size and color (black) for padding
        result = np.full((new_image_height,new_image_width, channels), (0,0,0), dtype=np.float64)
        
    else:
        old_image_height, old_image_width = f.shape
        # create new image of desired size and color (black) for padding
        result = np.full((new_image_height,new_image_width), 1, dtype=np.float64)
        
    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2
    
    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
           x_center:x_center+old_image_width] = f
        
    return result

def auto_pad(f, lv):
    """
    Automatically pads an image depending on how many wavelet levels are needed
     
    :param ndarray(N,M) f: The image on which apply the padding
    :param int lv: Number of levels in which decompose the image
    :return: The padded image
    """
    old_image_height = f.shape[0]
    old_image_width = f.shape[1]
    
    if old_image_height%(2**lv)==0:
        new_image_height = old_image_height
    else:
        new_image_height = ((old_image_height//(2**lv))+1)*(2**lv)
        
    if old_image_width%(2**lv)==0:
        new_image_width = old_image_width
    else:
        new_image_width = ((old_image_width//(2**lv))+1)*(2**lv)
        
    return pad(f, new_image_width, new_image_height)

def reverse_pad(f_org, f):
    """
    Removes padding from an image
     
    :param ndarray(N,M) f_org: The original image
    :param ndarray(N,M) f: The image from which remove the padding
    :return: The image without padding
    """
    Nx_org = f_org.shape[0]
    Ny_org = f_org.shape[1]
    
    Nx = f.shape[0]
    Ny = f.shape[1]
    
    if Nx!=Nx_org:
        cut_x = Nx - Nx_org
        
        cut_x_1 = cut_x//2
        cut_x_2 = cut_x//2 + cut_x%2
        
        f = f [cut_x_1:Nx-cut_x_2,:]
        
    if Ny!=Ny_org:
        cut_y = Ny - Ny_org
        
        cut_y_1 = cut_y//2
        cut_y_2 = cut_y//2 + cut_y%2
        
        f = f [:,cut_y_1:Ny-cut_y_2]
        
    return f

#%% Algorithm implementation

def sub_band_denoise(image_sub_band_x, noise_sub_band_x, window_size):
    #Covariance matrixes building
    Y_noise = []
    Y_image = []    
        
    for y in range(0, image_sub_band_x.shape[0], window_size[1]):
      for x in range(0, image_sub_band_x.shape[1], window_size[0]):
          window_image = image_sub_band_x[y:y + window_size[1], x:x + window_size[0]]
          window_noise = noise_sub_band_x[y:y + window_size[1], x:x + window_size[0]]
          if window_noise.shape == window_size and window_image.shape == window_size:
              Y_noise.append(window_noise.flatten())
              Y_image.append(window_image.flatten())

  
    Y_noise = np.array(Y_noise).T
    Y_image = np.array(Y_image).T
    
    C_w = np.cov(Y_noise)
    C_y = np.cov(Y_image)
    
    #C_u estimating
    C_u = C_y - C_w
    
    #Building terms independentely from sub-bands neighboors
    z_terms = {}
    z_terms_inv = {}
    z_terms_det = {}
    den_z = {}
    for z in z_lin:
        z_terms[z] = (z*C_u)+C_w
        z_terms_inv[z] = inv(z_terms[z])
        z_terms_det[z] = (det(z_terms[z]))
        den_z[z] = math.sqrt( ((2*pi)**N) * (z_terms_det[z]))
    
    #For each neighboor
    new_band = np.zeros((image_sub_band_x.shape)) + image_sub_band_x
    for y in range(0, image_sub_band_x.shape[0]-window_size[1], 1):
      for x in range(0, image_sub_band_x.shape[1]-window_size[0], 1):
          neighborhood = image_sub_band_x[y:y + window_size[1], x:x + window_size[0]]
          if neighborhood.shape == window_size:
              E = {}
              p_y = {}
              
              #Row neighboor
              flatten = neighborhood.flatten()
              #Column neighboor
              flatten_T = np.atleast_2d(neighborhood.flatten())
              
              #Compute:
              # • E[x|y,z]  ->  E[z] 
              # • p(y|z)    ->  p_y[z]
              
              for z in z_lin:                      
                  E[z] = ((z * C_u) @ z_terms_inv[z]) @ flatten
                  num = math.exp(- (flatten_T @ z_terms_inv[z] @ flatten)/2)
                  p_y[z] = num/den_z[z]
              
              #Compute p(z|y)' denominator
              p_zy = {}
              count = 0
              for z in z_lin:
                  p_zy[z] = (p_y[z]*p_z[z])
                  count += p_y[z]*p_z[z]
                   
              #Compute p(z|y) -> p_zy[z]
              for z in z_lin:
                  p_zy[z] = p_zy[z]/count
              
              #Finally compute E[x_c|y] -> x_c
              x_c=0
              for z in z_lin:
                  x_c+=E[z][4] * p_zy[z]
              
              #Assign x_c to the current sub-band' neighboor center
              new_band[y+window_size[1]//2][x+window_size[0]//2] = x_c
          
    return new_band
    

def BLS_GSM_gray(f, lv):
    """
    Implements the BLS_GSM denoising algorithm on a single channel image.
     
    :param ndarray(N,M) f: The image on which apply the algorithm
    :param int lv: The number of levels on which decompose the input image
    :return: The denoised single channel image
    """
    
    #Checks if the image is single channel
    assert not len(f.shape)>2
    
    f = img_as_float(f)    
    
    #Estimating sigma_g and building noise image
    sigma_g = round(sk.restoration.estimate_sigma(f), 3)
    
    Noise_G = img_as_ubyte(np.clip(mean_g + sigma_g*np.random.randn(*f.shape), -1, 1))

    #Dividing the image and the noise profile into sub-bands
    coeffs = pywt.swt2(f, tras, level=lv, norm=True)
    noise_coeffs = pywt.swt2(img_as_float(Noise_G), tras, level=lv, norm=True)
    
    new_coeffs = coeffs.copy()
    new_bands = []
    
    #For each sub-band
    for ii in range(0,len(new_coeffs),1):
        print('lvl: ' + str(ii+1))
        image_sub_bands = new_coeffs[ii][1]
        noise_sub_bands = noise_coeffs[ii][1]
        for jj in range(0, len(noise_sub_bands), 1):
            #Progress print
            print('sub-band: ' + str(jj+1))
            
            image_sub_band_x = image_sub_bands[jj]
            noise_sub_band_x = noise_sub_bands[jj]
            
            new_bands.append(sub_band_denoise(image_sub_band_x, noise_sub_band_x, window_size))


    #Reconstruct the new coefficents from the created new_bands
    rec_coeffs = []
    count = 0
    for level in range(0,lv,1):
        rec_coeffs.append((coeffs[level][0], (new_bands[count],new_bands[count+1],new_bands[count+2])))
        count+=3
    
    #Build the denoised image applying the inverse swt2 on the new coefficents
    return img_as_ubyte(np.clip(pywt.iswt2(rec_coeffs, tras, norm=True), -1, 1))



def BLS_GSM_rgb(f, lv):
    """
    Implements the BLS_GSM denoising algorithm on a multi-channel image.
     
    :param ndarray(N,M,C) f: The image on which apply the algorithm
    :param int lv: The number of levels on which decompose the input image
    :return: The denoised multi-channel image
    """
    assert len(f.shape)>2
    
    f = img_as_float(f)
    
    #Estimating sigma_g and building noise image
    sigma_g = [round(x,3) for x in sk.restoration.estimate_sigma(f, channel_axis=2)]
    print(sigma_g)
    
    Noise_G = img_as_ubyte(np.clip(mean_g + sigma_g*np.random.randn(*f.shape), -1, 1))
    
    f_s = [f[:,:,i] for i in range(0,3,1)]
    Noise_G_s = [(img_as_float(Noise_G[:,:,i])) for i in range(0,3,1)]
    
    channels = []
    
    #For each channel
    for i in range(0,3,1):
        print('c: ', i+1)
        f = f_s[i]
        Noise_G = Noise_G_s[i]
        
        #Dividing the image and the noise profile into sub-bands
        coeffs = pywt.swt2(f, tras, level=lv)
        noise_coeffs = pywt.swt2(Noise_G, tras, level=lv)
        
        new_coeffs = coeffs.copy()
        new_bands = []
        
        #For each sub-band
        for ii in range(0,len(new_coeffs),1):
            print('lvl: ' + str(ii+1))
            image_sub_bands = new_coeffs[ii][1]
            noise_sub_bands = noise_coeffs[ii][1]
            for jj in range(0, len(noise_sub_bands), 1):
                #Progress print
                print('sub-band: ' + str(jj+1))
                
                image_sub_band_x = image_sub_bands[jj]
                noise_sub_band_x = noise_sub_bands[jj]
                
                new_bands.append(sub_band_denoise(image_sub_band_x, noise_sub_band_x, window_size))
                
        #Reconstruct the new coefficents from the created new_bands
        rec_coeffs = []
        count = 0
        for level in range(0,lv,1):
            rec_coeffs.append((coeffs[level][0], (new_bands[count],new_bands[count+1],new_bands[count+2])))
            count+=3
        
        #Build the denoised image applying the inverse swt on the new coefficents
        channels.append(img_as_ubyte(np.clip(pywt.iswt2(rec_coeffs, tras), -1, 1)))
        
    return np.dstack((channels[0],channels[1],channels[2]))


def BLS_GSM(f):
    """
    Applyes the BLS-GSM algorithm
     
    :param ndarray(N,M) f: The image on which apply the algorithm
    :return: The denoised image
    """
    
    f_org = f.copy()
    
    #Compute on how many levels decompose the image
    #based on image sizes
    Nx = f.shape[0]
    Ny = f.shape[1]
    
    lv = max(int(math.log2(min(Nx,Ny))-4)//2, 1)
        
    #Pad the image if needed
    if Nx%(2**lv)!=0 or Ny%(2**lv)!=0:
        f = auto_pad(f, lv)
    
    #Apply BLS-GSM algorithm accorting to image' number of channels
    if len(f.shape)>2:
        res = BLS_GSM_rgb(f, lv)
    else:
        res = BLS_GSM_gray(f, lv)

    #Remove eventually added padding and return the denoised image
    return reverse_pad(f_org, res)
    