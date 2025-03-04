import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_image_info(img, path):
    """
    Extracts metadata about an image, including file name, format, 
    size, number of channels, file extension, and additional details like 
    color mode.
    """
    pil_image = Image.open(path)  
    channels = img.shape[2] if len(img.shape) == 3 else 1
    color_mode = pil_image.mode  
    file_size = os.path.getsize(path) 

    info = {
        "File": os.path.basename(path),
        "Format": pil_image.format,
        "Size (px)": pil_image.size,
        "Channels": channels,
        "Color Mode": color_mode,
        "File Size (bytes)": file_size,
        "Extension": os.path.splitext(path)[-1].lower()
    }

    return info

def plot_histogram(img):
    """    
    Generates and displays the brightness histogram of a grayscale image.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.hist(img_gray.ravel(), bins=256, range=[0, 256], color='black', alpha=0.75)
    plt.title("Brightness histogram")
    plt.xlabel("Brightness")
    plt.ylabel("Number of pixels")
    plt.show()

def enhance_contrast_hist_equalization(img):
    """
    Enhances image contrast using histogram equalization.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(img_gray)
    return equalized

def enhance_contrast_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhances contrast using Adaptive Histogram Equalization (CLAHE).
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(img_gray)
    return clahe_img

def enhance_contrast_gamma(img, gamma=1.5):
    """
    Enhances image contrast using Gamma Correction.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    gamma_corrected = cv2.LUT(img, table)
    return gamma_corrected
