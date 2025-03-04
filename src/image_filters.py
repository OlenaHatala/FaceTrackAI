import cv2
import numpy as np

def gaussian_kernel(size, sigma=1.0):
    """
    Generates a Gaussian kernel.

    Parameters:
    - size (int): Kernel size (must be odd).
    - sigma (float): Standard deviation.

    Returns:
    - numpy.ndarray: Gaussian kernel.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel) 


def gaussian_filter(img, kernel_size=5, sigma=1.0):
    """
    Applies Gaussian blur manually using a convolution operation.

    Parameters:
    - img (numpy.ndarray): Input image.
    - kernel_size (int): Size of the Gaussian kernel (must be odd).
    - sigma (float): Standard deviation.

    Returns:
    - numpy.ndarray: Blurred image.
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    return cv2.filter2D(img, -1, kernel)  

    # return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def median_filter(img, kernel_size=5):
    """
    Applies Median filter manually using a sliding window.

    Parameters:
    - img (numpy.ndarray): Input image.
    - kernel_size (int): Size of the kernel (must be odd).

    Returns:
    - numpy.ndarray: Denoised image.
    """
    # pad = kernel_size // 2
    # img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    # img_filtered = np.zeros_like(img)

    # for i in range(pad, img.shape[0] + pad):
    #     for j in range(pad, img.shape[1] + pad):
    #         for c in range(img.shape[2]):  
    #             img_filtered[i-pad, j-pad, c] = np.median(img_padded[i-pad:i+pad+1, j-pad:j+pad+1, c])
    
    # return img_filtered
    return cv2.medianBlur(img, kernel_size)


def bilateral_filter(img, diameter=9, sigma_color=75, sigma_space=75):
    """
    Applies a Bilateral filter manually using Gaussian weights in both spatial and color domains.

    Parameters:
    - img (numpy.ndarray): Input image.
    - diameter (int): Diameter of each pixel neighborhood.
    - sigma_color (float): Filter sigma in the color space.
    - sigma_space (float): Filter sigma in the coordinate space.

    Returns:
    - numpy.ndarray: Smoothed image with edge preservation.
    """
    # pad = diameter // 2
    # img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    # img_filtered = np.zeros_like(img)

    # for i in range(pad, img.shape[0] + pad):
    #     for j in range(pad, img.shape[1] + pad):
    #         for c in range(img.shape[2]):  
    #             pixel_val = img_padded[i, j, c]
    #             window = img_padded[i-pad:i+pad+1, j-pad:j+pad+1, c]

    #             ax = np.linspace(-pad, pad, diameter)
    #             xx, yy = np.meshgrid(ax, ax)
    #             spatial_weights = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_space**2))

    #             intensity_diff = window - pixel_val
    #             intensity_weights = np.exp(-(intensity_diff**2) / (2.0 * sigma_color**2))

    #             bilateral_weights = spatial_weights * intensity_weights
    #             bilateral_weights /= np.sum(bilateral_weights)  # Normalize

    #             img_filtered[i-pad, j-pad, c] = np.sum(window * bilateral_weights)
    
    # return img_filtered
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)