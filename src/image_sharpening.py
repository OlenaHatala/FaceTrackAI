import cv2

def laplacian_sharpening(img):
    """
    Sharpens the image using the Laplacian operator.

    Parameters:
    - img (numpy.ndarray): Input image.

    Returns:
    - numpy.ndarray: Sharpened image.
    """
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(img - laplacian)
    return sharpened

def unsharp_masking(img, sigma=1.0, strength=1.5):
    """
    Sharpens the image using the Unsharp Masking technique.

    Parameters:
    - img (numpy.ndarray): Input image.
    - sigma (float): Gaussian blur standard deviation.
    - strength (float): Sharpening strength.

    Returns:
    - numpy.ndarray: Sharpened image.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    return sharpened
