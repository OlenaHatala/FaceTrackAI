import cv2
import matplotlib.pyplot as plt

def extract_features(image, method="SIFT"):
    '''
    Extracts keypoints and descriptors from an image using specified feature extraction method.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        method (str): Feature extraction method (SIFT, SURF, ORB, HOG).
    
    Returns:
        tuple: A tuple containing keypoints and descriptors.
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "SIFT":
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
    elif method == "SURF":
            try:
                surf = cv2.xfeatures2d.SURF_create()
                keypoints, descriptors = surf.detectAndCompute(gray, None)
            except AttributeError:
                print("SURF is not available in your OpenCV build. Using SIFT instead.")
                sift = cv2.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(gray, None)
    elif method == "ORB":
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
    elif method == "HOG":
        hog = cv2.HOGDescriptor()
        descriptors = hog.compute(gray)
        keypoints = []  # HOG не повертає ключових точок
    else:
        raise ValueError("Unsupported feature extraction method")
    
    return keypoints, descriptors


def visualize_features(image, keypoints, method):
    '''
    Visualizes detected keypoints or descriptors for a given feature extraction method.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        keypoints (list or None): List of detected keypoints, if applicable.
        descriptors (numpy.ndarray or None): Computed feature descriptors.
        method (str): Feature extraction method.
    '''
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(f"Feature Detection using {method}")
    plt.axis("off")
    plt.show()