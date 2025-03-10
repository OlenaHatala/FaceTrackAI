import cv2
import numpy as np
import heapq

def threshold(img, thr_type=cv2.THRESH_BINARY):
    """
    Applies thresholding to an input grayscale image.

    Parameters:
    - img (numpy.ndarray): Input grayscale image.
    - thr_type (int, optional): Thresholding type (default is cv2.THRESH_BINARY).

    Returns:
    - numpy.ndarray: Thresholded binary image.
    """
    _, thr_img = cv2.threshold(img, 120, 255, thr_type)
    return thr_img

def otsu_threshold(img):
    """
    Applies Otsu's thresholding method for object separation.

    Parameters:
    - img (numpy.ndarray): Input grayscale image.

    Returns:
    - numpy.ndarray: Binary thresholded image.
    """
    _, otsu_thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresh

def watershed_segmentation(gray_image):
    """
    Implements the Watershed algorithm for image segmentation without using cv2.watershed().

    Parameters:
    - gray_image (numpy.ndarray): Input grayscale image.

    Returns:
    - numpy.ndarray: Marker image after applying the Watershed algorithm.
    """

    # 1. Попередня обробка: розмиття та порогова сегментація
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Видалення шуму морфологічними операціями
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Визначення фонового та переднього планів
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 4. Визначення невизначеної області
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5. Створення початкових маркерів
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Зсув маркерів, щоб фон був 1, а об'єкти - від 2
    markers[unknown == 255] = 0  # Позначаємо невідомі області як 0

    # 6. Реалізація алгоритму Watershed вручну
    h, w = gray_image.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Чотири напрямки: вгору, вниз, вліво, вправо
    priority_queue = []  # Черга з пріоритетами для затоплення областей

    # Додаємо всі початкові маркери до черги з їхніми градаціями
    for y in range(h):
        for x in range(w):
            if markers[y, x] > 1:  # Пропускаємо фон (1) і невідомі області (0)
                heapq.heappush(priority_queue, (gray_image[y, x], y, x))

    # 7. Процес розширення областей
    while priority_queue:
        _, y, x = heapq.heappop(priority_queue)

        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and markers[ny, nx] == 0:
                markers[ny, nx] = markers[y, x]  # Наслідуємо маркер сусіда
                heapq.heappush(priority_queue, (gray_image[ny, nx], ny, nx))

    # 8. Позначення контурів вододілів
    result = np.zeros_like(gray_image, dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if any(0 <= y + dy < h and 0 <= x + dx < w and markers[y, x] != markers[y + dy, x + dx]
                   for dy, dx in directions):
                result[y, x] = 255  # Контур вододілу

    return result


def grabcut(img, rect, iterations=5):
    """
    Implements the GrabCut algorithm for image segmentation without using cv2.grabCut() or maxflow.

    Parameters:
    - img (numpy.ndarray): Input BGR image.
    - rect (tuple): Bounding box for the foreground object (x, y, width, height).
    - iterations (int): Number of refinement iterations.

    Returns:
    - numpy.ndarray: Binary mask where the foreground is white (255) and the background is black (0).
    """
    h, w = img.shape[:2]

    # Step 1: Initialize the mask (0 = background, 1 = foreground)
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, w, h = rect
    mask[y:y+h, x:x+w] = 1  # Mark initial foreground region

    # Step 2: Convert image to LAB color space for better segmentation
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_flat = img_lab.reshape((-1, 3)).astype(np.float32)

    for _ in range(iterations):
        # Step 3: Extract foreground and background pixels
        fg_pixels = img_flat[mask.flatten() == 1]
        bg_pixels = img_flat[mask.flatten() == 0]

        # Step 4: Compute mean color for foreground & background
        fg_mean = np.mean(fg_pixels, axis=0)
        bg_mean = np.mean(bg_pixels, axis=0)

        # Step 5: Compute distance of each pixel to the foreground & background means
        fg_dist = np.linalg.norm(img_flat - fg_mean, axis=1)
        bg_dist = np.linalg.norm(img_flat - bg_mean, axis=1)

        # Step 6: Assign each pixel to the closer cluster
        new_mask = np.where(fg_dist < bg_dist, 1, 0).astype(np.uint8)

        # Reshape the mask back to the image shape
        mask = new_mask.reshape((h, w))

    # Step 7: Convert the final mask to binary (0 = background, 255 = foreground)
    binary_mask = (mask * 255).astype(np.uint8)
    
    return binary_mask


def get_object_bounding_box(img):
    """
    Automatically finds the bounding box of the largest object in an image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return (x, y, w, h)
    else:
        return (0, 0, img.shape[1], img.shape[0])  # Якщо контурів нема, повертаємо все зображення


