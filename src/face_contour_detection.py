import cv2
import matplotlib.pyplot as plt


def detect_face_contours(image):
    '''
    Detects faces in an image and extracts their contours.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
    
    Returns:
        tuple: A tuple containing detected faces and a list of contours.
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    contours_list = []
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        edges = cv2.Canny(roi, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.append((x, y, w, h, contours))
    
    return faces, contours_list


def show_detected_faces(image, contours_list=None):
    '''
    Draws rectangles around detected faces and overlays their contours if available.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        contours_list (list, optional): List of detected face contours.
    '''
    img_copy = image.copy()
    
    for (x, y, w, h, contours) in contours_list:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if contours_list:
          for contour in contours:
              contour[:, 0, 0] += x  # Зміщуємо контури відповідно до області обличчя
              contour[:, 0, 1] += y
              cv2.drawContours(img_copy, [contour], -1, (0, 0, 255), 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()