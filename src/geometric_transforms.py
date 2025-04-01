import cv2
import numpy as np

def scale_image(image, fx=1.0, fy=1.0):
    return cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

def rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h))

def perspective_transform(image, src_pts, dst_pts):
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    (h, w) = image.shape[:2]
    return cv2.warpPerspective(image, M, (w, h))