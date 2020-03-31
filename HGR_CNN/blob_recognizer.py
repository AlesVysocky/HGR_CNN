import cv2

def find_blob(image):
    center = None
    try:
        _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except:
        contours,_ = cv2.findContours(image.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return center
    largest_blob = max(contours, key=lambda element: cv2.contourArea(element))
    M = cv2.moments(largest_blob)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return center