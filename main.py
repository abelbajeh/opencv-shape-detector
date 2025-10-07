import cv2
import numpy as np


def detect_contours(image, paper, original):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area > 10:
            cv2.drawContours(paper, [cnt], -1, (0, 255, 0), 3)
            approx = cv2.approxPolyDP(cnt, 0.05*peri,True)
            print(approx)
            cv2.drawContours(paper, approx, -1,(0,0,255), 3)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(original,(x,y), (x+w, y+h), (0, 255, 0), 2)
            shape = None
            if len(approx) == 4: shape = "rect"
            if len(approx) == 3: shape = "tri"
            if len(approx) == 6: shape = "hex"
            if len(approx) == 4: shape = "rect"


img = cv2.imread("img.png")
h, w, _ = img.shape
sheet = np.zeros((w, h, 3), np.uint8)
img_blur = cv2.GaussianBlur(img, (5, 5), 0)
img_canny = cv2.Canny(img_blur, 50, 150)

detect_contours(img_canny, sheet, img)


cv2.imshow("original", img)
cv2.imshow("sheet", sheet)
cv2.imshow("blur", img_blur)
cv2.imshow("canny", img_canny)
cv2.waitKey(0)
