import numpy as np
import pyautogui
import PIL
from PIL import ImageGrab
import argparse
import cv2
import time

"""
pil_image = PIL.ImageGrab.grab(bbox=None, include_layered_windows=False, all_screens=True)
open_cv_image = np.array(pil_image) 
# Convert RGB to BGR 
open_cv_image = open_cv_image[:, :, ::-1].copy()
cv2.imshow("Image", open_cv_image)
cv2.waitKey(0)
"""

# image = cv2.imread("C9-DRX-M1-R2.jpg")
# image = cv2.resize(image, (1920, 1080))

# Green: X = 110; Y = 650-725 (675, 700); Colors: R 40-56, G 152-176, B 112-120
# Red: X = 1810: Y = 650-725 (675, 700); Colors: R 208-240, G 64-80, B 88-104

def checkIsScoreboard(image):
    leftCoordinate = image[700, 110]
    # rightCoordinate = image[700, 1810]

    isGreen = True
    vibrantGreenBoundries = [[112, 144, 48], [128, 176, 64]]
    for i in range(len(leftCoordinate)):
        isGreen = isGreen and (leftCoordinate[i] > vibrantGreenBoundries[0][i] and leftCoordinate[i] < vibrantGreenBoundries[1][i])

    isRed = True
    vibrantRedBoundries = [[88, 64, 208], [104, 80, 240]]
    for i in range(len(leftCoordinate)):
        isRed = isRed and (leftCoordinate[i] > vibrantRedBoundries[0][i] and leftCoordinate[i] < vibrantRedBoundries[1][i])
    
    print(isGreen)
    print(isRed)

    if not isGreen and not isRed:
        return False
    return True

counter = 0
while(True):
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    if checkIsScoreboard(image):
        print("Found Scoreboard! Writing.....")
        filename = "screenshotTool/generated/image" + str(counter) + ".jpg"
        print(filename)
        cv2.imwrite(filename, image)
        counter += 1
        time.sleep(15)
    else:
        print("No Scoreboard Found")
        time.sleep(1)