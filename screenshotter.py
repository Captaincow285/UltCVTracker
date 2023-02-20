import numpy as np
import argparse
import cv2


image = cv2.imread("liveImage.jpg")
image = cv2.resize(image, (1920, 1080))

# Green: X = 110; Y = 650-725 (675, 700); Colors: R 40-56, G 152-176, B 112-120
# Red: X = 1810: Y = 650-725 (675, 700); Colors: 

