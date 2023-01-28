# import the necessary packages
import numpy as np
import argparse
import cv2
import pytesseract
import openpyxl

pytesseract.pytesseract.tesseract_cmd=r'C:\Users\Capta\Documents\Tesseract\tesseract.exe'

table = [["map", "round", "team1name", "team1ults","team2name", "team2ults", "winningTeam"]]
ultGreen = []
ultRed = []
playersGreen = []
playersRed = []
topBarText = []

def readRibScrape(filepath):
    excelFile = openpyxl.load_workbook(filepath)
    

def readImage(path): #/TODO: turn the arrays into a return statement instead of global variables
    # load and resize the image
    image = cv2.imread(path)
    image = cv2.resize(image, (1920, 1080))

    OCRImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    #OCRImage = cv2.dilate(OCRImage, kernel, iterations=1)
    #OCRImage = cv2.erode(OCRImage, kernel, iterations=1)
    # OCRImage = cv2.adaptiveThreshold(cv2.bilateralFilter(OCRImage, 9, 48, 48), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # OCRImage = cv2.threshold(OCRImage, 0, 255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)

    """
    Standard color samples:
    Green: Bmin 64/72, Bmax 88/96, Gmin 80, Gmax 112/128, Rmin 32/40, Rmax 64
    Red: Bmin 48, Bmax 64, Gmin 48/16?, Gmax 64, Rmin 80, Rmax 128

    Oddball color samples:
    Green: Bmin 32, Bmax 48, Gmin 32, Gmax 48, Rmin 16, Rmax 32
    Red: Bmin 16/24, Bmax 40/48, Gmin 16, Gmin 32, Rmin 48, Rmax 64
    """

    # define the list of boundaries
    colorBoundries = [
        ([64, 72, 40], [96, 128, 64]), #BGR value for green
        ([48, 48, 80], [80, 80, 128]), #BGR value for red
        # ([32, 16, 16], [48, 48, 32]), #BGR for green edge case, no longer needed
    ]

    """
    # these values are divided by 2 compared to the actual image size; if image size is reverted these need to be doubled
    ultXValues = [215, 745] 
    ultYValues = [400, 432, 464, 496, 528]
    """
    """
    Locations of ult images, top left corner:
    375, (742, 806, 870, 934, 998, 1062)
    X: (430, 1490), (800, 864, 928, 992, 1056)
    """
    ultXValues = [430, 1490] 
    ultYValues = [800, 864, 928, 992, 1056]

    greenValues = []
    redValues = []

    for yCoord in ultYValues:
        greenValues.append(image[yCoord, ultXValues[0]])
        redValues.append(image[yCoord, ultXValues[1]])


    #/TODO: When iterating, these will need to be duplicated outside of the loop to expose them
    ultGreen = []
    ultRed = []
    playersGreen = []
    playersRed = []
    topBarText = []

    """"
    Player name dimensions:
    Y: 765 top, 20px height, 64px in between
    X: 95 aligned left side, 1825 aligned right side. Names up to 125px long?
    """
    # playerNameYValues = [765, 829, 893, 957, 1021]
    playerNameYValues = [760, 824, 888, 952, 1016]
    playerNameXValues = [95, 1700]

    for yCoord in playerNameYValues:
        greenCrop = OCRImage[yCoord:(yCoord + 30), playerNameXValues[0]:(playerNameXValues[0] + 125)]
        greenCrop = cv2.resize(greenCrop, (250, 60))
        greenCrop = cv2.threshold(greenCrop, 160, 255, cv2.THRESH_BINARY)
        greenCrop = greenCrop[1]
        greenCrop = cv2.erode(greenCrop, kernel, iterations=2)
        greenCrop = cv2.dilate(greenCrop, kernel, iterations=1)
        name = pytesseract.image_to_string(greenCrop, config= "-c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --psm 8")
        playersGreen.append(name.rstrip('\n'))

        redCrop = OCRImage[yCoord:(yCoord + 30), playerNameXValues[1]:(playerNameXValues[1] + 125)]
        redCrop = cv2.resize(redCrop, (250, 60))
        redCrop = cv2.threshold(redCrop, 160, 255, cv2.THRESH_BINARY)
        redCrop = redCrop[1]
        # redCrop = cv2.fastNlMeansDenoising(redCrop,None,10,10,7,21)
        redCrop = cv2.erode(redCrop, kernel, iterations=1)
        #redCrop = cv2.dilate(redCrop, kernel, iterations=1)
        # cv2.imshow("image", redCrop)
        # cv2.waitKey(0)
        name = pytesseract.image_to_string(redCrop, config= "-c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --psm 8")
        playersRed.append(name.rstrip('\n'))

    for pixelGreen in greenValues:
        isGreen = True
        for i in range(len(pixelGreen)):
            isGreen = isGreen and (pixelGreen[i] > colorBoundries[0][0][i] and pixelGreen[i] < colorBoundries[0][1][i])
        if isGreen:
            ultGreen.append(True)
        else:
            ultGreen.append(False)

    for pixelRed in redValues:
        isRed = True
        for i in range(len(pixelRed)):
            isRed = isRed and (pixelRed[i] > colorBoundries[1][0][i] and pixelRed[i] < colorBoundries[1][1][i])
        if isRed:
            ultRed.append(True)
        else:
            ultRed.append(False)
    
    # topBarText key: team 1, team 1 score, team 2 score, team 2
    topBar = OCRImage[10:60, 630:710]
    topBar = cv2.erode(topBar, kernel, iterations=1)
    cv2.imshow("images", topBar)
    cv2.waitKey(0)
    team1Name = pytesseract.image_to_string(topBar, config= "-c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8")
    topBarText.append(team1Name.rstrip('\n'))

    topBar = OCRImage[5:65, 800:870]
    topBar = cv2.erode(topBar, kernel, iterations=1)
    # cv2.imshow("images", topBar)
    # cv2.waitKey(0)
    team1Name = pytesseract.image_to_string(topBar, config= "-c tessedit_char_whitelist=01234567890 --psm 8")
    topBarText.append(team1Name.rstrip('\n'))

    print(topBarText)
    print(playersGreen)
    print(playersRed)
    return []

# Row format: "map", "round", "team1name", "team1ults","team2name", "team2ults", "winningTeam"
def matchNames():
    pass

"""
# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	# find the colors within the specified boundaries and apply the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)
"""

"""
1. Import image
2. Cut out the Green ult column, Red Ult Column, and numbers
3. Use Tesseract to record the round #
4. For each ult column, check 5 places (relative to column coordinates) if the color is correct
"""

row = [] #reset after each iteration

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
readImage(args["image"])

# convert into the correct format after getting rib data
table.append()
