# import the necessary packages
import numpy as np
import argparse
import cv2
import pytesseract
import openpyxl

pytesseract.pytesseract.tesseract_cmd=r'C:\Users\Capta\Documents\Tesseract\tesseract.exe'

table = [["map", "round", "team1name", "team1eco", "team1ults","team2name", "team2eco", "team2ults", "winningTeam"]]
ultLeft = []
ultRight = []
playersLeft = []
playersRight = []
topBarText = []
roundsWorkheet = None

def readRibScrape(filepath):
    excelFile = openpyxl.load_workbook(filepath)
    roundsWorksheet = excelFile('WIN CONDITIONS')
    infoWorksheet = excelFile('PLAYERS INFO')

    matchinfo = []
    selectedData  = []
    for row in infoWorksheet:
        selectedData.append([row[2].value, row[3].value])
        selectedData.append([row[6].value, row[7].value])
        selectedData.append([row[10].value, row[11].value])
        selectedData.append([row[14].value, row[15].value])
        selectedData.append([row[16].value, row[17].value])
        selectedData.append([row[20].value, row[21].value])
        selectedData.append([row[24].value, row[25].value])
        selectedData.append([row[28].value, row[29].value])
        selectedData.append([row[32].value, row[33].value])
        selectedData.append([row[36].value, row[37].value])
        matchinfo.append(selectedData)


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

    leftValues = []
    rightValues = []

    for yCoord in ultYValues:
        leftValues.append(image[yCoord, ultXValues[0]])
        rightValues.append(image[yCoord, ultXValues[1]])


    #/TODO: When iterating, these will need to be duplicated outside of the loop to expose them
    ultLeft = []
    ultRight = []
    playersLeft = []
    playersRight = []
    topBarText = []
    loadoutValues = [] #Green value, Red value

    """"
    Player name dimensions:
    Y: 765 top, 20px height, 64px in between
    X: 95 aligned left side, 1825 aligned right side. Names up to 125px long?
    """
    # playerNameYValues = [765, 829, 893, 957, 1021]
    playerNameYValues = [760, 824, 888, 952, 1016]
    playerNameXValues = [95, 1700]

    for yCoord in playerNameYValues:
        leftCrop = OCRImage[yCoord:(yCoord + 30), playerNameXValues[0]:(playerNameXValues[0] + 125)]
        leftCrop = cv2.resize(leftCrop, (250, 60))
        leftCrop = cv2.threshold(leftCrop, 160, 255, cv2.THRESH_BINARY)
        leftCrop = leftCrop[1]
        leftCrop = cv2.erode(leftCrop, kernel, iterations=2)
        leftCrop = cv2.dilate(leftCrop, kernel, iterations=1)
        name = pytesseract.image_to_string(leftCrop, config= "-c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --psm 8")
        playersLeft.append(name.rstrip('\n'))

        rightCrop = OCRImage[yCoord:(yCoord + 30), playerNameXValues[1]:(playerNameXValues[1] + 125)]
        rightCrop = cv2.resize(rightCrop, (250, 60))
        rightCrop = cv2.threshold(rightCrop, 160, 255, cv2.THRESH_BINARY)
        rightCrop = rightCrop[1]
        rightCrop = cv2.erode(rightCrop, kernel, iterations=1)
        rightCrop = cv2.dilate(rightCrop, kernel, iterations=1)
        name = pytesseract.image_to_string(rightCrop, config= "-c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --psm 8")
        playersRight.append(name.rstrip('\n'))

    isGreen = True
    checkedPixel = image[700, 110]
    vibrantGreenBoundries = [[112, 144, 48], [128, 176, 64]]
    for i in range(len(checkedPixel)):
        isGreen = isGreen and (checkedPixel[i] > vibrantGreenBoundries[0][i] and checkedPixel[i] < vibrantGreenBoundries[1][i])
    #print(checkedPixel)
    #print(isGreen)

    for leftPixel in leftValues:
        selectedColor = None
        if isGreen:
            selectedColor = colorBoundries[0]
        else:
            selectedColor = colorBoundries[1]
        isCorrectColor = True
        for i in range(len(leftPixel)):
            isCorrectColor = isCorrectColor and (leftPixel[i] > selectedColor[0][i] and leftPixel[i] < selectedColor[1][i])
        if isCorrectColor:
            ultLeft.append(True)
        else:
            ultLeft.append(False)

    for rightPixel in rightValues:
        selectedColor = None
        if isGreen:
            selectedColor = colorBoundries[1]
        else:
            selectedColor = colorBoundries[0]
        isCorrectColor = True
        for i in range(len(rightPixel)):
            isCorrectColor = isCorrectColor and (rightPixel[i] > colorBoundries[1][0][i] and rightPixel[i] < colorBoundries[1][1][i])
        if isCorrectColor:
            ultRight.append(True)
        else:
            ultRight.append(False)
    
    # Loadout values: Y 705-740, X green 155-245, X red 1705-1800
    loadoutValueImage = OCRImage[705:740, 155:245]
    value = pytesseract.image_to_string(loadoutValueImage, config= "-c tessedit_char_whitelist=01234567890, --psm 8")
    loadoutValues.append(int(value.replace(",", "")))
    
    loadoutValueImage = OCRImage[705:740, 1705:1800]
    value = pytesseract.image_to_string(loadoutValueImage, config= "-c tessedit_char_whitelist=01234567890, --psm 8")
    loadoutValues.append(int(value.replace(',', '')))
    
    # topBarText key: team 1, team 1 score, team 2 score, team 2
    topBar = OCRImage[10:60, 630:710]
    topBar = cv2.erode(topBar, kernel, iterations=1)
    team1Name = pytesseract.image_to_string(topBar, config= "-c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8")
    topBarText.append(team1Name.rstrip('\n'))

    topBar = OCRImage[5:65, 800:870]
    topBar = cv2.erode(topBar, kernel, iterations=1)
    team1Name = pytesseract.image_to_string(topBar, config= "-c tessedit_char_whitelist=01234567890 --psm 8")
    topBarText.append(team1Name.rstrip('\n'))

    print(loadoutValues)
    print(topBarText)
    print(playersLeft)
    print(ultLeft)
    print(playersRight)
    print(ultRight)
    #return []

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
# table.append()
