import numpy as np
import cv2
import math

# Load an imgae
img = cv2.imread('./lotto_gray.jpg')

def showImage(res): 
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

# detect edges in the image
edged = cv2.Canny(img, 10, 250)

# construct and apply a closing kernel to 'close' gaps between 'white'
# pixels
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

# find contours (i.e. the 'outlines') in the image 
(_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if the approximated contour has four points,
    # then assume that the contoure is a rectangle and thus has four vertices
    if len(approx) == 4:
        # cv2.drawContours(img, [approx], -1, (0, 255, 0), 4)
        # showImage(img)

        # cv2.minAreaRect: returns values in the 
        # range [-90, 0); as the rectangle rotates clockwise the
        # returned angle trends to 0 -- in this special case we
        # need to add 90 degrees to the angle
        rect = cv2.minAreaRect(approx)
        angle = rect[-1]
        
        # if angle < -45:
        #     angle = -(90 + angle)

        # otherwise, just take the inverse of the angle to make
        # it positive
        # else:
        #     angle = -angle

        # rotate the image to deswke it
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
        rotatedApprox = cv2.transform(approx, M)

        # Crop image
        x, y, w, h = cv2.boundingRect(rotatedApprox)
        cropImg = rotated[(y + 10):(y+h - 10), (x + 10):(x+w - 10)]

# showImage(cropImg)

# Convert image to binary
# Threshold = 127
(thresh, imBW) = cv2.threshold(cropImg, 200, 255, cv2.THRESH_BINARY)

height, width, channel = imBW.shape
        
# showImage(imBW)

# find two lines
edges = cv2.Canny(imBW, 50, 150, apertureSize = 3)

minLineLength = width - 500
lines = cv2.HoughLinesP(image = edges, rho = 0.02, 
theta = np.pi/500, threshold = 10, lines = np.array([]), minLineLength = minLineLength, maxLineGap = 5)

checkedLines = [] 
for line in lines:
    x1, y1, x2, y2 = line[0]
    if len(checkedLines) == 0:
        checkedLines.append(((x1, y1), (x2, y2)))
        cv2.line(imBW, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        recentLine = checkedLines[-1] 
        p1, p2 = recentLine
        recentAvgY = int((p1[-1] + p2[-1]) / 2)
        avgY = int((y1 + y2) / 2)

        if abs(recentAvgY - avgY) > 100:
            cv2.line(imBW, (x1, y1), (x2, y2), (0, 255, 0), 2)
            checkedLines.append(((x1, y1), (x2, y2)))
            break

y1 = checkedLines[0][0][1]
y2 = checkedLines[1][0][1]

# Swap
if y1 > y2:
    temp = y1
    y1 = y2
    y2 = temp

numSecImg = imBW[(y1 + 10):(y2 - 10), 10:(width - 10)]
numSecImg = cv2.cvtColor(numSecImg, cv2.COLOR_BGR2GRAY)
height, width = numSecImg.shape
# showImage(numSecImg)

# Projection
horiProjection = np.zeros(height, dtype='int32')
vertiProjection = np.zeros(width, dtype='int32')

numStartEndPoints = []
numHeightStartEndPoints = []

# Find the start and end point of the number and row
with open("horiProjection.txt", "w") as hFile, open("vertiPorjection.txt", "w") as vFile:
    for p, value  in np.ndenumerate(numSecImg):
        if value == 0:
            y, x = p
            horiProjection[y] = horiProjection[y] + 1
            vertiProjection[x] = vertiProjection[x] + 1

    recent = 0
    pair = []
    for idx, v in np.ndenumerate(horiProjection):
        idx = idx[0]  
        hFile.write(str(idx))
        hFile.write(' ')
        hFile.write(str(v))
        hFile.write('\n')

        if (v == 0 and recent != 0) or (v != 0 and recent == 0):
            pair.append(idx)
            if len(pair) == 2:
                numHeightStartEndPoints.append(pair)
                pair = []
        recent = v

    pair = []
    for idx, v in np.ndenumerate(vertiProjection):
        idx = idx[0]
        vFile.write(str(idx))
        vFile.write(' ')
        vFile.write(str(v))
        vFile.write('\n')

        if (v == 0 and recent != 0) or (v != 0 and recent == 0):
            pair.append(idx)
            if len(pair) == 2:
                numStartEndPoints.append(pair)
                pair = []
        recent = v

    hFile.write(str(numHeightStartEndPoints))
    vFile.write(str(numStartEndPoints))
    numStartEndPoints = numStartEndPoints[2:]
    
    hFile.close()
    vFile.close()

# Slice the row into a digit 
lottoNumImgs = np.array([]) 
lottoNumProjection = np.array([]) 
for heightStartEndPoint in numHeightStartEndPoints:
    row = np.array([]) 
    projectionRow = np.array([]) 
    for numStartEndPoint in numStartEndPoints:
        y1, y2 = heightStartEndPoint
        x1, x2 = numStartEndPoint
        # Get image part
        num = numSecImg[y1:y2, x1:x2]
        for _ in range(30 - num.shape[0]):
            num = np.append(num, [[255 for _ in range(num.shape[1])]], axis=0)
        for _ in range(20 - num.shape[1]):
            num = np.insert(num, [num.shape[1] - 1], [[255] for _ in range(num.shape[0])], axis=1)
        num = num.flatten()
        if row.size == 0:
            row = np.array([num]) 
        else:
            row = np.append(row, [num], axis=0)

        # Get Projection Part
        projection = np.array([])
        hori = horiProjection[y1:y2]
        verti = vertiProjection[x1:x2]

        if len(hori) < 30:
            hori = np.append(hori, [0 for _ in range(30 - hori.size)])
        projection = np.append(projection, hori, axis=0)
        if len(verti) < 20:
            verti = np.append(verti, [0 for _ in range(20 - verti.size)])
        projection = np.append(projection, verti, axis=0)
        if projectionRow.size == 0:
            projectionRow = np.array([projection])
        else:
            projectionRow = np.append(projectionRow, [projection], axis=0)

    if lottoNumImgs.size == 0:
        lottoNumImgs = np.array([row])
    else:
        lottoNumImgs = np.append(lottoNumImgs, [row], axis=0)

    if lottoNumProjection.size == 0:
        lottoNumProjection = np.array([projectionRow])
    else:
        lottoNumProjection = np.append(lottoNumProjection, [projectionRow], axis=0)


from perceptron import Perceptron

if __name__ == '__main__':
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    d = np.array([
        [0],
        [0],
        [0],
        [1]
    ])

    perceptron = Perceptron(inputSize = 2)
    perceptron.training(X, d)

# Using projection data
perceptron = Perceptron(inputSize=lottoNumProjection[0][0].size, numberOfNuerons=10, learningRate=0.10) 

expectedOutput = np.array([])
expectedValues = [
    [1, 5, 1, 8, 2, 0, 2, 6, 3, 7, 4, 4],
    [0, 1, 0, 7, 0 ,8, 1, 6, 2, 3, 2, 8],
    [0, 4, 0, 5, 1, 4, 2, 0, 2, 6, 4, 4],
    [0, 1, 0, 2, 1, 2, 1, 5, 2, 8, 3, 6],
    [0, 2, 1, 1, 2, 0, 2, 1, 3, 1, 3, 8]
]

def changePredcitedToInt(predicted):
    resultStr = ''
    for i in range(predicted.size):
       if predicted[i] == 1:
           resultStr = resultStr +  str(i) 
    return resultStr

def makeExpectedArray(num):
    expected = np.zeros(10)
    expected[num] = 1
    return expected

# training all of them
for i in range(lottoNumProjection.shape[0]):
    expectedRow = np.array([])
    for v in expectedValues[i]:
        expected = makeExpectedArray(v)
        if expectedRow.size == 0:
            expectedRow = np.array([expected]) 
        else:
            expectedRow = np.append(expectedRow, [expected], axis=0)
    perceptron.training(lottoNumProjection[i], expectedRow)

# predicting all of them
numberOfAnswer = 0
for i in range(lottoNumProjection.shape[0]):
    resultStr = ''
    for j in range(lottoNumProjection[i].shape[0]):
        predicted = changePredcitedToInt(perceptron.predict(lottoNumProjection[i][j]))
        resultStr = resultStr + predicted
        if (j+1) % 2 == 0 and j != 0:
            resultStr = resultStr + ' ' 
        if expectedValues[i][j] == int(predicted):
            numberOfAnswer = numberOfAnswer + 1 
    print(resultStr)

print('Anwsert Rate: ', float(numberOfAnswer/60)*100.0)
        
showImage(numSecImg)

print('=======')
# Using original pixels of images
perceptron = Perceptron(lottoNumImgs[0][0].size, numberOfNuerons=10, learningRate=0.3)

# training
for i in range(lottoNumImgs.shape[0]):
    expectedRow = np.array([])
    for v in expectedValues[i]:
        expected = makeExpectedArray(v)
        if expectedRow.size == 0:
            expectedRow = np.array([expected]) 
        else:
            expectedRow = np.append(expectedRow, [expected], axis=0)
    perceptron.training(lottoNumImgs[i], expectedRow)

# predicting
numberOfAnswer = 0
for i in range(lottoNumImgs.shape[0]):
    resultStr = ''
    for j in range(lottoNumImgs[i].shape[0]):
        predicted = changePredcitedToInt(perceptron.predict(lottoNumImgs[i][j]))
        resultStr = resultStr + predicted
        if (j+1) % 2 == 0 and j != 0:
            resultStr = resultStr + ' ' 
        if expectedValues[i][j] == int(predicted):
            numberOfAnswer = numberOfAnswer + 1 
    print(resultStr)

print('Anwsert Rate: ', float(numberOfAnswer/60)*100.0)


showImage(numSecImg)




