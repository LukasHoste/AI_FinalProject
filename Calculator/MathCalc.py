# Import the necessary packages
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import scipy.special
import math

# PROBLEMS WITH ORDERING FROM WHITEBOARD, MAYBE ASK THE USER TO ONLY TAKE A PICTURE OF THE BRACKETS IF IT DOESNT WORK

# This is only meant for one occurence of these chars
def getDataBetweenChar(firstChar,finalChar, string):
    result = ""
    for i in range(len(string)):
        
        # print(string[i])
        if string[i] == firstChar:
            # print("firstchar encountered")
            result = ""
        elif(string[i] == finalChar):
            # print("final")
            return result
        else:
            # print(result)
            result+= string[i]

# calculate the binomial function from a given string
def calculateBinomial(string):
    val1 = string.split('-')[0]
    val2 = string.split('-')[1]
    try:
        int(val1)
        int(val2)
    except:
        return "problem with values passed to calculateBinomial, might be due to wrong order"
    print(int(val1))
    print(int(val2))
    return scipy.special.comb(int(val1),int(val2))

# get values a, b, c from the quadratic formula
def getQuadVals(string):
    vals = ["","",""]
    valIndex = 0
    afterX = False
    # negative = False
    for i in range(len(string)):
        if string[i] == "x":
            valIndex+=1
            afterX = True
        elif string [i] == "+" or string[i] == "-":
            # continue
            afterX = False
            if string[i] == "-":
                # negative = True
                vals[valIndex] += string[i]
        elif not afterX:
            vals[valIndex] += string[i]
    return vals

# solve the quadratic formula (e.g. 5x^2 + 4x -1 or 5x^2 + 1)
def CalculateX(values, string):
    xCount = string.count('x')
    print("number of x: "+ str(xCount))
    if xCount == 1:
        try:
            a = int(values[0])
            b = 0
            c = int(values[1])
        except:
            print("Did not recognize one of your values or the forumla type was wrong")
            return
    else:
        try:
            a = int(values[0])
            b = int(values[1])
            c = int(values[2])
        except:
            print("Did not recognize one of your values or the forumla type was wrong")
            return
    # c= 0
    discriminant = (b**2)-(4*a*c)
    print("discriminant" + str(discriminant))
    if discriminant >= 0:
        x1 = (-b-math.sqrt(discriminant))/(2*a)
        x2 = (-b+math.sqrt(discriminant))/(2*a)
        print("The solutions are " +str(x1)+ " and " + str(x2))
    else:
        print("Complex Roots")  
        CompA = (-b)/(2*a)
        compB = (math.sqrt(abs(discriminant)))/(2*a)
        print(str(CompA) + " + " + str(compB) + "i and " + str(CompA) + " - " + str(compB) + "i")


 
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-m", "--model", required=True, help="Path to the pre-trained model")
args = vars(ap.parse_args())

###############################################
# This takes two stages
# The first stage is to segment characters
# The second stage is to recognise characters
###############################################

###############################################
# The first stage
###############################################

# Read the image and convert to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show the original image
cv2.imshow("License Plate", image)

# Apply Gaussian blurring and thresholding 
# to reveal the characters on the license plate
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# apply adaptive thresholding
thresh = cv2.adaptiveThreshold(blurred, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 255, 5)
# TEST FOR WHITEBOARD
# 75 for whiteboard seems to be better than 45, setting it to 255 seems to give even better results on img 5, however order of chars is wrong though
# so solution would be to be able to select whiteboard in app, or maybe there should be more white padding around the img ?
# thresh = cv2.adaptiveThreshold(blurred, 255,
# 	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 75, 13)

# show after thresholding
cv2.imshow("threshold", thresh)

# Perform connected components analysis on the thresholded images and
# initialize the mask to hold only the components we are interested in
_, labels = cv2.connectedComponents(thresh)
mask = np.zeros(thresh.shape, dtype="uint8")

# Set lower bound and upper bound criteria for characters
total_pixels = image.shape[0] * image.shape[1]

# IF CHARS NOT RECOGNIZED AAAAANNNNNPASSSEN!!!!!!!!

lower = total_pixels // 2500 # heuristic param, can be fine tuned if necessary
# TEST FOR WHITEBOARD
# lower = total_pixels // 5000 # heuristic param, can be fine tuned if necessary
upper = total_pixels // 20    # heuristic param, can be fine tuned if necessary

# Loop over the unique components
for (i, label) in enumerate(np.unique(labels)):
	# If this is the background label, ignore it
	if label == 0:
		continue
 
	# Otherwise, construct the label mask to display only connected component
	# for the current label
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)
 
	# If the number of pixels in the component is between lower bound and upper bound, 
	# add it to our mask
	if numPixels > lower and numPixels < upper:
		mask = cv2.add(mask, labelMask)

# Find contours and get bounding box for each contour
cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundingBoxes = [cv2.boundingRect(c) for c in cnts]

cv2.imshow("removed", mask)




# Sort the bounding boxes from left to right, top to bottom
# sort by Y first, and then sort by X if Ys are similar
def compare(rect1, rect2):
    if abs(rect1[1] - rect2[1]) > 250:
        return rect1[1] - rect2[1]
    else:
        return rect1[0] - rect2[0]
# boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )

# testing sorting methods
# Calculate maximum rectangle height
c = np.array(boundingBoxes)
max_height = np.max(c[::, 3])
# print("max height: "+ str(max_height))

# Sort the contours by y-value
by_y = sorted(boundingBoxes, key=lambda x: x[1])  # y values

line_y = by_y[0][1]       # first y
line = 1
by_line = []

# Assign a line number to each contour
for x, y, w, h in by_y:
    if y > line_y + max_height:
    # if y > line_y + max_height*2:
    # Works quite good but is hardcoded
    # if y > line_y + max_height+(max_height/4):
        line_y = y
        line += 1
        
    by_line.append((line, x, y, w, h))

# This will now sort automatically by line then by x
boundingBoxes = [(x, y, w, h) for line, x, y, w, h in sorted(by_line)]

for x, y, w, h in boundingBoxes:
    print(f"{x:4} {y:4} {w:4} {h:4}")



###############################################
# The second stage
###############################################

# Define constants
TARGET_WIDTH = 128
TARGET_HEIGHT = 128

# chars = [
#     '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G',
#     'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
#     ]

# the labels
chars = [
    '0','1','2','3','4','5','6','7','8','9','-','+','x'
    ]
    
# Load the pre-trained convolutional neural network
model = load_model(args["model"], compile=False)

vehicle_plate = ""
# Loop over the bounding boxes
for rect in boundingBoxes:

    # Get the coordinates from the bounding box
    x,y,w,h = rect

    # Crop the character from the mask
    # and apply bitwise_not because in our training data for pre-trained model
    # the characters are black on a white background
    crop = mask[y:y+h, x:x+w]
    crop = cv2.bitwise_not(crop)
    # cv2.imshow("before-make-border", crop)

    # Get the number of rows and columns for each cropped image
    # and calculate the padding to match the image input of pre-trained model
    rows = crop.shape[0]
    columns = crop.shape[1]
    
    paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
    paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)
    
    # cv2.imshow("befor make border",crop)
    # Apply padding to make the image fit for neural network model
    # MIGHT HAVE TO MAKE THIS A LITTLE LESS HARDCODED
    if rows > columns*2: 
        print("rectangle height")
        crop = cv2.copyMakeBorder(crop, paddingY//2, paddingY//2, 275, 275, cv2.BORDER_CONSTANT, None, 255)
    elif columns > rows*2:
        print("rectangle width")
        crop = cv2.copyMakeBorder(crop,275,275,paddingX,paddingX,cv2.BORDER_CONSTANT,None,255)
    else:
        crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)
    # crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)



    # Convert and resize image
    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)   
    cv2.imshow("crop-before-resize",crop)

    crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))
    # crop = imutils.resize(crop, width=TARGET_WIDTH, height=TARGET_HEIGHT)

    cv2.imshow("crop-after-resize",crop)
    # Prepare data for prediction
    crop = crop.astype("float") / 255.0
    iteration = str(i)
    # cv2.imshow("crop"+ iteration, crop.copy())
    cv2.imshow("bracket", crop)
    crop = img_to_array(crop)
    crop = np.expand_dims(crop, axis=0)

    #Make prediction
    prob = model.predict(crop)[0]
    idx = np.argsort(prob)[-1]
    print(idx)
    vehicle_plate += chars[idx]
    prediction = chars[idx]


    # Show bounding box and prediction on image
    cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)
    cv2.putText(image, prediction, (x,y+15), 0, 0.8, (0, 0, 255), 2)

# Show final image
cv2.imshow('Final', image)
print("Vehicle plate: " + vehicle_plate)

print("What formula would you like to solve (binomial, quadratic): ")
formula = input()
if formula == "quadratic":
    values = getQuadVals(vehicle_plate)
    CalculateX(values, vehicle_plate)
    print(values)
elif formula == "binomial":

    betweenBrackets = getDataBetweenChar('(',')', vehicle_plate)
    if betweenBrackets == None:
        print("could not recognize enough chars or the forumla type was wrong")
    else:
        if "-" in betweenBrackets:
            
            result = calculateBinomial(betweenBrackets)
            print(result)
        else:
            print("could not properly recognise the minus in the image please try again or the forumla type was wrong")
else:
    print("Not a recognized formula type")
cv2.waitKey(0)