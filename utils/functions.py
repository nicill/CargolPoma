
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from fastai.vision import open_image
from imgaug import augmenters as iaa
import random
import torch

def augment(image, code, outputFile, verbose=True):
    if code == 0:
        if verbose: print("Doing Data augmentation 0 (H fip) to image "+outputFile)
        #image_aug = iaa.Fliplr(1.0)(images=image)
        image_aug = iaa.Rot90(1)(image=image)
        cv2.imwrite(outputFile,image_aug)
    elif code==1:
        if verbose: print("Doing Data augmentation 1 (V flip) to image "+outputFile)
        image_aug = iaa.Flipud(1.0)(image=image)
        cv2.imwrite(outputFile,image_aug)
    elif code==2:
        if verbose: print("Doing Data augmentation 2 (Gaussian Blur) to image "+outputFile)
        image_aug = iaa.GaussianBlur(sigma=(0, 0.5))(image=image)
        cv2.imwrite(outputFile,image_aug)
    elif code==3:
        if verbose: print("Doing Data augmentation 3 (rotation) to image "+outputFile)
        angle=random.randint(0,45)
        rotate = iaa.Affine(rotate=(-angle, angle))
        image_aug = rotate(image=image)
        cv2.imwrite(outputFile,image_aug)
    elif code==4:
        if verbose: print("Doing Data augmentation 4 (elastic) to image "+outputFile)
        image_aug = iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.1)(image=image)
        cv2.imwrite(outputFile,image_aug)
    elif code==5:
        if verbose: print("Doing Data augmentation 5 (contrast) to image "+outputFile)
        image_aug=iaa.LinearContrast((0.75, 1.5))(image=image)
        cv2.imwrite(outputFile,image_aug)
    else:
        print("Doing some other Data augmentation to image "+outputFile)
        #https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html


def sliding_windowMosaicMask(image, mask, stepSize, windowSize):

	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]], mask[y:y + windowSize[1], x:x + windowSize[0]])

# receive an image, a patch size and a step, a classifier and a label name,
# create a mask of all the patches that belong to the label according to the classifier


def classifyPatches(mosaic, step, patchSize, net , label=""):
	# create empty output mask
	# create a mask of ones to be added each time a patch is found to contain the label
	outputMask = np.zeros((mosaic.shape[0], mosaic.shape[1]), dtype="uint8")

	for (x, y, mosaicW, maskW) in sliding_windowMosaicMask(mosaic, outputMask, stepSize=step, windowSize=(patchSize, patchSize)):

		# create a mask of ones to be added each time a patch is found to contain the label
		maskOfOnes = 1*np.ones((maskW.shape[0], maskW.shape[1]), dtype="uint8")
		# Can we do the following shit in a less ugly way???? (no, learner.predict(mosaicW) does not work)


        with torch.no_grad():
            #torch.cuda.synchronize()
            #seg_pi, unc_pi, _, tops_pi = self(data_tensor)
            seg_pi = net(data_tensor)
            #print("seg_pi is "+str(seg_pi))
            img=seg_pi[0][0]

            print(img)

"""
		tempImage = cv2.resize(mosaicW, (classifierPatchSize, classifierPatchSize))
		cv2.imwrite("./tempImage.jpg", tempImage)
		img = open_image("./tempImage.jpg")
"""

		pred_class, pred_idx, outputs = learner.predict(img)
		if label in str(pred_class).split(";"):
			# maskW = np.add(maskW, maskOfOnes)
			maskW[:] += 1
			# print(str(np.count_nonzero(maskW)) +" "+str(np.count_nonzero(outputMask)))

	# At the end of the loop, we have added ones every time a patch has been classified as belonging to the class
	oMaskMax = np.max(outputMask)
	heatMapPerc=0.5
	outputMask[outputMask<int(oMaskMax*heatMapPerc)]=0
	outputMask[outputMask!=0]=255
	# print("output mask max!! "+str(oMaskMax))
	return outputMask


def show_img(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def borderPoint(img,p,w_size):

	imgH, imgW = img.shape
	margin = w_size // 2

	return p[0]<margin or (imgW-p[0])<margin or p[1]<margin or (imgH-p[1])<margin


# Function to take a binary image and output the center of masses of its connected regions
# THIS METHOD IS A COPY OF crownSectmenterEvaluator method! must be deleted!!!
def listFromBinary(im, patchSize):
	# open filename
	# im=cv2.imread(fileName,cv2.IMREAD_GRAYSCALE)
	if im is None: return []
	else:
		mask = cv2.threshold(255-im, 40, 255, cv2.THRESH_BINARY)[1]

		# compute connected components
		numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)
		# print("crownSegmenterEvaluator, found "+str(numLabels)+" "+str(len(centroids))+" points for file "+fileName)

		# im2 = 255 * np. ones(shape=[im.shape[0], im.shape[1], 1], dtype=np. uint8)

		# print(" listFromBinary, found  "+str(len(centroids)))
		# print(centroids)

		newCentroids=[]
		for c in centroids:
			if not borderPoint(im,c,patchSize): newCentroids.append(c)
		# print(" listFromBinary, refined  "+str(len(newCentroids)))
		# print(newCentroids)

		# print("old centroids: "+str(len(centroids))+" new centroids: "+str(len(newCentroids)))

		# SAVING IMAGE OF SELECTED CENTROIDS
		# check_mask = np.zeros((mask.shape[0], mask.shape[1],3), np.uint8)
		# check_mask[:] = (255,255,255)
		# for c in newCentroids:
		# 	check_mask = cv2.circle(check_mask, (int(c[0]), int(c[1])), 10, (255,0,0), 5)
		# cv2.imwrite("check_mask.jpg", check_mask)
		# print("check_mask saved")

		return newCentroids[1:]


def getSquare(w_size, p, img):

	height, width, _ = img.shape

	# isInside = (int(p[0])-w_size//2) >= 0 and (int(p[0])+w_size//2) < width and (int(p[1])-w_size//2) >= 0 and (int(p[1])+w_size//2) < height

	# assert isInside, "The required window is out of bounds of the input image"

	# opencv works with inverted coords, so we have to invert ours.
	return img[int(p[1])-w_size//2:int(p[1])+w_size//2, int(p[0])-w_size//2:int(p[0])+w_size//2]

def isInLayer(center,layer):
	return layer[int(center[0]),int(center[1])]==255
