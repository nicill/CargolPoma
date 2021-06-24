# This method takes the point coordinates of each tree top given by crownSegementerEvaluator and extracts
# a squared patch arround each one. Then, uses the classified masks of the images to know in which species belongs.
# Then stores each small labeled patch in a folder.

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from pathlib import Path
from utils.functions import *
import os

folderInfo = namedtuple("folderInfo","mainPath imageFolder maskFolder outputFolder augmentation numClasses layerNameList" )
allowed_ext = [".jpg", ".JPG", ".jpeg", ".JPEG"]
_ext = ".jpg"


def interpretParameters(paramFile, verbose=False):

	# read the parameter file line by line
	f = open(paramFile, "r")
	patchSize = -1
	imageDict = {}
	layerNameList = []
	augmentation = False

	for x in f:
		lineList = x.split(" ")
		# read every line
		first = lineList[0]

		if first[0] == "#":  # if the first character is # treat as a comment
			if verbose:print("COMMENT: "+str(lineList))
		elif first == "\n":  # account for blank lines, do nothing
			pass
		elif first == "patchSize":
			patchSize = int(lineList[1].strip())
			if verbose:print("Read Patch Size : "+str(patchSize))
		elif first == "images":

			# read the number of layers and set up reading loop
			mainPath = lineList[1]
			imageFolder = lineList[2]
			maskFolder = lineList[3]
			outputFolder = lineList[4]
			if lineList[5] == "augmentation":
				augmentation = True
			numClasses = int(lineList[6].strip())
			for i in range(numClasses):
				layerNameList.append(lineList[7+i])

			# make dictionary entry for this path
			imageDict[imageFolder] = folderInfo(mainPath, imageFolder, maskFolder, outputFolder, augmentation, numClasses, layerNameList)

			if verbose:
				print("\n\n\n")
				# print(imageDict[image])
				# print("\n\n\n")
				# print("Read layers and file : ")
				# print("filePath "+filePath)
				# print("image "+image)
				# print("num Classes "+str(numClasses))
				# print("layerName List "+str(layerNameList))
				# print("layer List "+str(layerList))
				# print("outputFolder "+outputFolder)
		else:
			raise Exception("ImagePatchAnnotator:interpretParameters, reading parameters, received wrong parameter "+str(lineList))

		if verbose:
			print(imageDict)

	return patchSize, imageDict


def main(argv):
	try:
		# verbose = False
		patchSize, imageDict = interpretParameters(argv[1])

		# if verbose: print(imageDict)
		for name, info in imageDict.items():  # FOR EACH folder

			imageFolder = info.mainPath + info.imageFolder
			maskFolder = info.mainPath + info.maskFolder
			outputFolder = info.mainPath + info.outputFolder
			augmentation = info.augmentation
			counters = {}
			for layerName in info.layerNameList:
				counters[layerName] = 0
			
			# reading image folder
			for file in os.listdir(imageFolder):
				imageName, file_ext = os.path.splitext(file)
				if file_ext not in allowed_ext:
					print("Unknown file format")
					continue
					
				image = cv2.imread(imageFolder + file, cv2.IMREAD_COLOR)

				for layerName in info.layerNameList:  # FOR EACH CLASS

					layerFilePath = maskFolder + imageName + "_" + layerName + _ext

					my_file = Path(layerFilePath)
					if not my_file.is_file():
						print("There is no mask of class [" + layerName + "] for image " + imageName)
						continue
						
					centroids = listFromBinary(cv2.imread(layerFilePath, cv2.IMREAD_GRAYSCALE), patchSize)

					# layer = cv2.bitwise_not(cv2.imread(layerFilePath, cv2.IMREAD_GRAYSCALE))

					for cent in centroids:  # FOR EACH ELEMENT
						try:
							square = getSquare(patchSize, (cent[0],cent[1]), image)
							ok_saved = cv2.imwrite(outputFolder + layerName + "PATCH" + str(counters[layerName]) + _ext, square)
							if not ok_saved: print("FAIL PRINTING")
							counters[layerName] += 1

							if augmentation:
								for i in range(6):

									augment(square, i, outputFolder + layerName + "PATCH" + str(counters[layerName]) + _ext, False)
									counters[layerName] += 1

						except AssertionError as error:
							print(error)

			print(counters)

	except AssertionError as error:
		print(error)

	# once this is finished, we should consider using the data augmenter to create more images.
	# for example, loop over all images, and generate a fixed number of augmented images per image
	# (loop over all generated patches), for each patch call "augment" with different code, store all images
	# check function processCSVFile in multiLabelDataAugmenter for details on hot to call augment and store the resulting images


# Exectuion example -> python cargolPomaPatcher.py <path_to_params_file>

if __name__ == "__main__":
	main(sys.argv)
