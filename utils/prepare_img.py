import sys
import cv2
import os

_ext = ".jpg"

def main(argv):

    imageFolder = argv[1]
    invert = argv[2]

    dest_path = imageFolder + "ready/"

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for file in os.listdir(imageFolder):

        imageName, file_ext = os.path.splitext(file)

        image = cv2.imread(imageFolder + file, cv2.IMREAD_COLOR)

        if image is not None:
            print("Preparing "+ imageName + " image....")

            if invert == "INVERT":
                image = 255 - image

            cv2.imwrite(os.path.join(dest_path, imageName + ".jpg"), image)


if __name__ == "__main__":
	main(sys.argv)
