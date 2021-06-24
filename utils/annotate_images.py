# importing the module
import sys
import os
import cv2
import numpy as np
import argparse
from pathlib import Path

from numpy.core.numeric import ones_like, zeros_like

class select_interesting_points():

    def __init__(self, img, imgs_path, cat, masks_path, inverse_scale_factor, white_dots, keep_extension):

        image_full_path = imgs_path + '/' + img
        self.image = img
        self.original_image = cv2.imread(image_full_path, 1)
        self.category = cat
        self.masks_path = masks_path
        self.inverse_scale_factor = inverse_scale_factor
        self.white_dots = white_dots
        self.keep_extension = keep_extension

        self.original_width, self.original_height = self.original_image.shape[:2]
        self.img = cv2.resize(self.original_image, (int(self.original_height/inverse_scale_factor), int(self.original_width/inverse_scale_factor)))
        self.clickedList = []

        self.window_name = 'Select all the ' + self.category
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, self.img)
        cv2.setMouseCallback(self.window_name, self.click_event)

        cv2.waitKey(0)
        self.print_mask()
        cv2.destroyWindow(self.window_name)


    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell

            # displaying the coordinates
            # on the image window
            cv2.circle(self.img, (x, y), 10, (255, 255, 255), 2)
            cv2.circle(self.img, (x, y), 9, (0, 0, 0), 2)


            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 255, 255), 2)
            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x-1, y-1), font,
                        1, (0, 0, 0), 2)
            cv2.imshow(self.window_name, self.img)

            self.clickedList.append((x, y))


    def print_mask(self):


        if self.white_dots:
            mask_image = np.zeros_like(self.original_image)
            dot_color = (255, 255, 255)
        else:
            mask_image = np.ones_like(self.original_image)
            mask_image *= 255
            dot_color = (0, 0, 0)


        for clicked_point in self.clickedList:
            clicked_point_res = tuple(int(x* inverse_scale_factor) for x in clicked_point)
            cv2.circle(mask_image, clicked_point_res, 30, dot_color, -1)

        if self.keep_extension:
            ext = self.image[-4:]
        else:
            ext = ".jpg"

        cv2.imwrite(str(self.masks_path.joinpath(self.image[:-4] + '_' + self.category + ext).absolute()) , mask_image)



# USAGE
# python annotate_images.py <path_to_images> <label1 label2 ... labeln> [--white_dots] [--keep_extension] [--inverse_scale_factor]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Mask generator annotation tool.')

    parser.add_argument('images_path',  help='Path of the images to create annotated masks from')
    parser.add_argument('categories_list', nargs='+', help='Categories to be annotated')
    parser.add_argument('--white_dots', help='To invert the resulting mask: white dots over black background', action='store_true')
    parser.add_argument('--inverse_scale_factor',type=int, help='Ratio to resice images to fit the screen', default=4)
    parser.add_argument('--keep_extension', help='Keep original file extension. Default save in jpg', action='store_true')

    args = parser.parse_args()

    images_path = args.images_path
    categories = args.categories_list
    white_dots = args.white_dots
    keep_extension = args.keep_extension

    inverse_scale_factor = args.inverse_scale_factor

    p = Path(images_path).parents[0]
    masks_path = p.joinpath('masks').absolute()
    if not os.path.isdir(masks_path):
        os.makedirs(masks_path)

    images_abs_path = Path(images_path).absolute().as_posix()
    for image in os.listdir(images_abs_path):
        for cat in categories:
            select_interesting_points(image, images_abs_path, cat, masks_path, inverse_scale_factor, white_dots, keep_extension)





