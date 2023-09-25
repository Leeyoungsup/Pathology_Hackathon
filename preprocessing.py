import numpy as np
from glob import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import sys

file_not_transition_path = sys.argv[1]
file_transition_path = sys.argv[2]
trans_folder = sys.argv[3]
l1_trans_folder = sys.argv[4]
not_transition_list = glob(str(file_not_transition_path)+'/*.tiff')
transition_list = glob(str(file_transition_path)+'/*.tiff')

transition_image_list = [f.replace(
    '/'+str(trans_folder)+'/', '/'+str(l1_trans_folder)+'/') for f in transition_list]
not_transition_image_list = [f.replace(
    '/'+str(trans_folder)+'/', '/'+str(l1_trans_folder)+'/') for f in not_transition_list]

for i in range(len(not_transition_list)):
    src_img = Image.open(not_transition_list[i])
    src_array_img = np.array(src_img)
    HSI_src_array_img = cv2.cvtColor(src_array_img, cv2.COLOR_RGB2HSV)
    ret, binary_array_img = cv2.threshold(
        HSI_src_array_img[:, :, 1], 127, 255, cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    morph_array_image = cv2.morphologyEx(binary_array_img, cv2.MORPH_CLOSE, k)
    cv2.imwrite(not_transition_image_list[i], morph_array_image)

for i in range(len(transition_list)):
    src_img = Image.open(transition_list[i])
    src_array_img = np.array(src_img)
    HSI_src_array_img = cv2.cvtColor(src_array_img, cv2.COLOR_RGB2HSV)
    ret, binary_array_img = cv2.threshold(
        HSI_src_array_img[:, :, 1], 127, 255, cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    morph_array_image = cv2.morphologyEx(binary_array_img, cv2.MORPH_CLOSE, k)
    cv2.imwrite(transition_image_list[i], morph_array_image)
