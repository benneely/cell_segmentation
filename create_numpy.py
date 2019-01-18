import os
import json
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
from skimage.draw import polygon

CELL_SAVE = 'distal_acinar_tubule_cells'

filenames = [
    'via_region_data_1.json',
    'via_region_data_2.json'
]

jdat = {}
for file in filenames:
    complete_file_path = os.path.join(CELL_SAVE, file)
    with open(complete_file_path) as f:
        jdat.update(json.load(f))
jdat.pop('2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001__45.png41696')
imgs = []
masks = []
for k, v in jdat.items():
    img = imread(os.path.join(CELL_SAVE, v['filename']))
    mask = np.zeros(shape=(img.shape[0], img.shape[1], 1))
    images_resize = np.zeros(shape=((128, 128, 3)))
    mask_resize = np.zeros(shape=((128, 128, 1)))
    images_resize[:, :, :] = scipy.misc.imresize(img, (128, 128, 3))
    for key, value in v['regions'].items():
        # contour = list(zip(value['shape_attributes']['all_points_x'], value['shape_attributes']['all_points_y']))
        # np_contour = np.array(contour)
        # cv2.polylines(img, [np_contour], True, (255, 0, 255), 1)
        rr, cc = polygon(
            value['shape_attributes']['all_points_y'],
            value['shape_attributes']['all_points_x']
        )
        mask[rr, cc, 0] = 1
    mask_resize[:, :, 0] = scipy.misc.imresize(mask.squeeze(), (128, 128))
    mask_resize = mask_resize/255
    imgs.append(images_resize.astype('uint8'))
    masks.append(mask_resize.astype('uint8'))

final_images = np.stack(imgs)
final_masks = np.stack(masks)

np.save('images.npy', final_images)
np.save('masks.npy', final_masks)


for k, v in jdat.items():
    img = imread(os.path.join(CELL_SAVE, v['filename']))
    print(v['filename'])
    plt.imshow(img)
    plt.title(v['filename'])
    plt.show()