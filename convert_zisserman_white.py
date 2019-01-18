import os
import json
import PIL


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
new = {}

for k, v in jdat.items():
    new[v['filename']] = {}
    new[v['filename']]['cell'] = []
    for key, value in v['regions'].items():
        new[v['filename']]['cell'].append(
            [list(a) for a in zip(value['shape_attributes']['all_points_x'],value['shape_attributes']['all_points_y'])]
        )

with open('distal_acinar_tubule_cells/regions.json', 'w') as f:
    json.dump(new, f)

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon

for k, v in new.items():
    img = PIL.Image.open('distal_acinar_tubule_cells/'+k)
    imgarr = np.array(img)
    mask = np.zeros(shape=(imgarr.shape[0], imgarr.shape[1]))
    for x in v['cell']:
        xarr = np.array(x)
        rr, cc = polygon(
            xarr[:,1],
            xarr[:,0]
        )
        mask[rr, cc] = 1
    plt.imshow(np.array(img))
    plt.imshow(mask, alpha=0.4)
    plt.show()