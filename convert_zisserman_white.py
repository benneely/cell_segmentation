import os
import json
import PIL.Image

CELL_SAVE = 'distal_acinar_tubule_cells'

filenames_old = [
    'via_region_data_1.json',
    'via_region_data_2.json',
    # '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001__45.json',
    # '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001__47.json',
    # '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001__62.json',
    # '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001__66.json'
]

filenames_new = [
    # 'via_region_data_1.json',
    # 'via_region_data_2.json',
    '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001__45.json',
    '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001__47.json',
    '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001__62.json',
    '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001__66.json'
]

filenames_dual = [
    # '16kb20182015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_004__107.json',
    # '24kb20182015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_004__107.json',
    # '32kb2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001__104.json',
    # '41kb 2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_002__104.json',
    '029_20X_C57Bl6_E16.5_LMM.json',
    # '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46.json'
]

filenames_mine = [
    'acta2_001_047.json',
    'acta2_001_113.json',
    'acta2_001_119.json',
    'acta2_002_038.json',
    'acta2_002_039.json',
    'acta2_002_040.json',
    'acta2_002_052.json',
    'acta2_002_102.json',
    'acta2_003_052.json',
    'acta2_004_084.json',
    'acta2_004_121.json',
    'acta2_004_117.json',
    'acta2_004_108.json',
    'acta2_004_103.json',
    'acta2_004_100.json',
    'acta2_004_95.json',
    'acta2_004_085.json',
    'acta2_004_071.json',
    'acta2_004_062.json',
    'acta2_004_050.json'

]

def extractit(filenames, cell_only=True):
    jdat = {}
    for file in filenames:
        complete_file_path = os.path.join(CELL_SAVE, file)
        with open(complete_file_path) as f:
            jdat.update(json.load(f))
    try:
        jdat.pop('2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001__45.png41696')
    except:
        pass

    final = {}
    for k, v in jdat.items():
        if len(v['regions'])>0:
            final[k] = v

    new = {}
    if cell_only:
        for k, v in final.items():
            new[v['filename']] = {}
            new[v['filename']]['cell'] = []
            for key, value in v['regions'].items():
                new[v['filename']]['cell'].append(
                    [list(a) for a in zip(value['shape_attributes']['all_points_x'],value['shape_attributes']['all_points_y'])]
                )
    else:
        for k, v in final.items():
            new[v['filename']] = {}
            new[v['filename']]['green'] = []
            new[v['filename']]['blue'] = []
            for key, value in v['regions'].items():
                if 'label' not in list(value['region_attributes'].keys()):
                    raise('Problem')
                if value['region_attributes']['label'] in ['green cell', 'green']:
                    new[v['filename']]['green'].append(
                        [list(a) for a in zip(value['shape_attributes']['all_points_x'],value['shape_attributes']['all_points_y'])]
                    )
                else:
                    new[v['filename']]['blue'].append(
                        [list(a) for a in
                         zip(value['shape_attributes']['all_points_x'], value['shape_attributes']['all_points_y'])]
                    )
    return new

# old = extractit(filenames_old)
# new = extractit(filenames_new)
neww = extractit(filenames_mine, cell_only=False)
final_dict = {**neww}
# final_dict = {**new, **old}


with open('distal_acinar_tubule_cells/regions.json', 'w') as f:
    json.dump(neww, f)

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon

for k, v in final_dict.items():
    img = PIL.Image.open('distal_acinar_tubule_cells/'+k)
    imgarr = np.array(img)
    mask = np.zeros(shape=(imgarr.shape[0], imgarr.shape[1]))
    for x in v['green']:
        xarr = np.array(x)
        rr, cc = polygon(
            xarr[:,1],
            xarr[:,0]
        )
        mask[rr, cc] = 1
    for x in v['blue']:
        xarr = np.array(x)
        rr, cc = polygon(
            xarr[:,1],
            xarr[:,0]
        )
        mask[rr, cc] = 1
    plt.imshow(np.array(img))
    plt.imshow(mask, alpha=0.4)
    plt.show()