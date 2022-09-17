import os
import sys
import json
import glob
import shutil

bs_dir = "/home/data/lwb/parking_0614"
sv_dir = r'/home/data/lwb/incremental_datasets/train/in99_0614'
files = os.listdir(bs_dir)
cnt = 0
for file in files:
    if ".json" not in file:
        continue
    cnt += 1
    d = {}
    json_file = (os.sep).join([bs_dir, file])
    with open(json_file, "r") as f:
        js_data = json.load(f)
    # print(js_data)
    file_name = js_data['file_name']
    name = file_name.split('.')[0]
    image_dir = (os.sep).join([sv_dir, 'image'])
    label_dir = (os.sep).join([sv_dir, 'label'])
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    dst_json_path = (os.sep).join([label_dir, name + '.json'])
    ori_png_path = (os.sep).join([bs_dir, name + '.png'])
    dst_png_path = (os.sep).join([image_dir, name + '.png'])
    d['imgName'] = name
    d['resolution'] = [1050, 350]
    d['datasetName'] = 'in99_0614'
    d['subimgType'] = 'right'
    d['cmPerPixel'] = 1.  # 1 pixel corresponds to 1 cm
    labels = js_data['labels']
    d['isOccupied'] = []
    d['points_x'] = []
    d['points_y'] = []
    for i in range(len(labels)):
        tag = labels[i]['property_coll']['is_take_place']['value']
        d['isOccupied'].append(0 if tag == "no" else 1)
        tmp_x = {}
        tmp_y = {}
        for p in labels[i]['landmark']:
            if   p['num'] == '1':
                tmp_x['B'] = int(p['x'])
                tmp_y['B'] = int(p['y'])
            elif p['num'] == '2':
                tmp_x['A'] = int(p['x'])
                tmp_y['A'] = int(p['y'])
            elif p['num'] == '3':
                tmp_x['D'] = int(p['x'])
                tmp_y['D'] = int(p['y'])
            # C点不需要
        tmp_x['C'] = 0
        tmp_y['C'] = 0
        d['points_x'].append([tmp_x['A'],
                              tmp_x['B'],
                              tmp_x['C'],
                              tmp_x['D']])
        d['points_y'].append([tmp_y['A'],
                              tmp_y['B'],
                              tmp_y['C'],
                              tmp_y['D']])
    if len(d['points_x']) == 0:
        continue
    # copy img
    shutil.copyfile(ori_png_path, dst_png_path)
    # save transformed result
    with open(dst_json_path, 'w') as f:
        json.dump(d, f, indent=4)
    # img_list
    txt_path = (os.sep).join([sv_dir, 'img_list.txt'])
    with open(txt_path, 'a+') as f:
        f.write(name + '\n')
    # if cnt == 2:
    #     break