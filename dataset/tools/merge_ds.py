import os
import cv2
import json
import shutil


bp = r'/home/data/lwb/data/ParkingSlot'
# bp = r'D:\dataset\ParkingSlot\in99_raw'
if 0:
    for ds in ['in99_0512', 'in99_0614']:
        # for m in ['train', 'test']:
        for m in ['train']:
            b1 = rf'{bp}/{ds}/{m}'
            p1 = f'{b1}/img_list.txt'
            with open(p1, 'r') as f:
                fs = f.readlines()
            for name in fs:
                name = name.split('\n')[0]
                if len(name) == 0:
                    continue
                p_img = f'{b1}/image/{name}.png'
                p_lab = f'{b1}/label/{name}.json'
                with open(p_lab, 'r') as fj:
                    lab = json.load(fj)
                # h, w = lab['resolution']
                img = cv2.imread(p_img)
                h, w, c = img.shape
                # if h != 384:
                if 1:
                    lab['resolution'] = [384, 128]
                    if lab['datasetName'] == 'avm':
                        lab['datasetName'] = 'in99_0512'
                    # lab['cmPerPixel'] = round(lab['cmPerPixel']* h / 384, 2)
                    lab['cmPerPixel'] = round(880 / 384, 2)
                    for j in range(len(lab['points_x'])):
                        a = lab['points_x'][j]
                        a = [int(k/w*128) for k in a]
                        a[2] = -1
                        lab['points_x'][j] = a
                        a = lab['points_y'][j]
                        a = [int(k/h*384) for k in a]
                        a[2] = -1
                        lab['points_y'][j] = a
                    img = cv2.resize(img, (128, 384), interpolation=cv2.INTER_AREA)
                sv_bp = '/home/data/lwb/data/ParkingSlot/in99/train'
                # sv_bp = r'D:\dataset\ParkingSlot\padding\in99\train'
                os.makedirs(f'{sv_bp}/image', exist_ok=True)
                os.makedirs(f'{sv_bp}/label', exist_ok=True)
                sv_img = f'{sv_bp}/image/{name}.png'
                sv_lab = f'{sv_bp}/label/{name}.json'
                sv_txt = f'{sv_bp}/img_list.txt'
                cv2.imwrite(sv_img, img)
                with open(sv_lab, 'w+') as f:
                    json.dump(lab, f, indent=4)
                with open(sv_txt, 'a+') as f:
                    f.write(name + '\n')


if 1:
    bp = r'D:\dataset\ParkingSlot\padding'
    sv = r'D:\dataset\ParkingSlot\shaped'
    for ds in ['benchmark', 'in99']:
        p0 = rf'{bp}/{ds}/train'
        s0 = rf'{sv}/{ds}/train'
        os.makedirs(f'{s0}/image', exist_ok=True)
        os.makedirs(f'{s0}/label', exist_ok=True)
        for name in os.listdir(p0 + '/image'):
            name = name.split('.')[0]
            img = cv2.imread(f'{p0}/image/{name}.png')
            with open(f'{p0}/label/{name}.json', 'r') as f:
                lab = json.load(f)
            with open(f'{s0}/img_list.txt', 'a+') as f:
                f.write(name + '\n')
            img = img[32:352]
            img = cv2.resize(img, (128, 384), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f'{s0}/image/{name}.png', img)
            py_new = []
            for py in lab['points_y']:
                py = [int((e-32)/320*384) for e in py]
                py[2] = -1
                py_new.append(py)
            lab['points_y'] = py_new
            lab['resolution'] = [320, 128]
            lab['cmPerPixel'] = round(880 / 384, 2)
            lab['datasetName'] = 'in99_shaped'
            with open(f'{s0}/label/{name}.json', 'w') as f:
                json.dump(lab, f, indent=4)