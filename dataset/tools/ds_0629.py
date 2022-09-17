import os
import cv2
import sys
import json
import shutil
from tqdm import tqdm
from glob import glob
import numpy as np


if 0:
    b0 = r'/home/data/lwb/data/20220629'
    b1 = r'/home/data/lwb/data/20220629_fev'
    for dname in tqdm(os.listdir(b0)):
        dir1 = b0 + '/' + dname
        for fname in os.listdir(dir1):
            path1 = dir1 + '/' + fname
            path2 = path1.replace('0629/', '0629_fev/')
            if 'json' in fname:
                ind = path2.rfind('/')
                os.makedirs(path2[:ind], exist_ok=True)
                shutil.copy(path1, path2)
                continue
            path2 = path2.replace('.mp4', '_fev')
            cap = cv2.VideoCapture(path1)
            frames_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(path2, '\t\t', frames_len)
            os.makedirs(path2, exist_ok=True)
            for k in tqdm(range(frames_len)):
                ret, img = cap.read()
                if ret:
                    save_path = path2 + rf'/{k+1:05d}.jpg'
                    try:
                        cv2.imwrite(save_path, img)
                    except:
                        print(f'imwrite error: {save_path}')
                        print(f'frames_length: {frames_len}')
            cap.release()


if 0:
    b0 = '/home/qwe/data/20220629_bev'
    ds = sorted(os.listdir(b0))
    for dcnt in [10, 5]:
        bn = 'bev2' if dcnt == 10 else 'bev1'
        for k in range(dcnt):
            for d in tqdm(ds):
                b1 = b0 + '/' + d
                l = b1 + '/left_bev'
                r = b1 + '/right_bev'
                os.makedirs(l.replace('20220629_bev', f'{bn}/bev_v{k+1}'))
                os.makedirs(r.replace('20220629_bev', f'{bn}/bev_v{k+1}'))
                src = b1 + '/front.json'
                dst = b1.replace('20220629_bev', f'{bn}/bev_v{k+1}') + '/front.json'
                shutil.copy(src, dst)
                for e in [l, r]:
                    fs = sorted(os.listdir(e))
                    for i in range(0, len(fs), dcnt):
                        i += k
                        if i > len(fs)-1: break
                        src = e + '/' + fs[i]
                        dst = src.replace('20220629_bev', f'{bn}/bev_v{k+1}')
                        shutil.copy(src, dst)


if 0:
    b = r'D:\dataset\fev_bev'
    b1 = b + '/bev_v5'
    b2 = b + '/bev_v5_r'
    for n1 in tqdm(os.listdir(b1)):
        d1 = b1 + '/' + n1
        os.makedirs(d1.replace('v5', 'v5_r'), exist_ok=True)
        os.makedirs(d1.replace('v5', 'json'), exist_ok=True)
        for n2 in os.listdir(d1):
            d2 = d1 + '/' + n2
            if 'json' in d2:
                with open(d2, 'r') as f:
                    js1 = json.load(f)
                sensor = []
                for e in js1['sensor_info']:
                    e = [float(ele) for ele in e]
                    sensor.append(e)
                js1['sensor_info'] = sensor
                sv_js1 = d2.replace('v5', 'v5_r')
                sv_js2 = d2.replace('v5', 'json')
                with open(sv_js1, 'w') as f:
                    json.dump(js1, f, indent=4)
                with open(sv_js2, 'w') as f:
                    json.dump(js1, f, indent=4)
            else:
                # continue
                os.makedirs(d2.replace('v5', 'v5_r'), exist_ok=True)
                for n3 in os.listdir(d2):
                    ip = d2 + '/' + n3
                    img = cv2.imread(ip)
                    if 'left' in ip:
                        img = cv2.flip(img, -1)
                    svp = ip.replace('v5', 'v5_r')
                    pad = np.zeros([85, 350, 3], np.uint8)
                    img = np.concatenate([pad, img, pad], axis=0)
                    img = cv2.resize(img, (128, 384), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(svp, img)



