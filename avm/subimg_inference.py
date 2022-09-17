import os
import cv2
import sys
import json
import time
import shutil
import platform
import datetime
import argparse
import torch
import torchvision
import numpy as np
import model.net as net
from tqdm import tqdm
from common import utils
from dataset.process import get_predicted_points, plot_slots
from loss.losses import deploy_preprocess
from avm.subimg_pipeline import AVM




def inference(k, params, image, model, avm, args):
    # avm处理，分辨率比例对齐，切图，左右子图
    aligned_info, img = avm.gengerate_single_input_subimg(params, image)
    with torch.no_grad():
        if params.in_dim == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=2)
        img = torchvision.transforms.ToTensor()(img)
        img = torch.stack([img])
        if params.cuda:
            img = img.cuda()
        f_h, f_w = params.feature_map_size
        # inference
        output = model(img)
        output = deploy_preprocess(output, params)
        pred_points = get_predicted_points(output[0], params)
        if 'raw_plot':
            eval_results = {}
            eval_results['pred_points'] = pred_points
            img = (img[0] * 255).permute(
                1, 2, 0).cpu().numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_plot = plot_slots(img, eval_results, params)
        if args.img_show:
            cv2.imshow('avm', img_plot)
            cv2.moveWindow("avm", 650, 300)
            key = cv2.waitKey(1)
            if key == 27:
                sys.exit()
        if args.save_plotted_img:
            save_path = os.path.join(avm.dst_frames_dir, f'{k:05d}.jpg')
            cv2.imencode('.jpg', img_plot)[1].tofile(save_path)

        return img_plot


def subimg_run(args):
    # read args
    args.confid_plot_inference = max(args.confid_thresh, 0.5)
    args.model_dir = os.path.join(os.getcwd(), 'experiments',
                                  'parking_slot_detection',
                                  f'exp_{args.exp_id}')
    args.restore_file = f'yolox_single_scale_model_latest.pth'
    if args.pth_type == 'best':
        args.restore_file = f'yolox_single_scale_test_model_best.pth'

    # merge params
    default_json_path = os.path.join('experiments', 'params.json')
    params = utils.Params(default_json_path)
    json_path = os.path.join(args.model_dir, 'params.json')
    model_params_dict = utils.Params(json_path).dict
    params.update(model_params_dict)

    # Update args into params
    params.update(vars(args))

    # Use GPU if available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_used
    params.cuda = torch.cuda.is_available()
    params.cuda = False

    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        gpu_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        model = net.fetch_net(params)


    # ------- inference -------
    avm = AVM(args)
    # cv2 video处理初始化
    avm.save_plotted_img = True
    avm.img_show = True
    capture = cv2.VideoCapture(avm.src_video_path)
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    output_path = avm.video_dir + rf'\{args.video_id}_pd.avi'
    W_f = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_f = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cnt_f = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (128, 384))
    # load model
    model = avm.load_checkpoints(params, model)
    model.eval()
    # k = 0
    # while True: # for视频流
    # for k in tqdm(range(cnt_f)):
    for k in range(cnt_f):
        success, img_src = capture.read()
        img_dst = inference(k+1, params, img_src, model, avm, args)
        writer.write(img_dst)
    capture.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.exp_id = 116
    args.gpu_used = '0'
    args.confid_thresh = 0.5
    args.pth_type = 'best'
    args.save_plotted_img = True
    args.img_show = True
    args.date = '20220419'
    args.video_id = '0002'
    args.avm_dir = r'D:\dataset\AVM\videos'

    subimg_run(args)