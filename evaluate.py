# -*- coding: utf-8 -*-
"""Evaluates the model"""

import argparse
import logging
import platform
import os
import cv2
import sys
import json
import shutil
import datetime
from tqdm import tqdm

import math
from visdom import Visdom

import numpy as np
import torch
from torch.autograd import Variable

import dataset.data_loader as data_loader
import model.net as net

from common import utils
from common.manager import Manager
from loss.losses import *
from dataset.process import *

result_all_exps = {}

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',
                    default='experiments/parking_slot_detection/exp_3',
                    help="Directory containing params.json")
parser.add_argument(
    '--restore_file',
    # default='yolox_single_scale_test_model_best.pth',
    default='yolox_single_scale_model_latest.pth',
    help="name of the file in --model_dir containing weights to load")
parser.add_argument('-gu',
                    '--gpu_used',
                    type=str,
                    default='0',
                    help='select the gpu for train or evaluation.')
'''

cd /data/model_compile && ./pt2dlcqt.sh lwb deploy_20220718 "1,1,384,128"

'''

def deploy(manager, data_batch):
    if os.path.exists('deploy'):
        shutil.rmtree('deploy')
    os.makedirs('deploy', exist_ok=True)
    input = data_batch["image"]
    imgName = data_batch['imgName'][0]
    script_model = torch.jit.trace(manager.model, input)
    script_model.save("deploy/model.pt")
    output = manager.model(input)
    print(f' input.shape: {input.shape}')
    print(f'output.shape: {output.shape}')
    print(f'img_name: {imgName}')
    img_type = manager.params.img_type
    src_path = os.path.join(manager.params.data_dir,
                            'benchmark', 'test', 'image',
                            imgName + '.' + img_type)
    dst_path = f'deploy/{imgName}.{img_type}'
    src_img = cv2.imread(src_path)
    src_img = cv2.resize(src_img, manager.params.input_size[::-1],
                         cv2.INTER_AREA)
    cv2.imencode(f'.{img_type}', src_img)[1].tofile(f'deploy/{imgName}.{img_type}')
    print(f'src_img_path: {src_path}')
    print(f'dst_img_path: {dst_path}')
    fid = open(f'deploy/input.raw', 'wb')
    input.detach().cpu().numpy().astype(np.float32).tofile(fid)
    fid = open(f'deploy/output.raw', 'wb')
    output.detach().cpu().numpy().tofile(fid)
    f = open('deploy/img_info.txt', 'w')
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f'UTC {time_str}\n')
    f.write(f"exp_id: {manager.params.exp_id}" + '\n')
    f.write(f"exp_info: {manager.params.restore_file}" + '\n')
    f.write(f"img_path: {src_path}" + '\n')
    f.write(f"img_name: {imgName}" + '\n')
    f.write(f"resolution: {data_batch['resolution'][0]} -> " + \
            f"{manager.params.input_size}" + '\n')
    f.write(f"input_shape: {input.shape}" + '\n')
    f.write(f"output_shape: {output.shape}" + '\n')
    f.close()


    # deploy_id = 2
    # dst_path = f'/data/model_compile/lwb/deploy_{deploy_id}'
    # os.makedirs(dst_path, exist_ok=True)
    # shutil.rmtree(dst_path)
    # shutil.copytree('deploy', dst_path)

    # sys.exit()


def evaluate(manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    print("==============eval begin==============")

    # loss status and eval status initial
    manager.reset_loss_status()
    manager.reset_metric_status(manager.params.eval_type)
    manager.model.eval()

    with torch.no_grad():
        # compute metrics over the dataset
        reg_scores = []

        total_loss = []
        ground_truths_list = []
        predictions_list = []

        confidence_errs = []
        available_errs = []
        coord_dist_errs = []
        point_x_errs = []
        point_y_errs = []
        entryLine_coord_errs = []
        entryLine_radian_errs = []
        entryLine_cos_errs = []
        entryLine_sin_errs = []
        sepLine_radian_errs = []
        sepLine_cos_errs = []
        sepLine_sin_errs = []

        iter_max = len(manager.dataloaders[manager.params.eval_type])
        ds_stats = ''
        for ds in manager.params.val_data_ratio:
            if ds[1] == 1:
                ds_stats += ds[0] + ' '
        print(f'save_img: {manager.params.val_data_save}  ' + \
              f'source: {manager.params.pd_or_gt}   ' + \
              f'dataset: {ds_stats}')
        manager.logger.info(f'dataset: {ds_stats}')

        with tqdm(total=iter_max) as t:
            for i, data_batch in enumerate(manager.dataloaders[manager.params.eval_type]):
                # deploy
                if manager.params.is_BNC:
                    # if i < 3: continue
                    deploy(manager, data_batch)
                batch_size = len(data_batch["image"])
                # evaluate
                if manager.params.pd_or_gt == 'predicted':
                    if i == 0:
                        print("\n\n======== prt predicted results ========\n\n")
                    # move to GPU if available
                    data_batch = utils.tensor_gpu(data_batch, params_gpu=manager.params.cuda)
                    # compute model output
                    eval_results_batch = []
                    image_batch = data_batch["image"]
                    imgName_batch = data_batch["imgName"]
                    dataset_name_batch = data_batch["datasetName"]
                    output_batch = manager.model(image_batch)
                    output_batch = deploy_preprocess(output_batch, manager.params)
                    for j in range(batch_size):
                        # compute all metrics on this batch
                        eval_results = compute_eval_results(j, data_batch, output_batch,
                                                            manager.params)
                        eval_results_batch.append(eval_results)
                        # data parse
                        image = image_batch[j]
                        eval_results['imgName'] = imgName_batch[j]
                        dataset_name = dataset_name_batch[j]
                        eval_results['resolution'] = data_batch['resolution'][j]
                        marking_points = data_batch["MarkingPoint"][j]

                        total_loss.append(eval_results['loss'])
                        ground_truths_list.append(marking_points)
                        predictions_list.append(eval_results['pred_points'])

                        confidence_errs += eval_results['confidence_err']
                        available_errs += eval_results['available_err']
                        coord_dist_errs += eval_results['coord_dist_err']
                        point_x_errs += eval_results['point_x_err']
                        point_y_errs += eval_results['point_y_err']
                        entryLine_coord_errs += eval_results['entryLine_coord_err']
                        entryLine_radian_errs += eval_results['entryLine_radian_err']
                        entryLine_cos_errs += eval_results['entryLine_cos_err']
                        entryLine_sin_errs += eval_results['entryLine_sin_err']
                        sepLine_radian_errs += eval_results['sepLine_radian_err']
                        sepLine_cos_errs += eval_results['sepLine_cos_err']
                        sepLine_sin_errs += eval_results['sepLine_sin_err']
                else:
                    if i == 0:
                        print("\n\n======== prt ground_truth ========\n\n")
                    image_batch = []
                    imgName_batch = []
                    dataset_name_batch = []
                    eval_results_batch = []
                    for j in range(batch_size):
                        eval_results = []
                        for info in data_batch['MarkingPoint'][j]:
                            eval_results.append([1.0, info])
                        eval_results_batch.append(eval_results)
                        image_batch.append(data_batch["image"][j])
                        imgName_batch.append(data_batch["imgName"][j])
                        dataset_name_batch.append(data_batch["datasetName"][j])

                if manager.params.val_data_save:
                    for j in range(batch_size):
                        image = image_batch[j]
                        imgName = imgName_batch[j]
                        dataset_name = dataset_name_batch[j]
                        eval_results = eval_results_batch[j]
                        if not manager.params.is_BNC:
                            image = image * 255
                        image = image.permute(
                            1, 2, 0).cpu().numpy().astype(np.uint8)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        image = plot_slots(image, eval_results, manager.params, imgName)
                        eval_save_result(image, dataset_name, imgName, manager, i, j)

                t.update()

                if manager.params.is_BNC:
                    break

        total_loss = np.mean(total_loss) if len(total_loss) else 0.

        confidence_errs = np.mean(confidence_errs) if len(confidence_errs) else 0.
        available_errs = np.mean(available_errs) if len(available_errs) else 0.
        coord_dist_errs = np.mean(coord_dist_errs) if len(coord_dist_errs) else 0.
        point_x_errs = np.mean(point_x_errs) if len(point_x_errs) else 0.
        point_y_errs = np.mean(point_y_errs) if len(point_y_errs) else 0.
        entryLine_coord_errs = np.mean(entryLine_coord_errs) if len(entryLine_coord_errs) else 0.
        entryLine_radian_errs = np.mean(entryLine_radian_errs) if len(entryLine_radian_errs) else 0.
        entryLine_cos_errs = np.mean(entryLine_cos_errs) if len(entryLine_cos_errs) else 0.
        entryLine_sin_errs = np.mean(entryLine_sin_errs) if len(entryLine_sin_errs) else 0.
        sepLine_radian_errs = np.mean(sepLine_radian_errs) if len(sepLine_radian_errs) else 0.
        sepLine_cos_errs = np.mean(sepLine_cos_errs) if len(sepLine_cos_errs) else 0.
        sepLine_sin_errs = np.mean(sepLine_sin_errs) if len(sepLine_sin_errs) else 0.
        entryLine_angle_errs = entryLine_radian_errs / math.pi * 180
        sepLine_angle_errs = sepLine_radian_errs / math.pi * 180

        reg_scores_entry_angle = entryLine_radian_errs + entryLine_cos_errs + entryLine_sin_errs
        reg_scores_sep_angle = sepLine_radian_errs + sepLine_cos_errs + sepLine_sin_errs
        reg_scores_objpoint = coord_dist_errs + point_x_errs + point_y_errs

        reg_scores = reg_scores_entry_angle + reg_scores_sep_angle + reg_scores_objpoint \
                     + entryLine_coord_errs + confidence_errs + available_errs

        # 计算AP
        precisions, recalls, \
        TP, FP, TN, FN, \
        Precision, Recall, Accuracy, \
        TP_ava, FN_ava, Recall_ava, \
        TP_occ, FN_occ, Recall_occ \
            = calc_precision_recall(ground_truths_list, predictions_list, manager.params)

        average_precision = calc_average_precision(precisions,
                                                   recalls,
                                                   version="COCO")

        Metric = {
            "total_loss": total_loss,
            "average_precision": average_precision,

            'Precision': Precision, 'Recall': Recall, 'Accuracy': Accuracy, 
            'Recall_ava': Recall_ava, 'Recall_occ': Recall_occ,
            'inference_threshold': manager.params.confid_plot_inference,
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'Slots=TP+FN': TP+FN,
            'TP_ava': TP_ava, 'FN_ava': FN_ava, 'ava_cnt': TP_ava+FN_ava,
            'TP_occ': TP_occ, 'FN_occ': FN_occ, 'occ_cnt': TP_occ+FN_occ,
            
            'confidence_errs': confidence_errs,
            'available_errs': available_errs,
            'coord_dist_errs': coord_dist_errs,
            'point_x_errs': point_x_errs,
            'point_y_errs': point_y_errs,
            'entryLine_coord_errs': entryLine_coord_errs,
            'entryLine_radian_errs': entryLine_radian_errs,
            'entryLine_angle_errs': entryLine_angle_errs,
            'entryLine_cos_errs': entryLine_cos_errs,
            'entryLine_sin_errs': entryLine_sin_errs,
            'sepLine_radian_errs': sepLine_radian_errs,
            'sepLine_angle_errs': sepLine_angle_errs,
            'sepLine_cos_errs': sepLine_cos_errs,
            'sepLine_sin_errs': sepLine_sin_errs,

            'reg_scores_entry_angle': reg_scores_entry_angle,
            'reg_scores_sep_angle': reg_scores_sep_angle,
            'reg_scores_objpoint': reg_scores_objpoint,
            'reg_scores': reg_scores
        }

        # result_all_exps[manager.params.exp_id] = Metric

        manager.update_metric_status(metrics=Metric,
                                     split=manager.params.eval_type,
                                     batch_size=manager.params.eval_batch_size)

        # update data to logger
        manager.logger.info(
            'Loss/valid epoch_{} {}: ' \
            \
            'total_loss: {:.6f} | ' \
            'average_precision: {:.6f} | ' \
            \
            'Precision: {:.4f}, Recall: {:.4f}, Accuracy: {:.4f} | ' \
            'Recall_ava: {:.4f}, Recall_occ: {:.4f} | ' \
            'inference_threshold: {} | ' \
            'TP={}, FP={}, TN={}, FN={}, Slots=TP+FN={} | ' \
            'TP_ava={}, FN_ava={}, ava_cnt={} | ' \
            'TP_occ={}, FN_occ={}, occ_cnt={} | ' \
            \
            'confidence_errs: {:.6f} | ' \
            'available_errs: {:.6f} | ' \
            'coord_dist_errs: {:.6f} | ' \
            'point_x_errs: {:.6f} | ' \
            'point_y_errs: {:.6f} | ' \
            'entryLine_coord_errs: {:.6f} | ' \
            'entryLine_radian_errs: {:.6f} | ' \
            'entryLine_angle_errs: {:.1f} | ' \
            'entryLine_cos_errs: {:.6f} | ' \
            'entryLine_sin_errs: {:.6f} | ' \
            'sepLine_radian_errs: {:.6f} | ' \
            'sepLine_angle_errs: {:.1f} | ' \
            'sepLine_cos_errs: {:.6f} | ' \
            'sepLine_sin_errs: {:.6f} | ' \
            \
            'reg_scores_entry_angle: {:.6f} | ' \
            'reg_scores_sep_angle: {:.6f} | ' \
            'reg_scores_objpoint: {:.6f} | ' \
            'reg_scores: {:.6f} '
            .format(manager.params.eval_type, manager.epoch_val,

                    total_loss,
                    average_precision,
                    Precision, Recall, Accuracy,
                    Recall_ava, Recall_occ,
                    manager.params.confid_plot_inference,
                    TP, FP, TN, FN, TP+FN,
                    TP_ava, FN_ava, TP_ava+FN_ava,
                    TP_occ, FN_occ, TP_occ+FN_occ,

                    confidence_errs,
                    available_errs,
                    coord_dist_errs,
                    point_x_errs,
                    point_y_errs,
                    entryLine_coord_errs,
                    entryLine_radian_errs,
                    entryLine_angle_errs,
                    entryLine_cos_errs,
                    entryLine_sin_errs,
                    sepLine_radian_errs,
                    sepLine_angle_errs,
                    sepLine_cos_errs,
                    sepLine_sin_errs,

                    reg_scores_entry_angle,
                    reg_scores_sep_angle,
                    reg_scores_objpoint,
                    reg_scores
                    )
        )

        # For each epoch, print the metric
        manager.print_metrics(manager.params.eval_type,
                              title=manager.params.eval_type,
                              color="green")

        # manager.epoch_val += 1
        manager.model.train()


def eval_save_result(save_file, dataset_name, img_name, manager, iter, idx_iter):
    try:
        if len(manager.params.save_img_dir.split(os.sep)) > 1:
            save_img_dir = manager.params.save_img_dir
        else:
            save_img_dir = os.path.join(manager.params.model_dir, 'image')
    except:
        save_img_dir = os.path.join(manager.params.model_dir, 'image')

    suff = 'pd' if manager.params.pd_or_gt == 'predicted' else 'gt'
    save_img_dir_epoch = os.path.join(save_img_dir, suff)
    if manager.epoch_val > 0:
        save_img_dir_epoch = os.path.join(save_img_dir, str(manager.epoch_val))
    if iter == 0 and idx_iter == 0:
        print(f'=====   save_img_dir: {save_img_dir_epoch}   =====\n')
        if os.path.exists(save_img_dir_epoch):
            shutil.rmtree(save_img_dir_epoch)

    save_dir_epoch_ds = os.path.join(save_img_dir_epoch, dataset_name)
    os.makedirs(save_dir_epoch_ds, exist_ok=True)
    save_path = os.path.join(save_dir_epoch_ds, img_name + '.jpg')
    cv2.imencode('.jpg', save_file)[1].tofile(save_path)  # 可以保存中文路径
    if manager.params.is_BNC:
        deploy_img_path = f'deploy/output_{img_name}.jpg'
        cv2.imencode('.jpg', save_file)[1].tofile(deploy_img_path)

    if type(save_file) == str:  # save string information
        f = open(os.path.join(save_img_dir_epoch, img_name), 'w')
        f.write(save_file)
        f.close()



def run_all_exps(eid, threshold, val_data_save):
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()

    args.data_dir = f'/home/data/lwb/pairable_parking_slot'
    args.data_dir = f'/home/data/lwb/data/ParkingSlot'
    if platform.system() == 'Windows': # Linux
        args.data_dir = r'D:\dataset\pairable_parking_slot'
        args.data_dir = r'D:\dataset\ParkingSlot\padding'
        args.num_workers_eval = 1
    base_path = os.path.join(os.getcwd(), 'experiments',
                             'parking_slot_detection')
    args.exp_id = eid
    args.model_dir = os.path.join(base_path, f'exp_{args.exp_id}')
    args.restore_file = 'yolox_single_scale_test_model_best.pth'
    # args.restore_file = 'yolox_single_scale_model_latest.pth'
    args.eval_batch_size = 64
    args.gpu_used = '4'
    # args.val_data_ratio = [["seoul", 0], ["tongji", 1]]
    # args.val_data_ratio = [["seoul", 1], ["tongji", 0]]
    # args.val_data_ratio = [["seoul", 0], ["tongji", 0], ["avm", 1]]
    # args.val_data_ratio = [["in99_0614", 1]]
    # args.val_data_ratio = [["ts_label", 1]]
    # args.val_data_ratio = [["add_in99", 1]]
    # args.val_data_ratio = [["benchmark", 1]]
    # args.val_data_ratio = [["tongji", 1]]
    # args.val_data_ratio = [["seoul", 1]]
    # args.val_data_ratio = [["in99", 1]]
    args.val_data_ratio = [["fullimg", 1]]
    args.squared_distance_thresh = 0.000277778
    # args.val_data_ratio = [["avm", 1]]
    args.dataset_type = 'test'
    # args.val_data_save = True
    args.val_data_save = val_data_save
    # args.pd_or_gt = 'ground truth'
    # args.save_img_dir = r'/home/liwenbo/avm_result'
    # args.confid_thresh = 0 # AP最高
    args.confid_plot_inference = 0.5   # 推理画图过滤用
    args.confid_plot_inference = threshold
    args.occupied_mode = False
    args.direction_rectangle = True
    # args.direction_rectangle = False
    # args.is_BNC = True


    # merge params
    default_json_path = os.path.join('experiments', 'params.json')
    params = utils.Params(default_json_path)
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), \
        "No json configuration file found at {}".format(json_path)
    model_params_dict = utils.Params(json_path).dict
    params.update(model_params_dict)

    # Only load model weights
    params.only_weights = True

    # Update args into params
    params.update(vars(args))
    # if deploy
    if params.is_BNC:
        args.eval_batch_size = 1
        args.num_workers_eval = 1
        args.cuda = False
        args.img_type = 'png'
        params.update(vars(args))

    # Use GPU if available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_used
    params.cuda = torch.cuda.is_available() \
        if not params.is_BNC else False

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # flop, parameters
    import thop
    input = torch.randn(1, params.in_dim, *params.input_size)
    model = net.fetch_net(params)
    flops, parameters = thop.profile(model, inputs=(input,), verbose=False)
    model.eval()
    output = model(input)
    split_line = '=' * 30
    prt_model_info = f'''
                {split_line}
                Input  shape: {tuple(input.shape[1:])}
                Output shape: {tuple(output.shape[1:])}
                Flops: {flops / 1e6:.1f} M
                Params: {parameters / 1e6:.1f} M
                {split_line}'''
    logger.info(prt_model_info)
    logger.info(f"exp_id: {params.exp_id}")

    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids)
    else:
        model = net.fetch_net(params)

    # Initial status for checkpoint manager
    manager = Manager(model=model,
                      optimizer=None,
                      scheduler=None,
                      params=params,
                      dataloaders=dataloaders,
                      writer=None,
                      logger=logger)

    # Reload weights from the saved file
    try:
        manager.load_checkpoints()
    except:
        return

    # Test the model
    logger.info("Starting test")

    # Evaluate
    evaluate(manager)
    
    # ffmpeg -r  5 -i ps2.0/%04d.jpg -vcodec mpeg4 -b:v 6000k fps_5.avi


if __name__ == '__main__':
    threshold = 0.5
    # val_data_save = False
    val_data_save = True
    # for i in range(197,200):
    for i in range(1):
        # run_all_exps(i, threshold, val_data_save)
        # run_all_exps(198, threshold, val_data_save)
        run_all_exps(202, threshold, val_data_save)
        # run_all_exps(73, threshold, val_data_save)
        # run_all_exps(173, threshold, val_data_save)
        # run_all_exps(184, threshold, val_data_save)

    with open("experiments/result_all_exps_v2.json", "w") as f:
        f.write(json.dumps(result_all_exps, indent=4))
