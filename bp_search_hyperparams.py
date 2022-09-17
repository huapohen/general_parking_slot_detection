"""Peform hyperparemeters search"""

import argparse
import collections
import itertools
import os
import sys
import shutil

from common import utils
from experiment_dispatcher import dispatcher, tmux

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir',
                    default='experiments',
                    help='Directory containing params.json')
parser.add_argument('--id', default=1, type=int, help="Experiment id")


def launch_training_job(exp_dir,
                        exp_name,
                        session_name,
                        param_pool_dict,
                        device_used,
                        params,
                        start_id=0):
    # Partition tmux windows automatically
    tmux_ops = tmux.TmuxOps()
    # Combining hyper-parameters and experiment ID automatically
    task_manager = dispatcher.Enumerate_params_dict(task_thread=0,
                                                    if_single_id_task=True,
                                                    **param_pool_dict)

    num_jobs = len([v for v in itertools.product(*param_pool_dict.values())])
    num_device = len(device_used)
    exp_cmds = []

    for job_id in range(num_jobs):

        device_id = device_used[job_id % num_device]
        hyper_params = task_manager.get_thread(ind=job_id)[0]

        job_name = 'exp_{}'.format(job_id + start_id)
        for k in hyper_params.keys():
            params.dict[k] = hyper_params[k]

        params.dict['model_dir'] = os.path.join(exp_dir, exp_name, job_name)
        model_dir = params.dict['model_dir']

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Write parameters in json file
        json_path = os.path.join(model_dir, 'params.json')
        params.save(json_path)

        # Launch training with this config
        cmd = 'python train.py ' \
              '--gpu_used {} ' \
              '--model_dir {} ' \
              '--exp_name {} ' \
              '--tb_path {}'.format(
            device_id, model_dir, exp_name + "_" + job_name,
            os.path.join(exp_dir, exp_name, "tf_log/"))
        exp_cmds.append(cmd)

        if_serial = num_jobs > num_device
        if if_serial:
            print("run task serially! ")
        else:
            print("run task parallelly! ")

    tmux_ops.run_task(exp_cmds,
                      task_name=exp_name,
                      session_name=session_name,
                      if_serial=if_serial)


def experiment():
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    if args.id == 1:
        exp_name = "parking_slot_detection"
        start_id = 141
        session_name = f'exp{start_id}'  # tmux session name, need pre-create
        param_pool_dict = collections.OrderedDict()
        device_used = collections.OrderedDict()
        param_pool_dict['net_type'] = ['yolox_single_scale']
        # param_pool_dict['upsample_type'] = ['bilinear']
        n_gpus = 1
        param_pool_dict['train_batch_size'] = [64*n_gpus]
        param_pool_dict['eval_batch_size'] = [64*n_gpus]
        param_pool_dict['num_workers'] = [4*n_gpus]
        param_pool_dict['in_dim'] = [1] # default 1: gray
        # param_pool_dict['yolo_depth'] = [1]
        # param_pool_dict['yolo_width'] = [1/8]
        # param_pool_dict['head_width'] = [0.5]

        yolo_width = 0.125
        yolo_depth = 1/3
        # param_pool_dict['dwconv'] = [True]
        # device_used = ['3', '3', '3', '3']
        device_used = ['3', '3']
        param_pool_dict["fix_expand"] = [True]
        param_pool_dict["save_every_epoch"] = [False]
        # ########################################################
        # exp 127 128 129 130 depth 变量 depth=1/3 or 1
        # 和 数据集ratio 变量 ratio=1 or 0.5
        param_pool_dict['yolo_depth'] = [1]
        # param_pool_dict['yolo_depth'] = [yolo_depth]
        # param_pool_dict['yolo_depth'] = [yolo_depth, 1]
        # exp 131 132  yolo_width = 0.25
        yolo_width = 0.25
        # exp 133 134  yolo_width = 0.25  depth=1
        # param_pool_dict['yolo_depth'] = [1]
        param_pool_dict['yolo_width'] = [yolo_width]
        param_pool_dict['head_width'] = [yolo_width]
        # param_pool_dict["occupied_mode"] = [False]
        # exp 135 136 137 138 / 139 140 / 141 142
        param_pool_dict["occupied_mode"] = [True]
        param_pool_dict['num_workers'] = [2 * n_gpus]
        # param_pool_dict['learning_rate'] = [0.1, 0.01, 0.001, 0.0001]
        # param_pool_dict['learning_rate'] = [0.001]
        param_pool_dict['learning_rate'] = [0.01]

        param_pool_dict['model_type'] = ['dark']
        # param_pool_dict['lightnet_with_act'] = [True, False]
        param_pool_dict['upsample_type'] = ['bilinear']
        # param_pool_dict['dwconv'] = [True]
        # param_pool_dict['expand_ratio'] = [0.5]
        param_pool_dict['new_flip_h_ratio'] = [0.5]
        param_pool_dict['shift_w_pixel'] = [1/3/1]
        param_pool_dict['shift_h_pixel'] = [1/12*3] # 1/12/2 default
        param_pool_dict['shift_w_ratio'] = [0.75]
        param_pool_dict['shift_h_ratio'] = [0.75]
        param_pool_dict['shift_w_left_ratio'] = [2/3]
        param_pool_dict['shift_before_rotate_ratio'] = [0.5] # default
        param_pool_dict["angle_range"] = [45] # key super-parameter
        param_pool_dict['filter_outer_point_B'] = [True]
        # 切换SepLine的deltanorm预测和direction预测
        # param_pool_dict['deltanorm_or_direction'] = ['deltanorm']
        param_pool_dict['deltanorm_or_direction'] = ['direction']
        # param_pool_dict['deltanorm_or_direction'] = ['deltanorm', 'direction']
        param_pool_dict['loss_l2_type'] = ['multi_v2']
        # param_pool_dict['loss_l2_type'] = [''] # default
        param_pool_dict["squared_distance_thresh"] = [0.005]
        # param_pool_dict["train_data_ratio"] = [[["seoul", 1], ["tongji", 1], ["avm", 1]]]
        # param_pool_dict["train_data_ratio"] = [[["seoul", 1], ["tongji", 1], ["avm", 1], ["in99_0614", 1]]]
        param_pool_dict["train_data_ratio"] = [[["seoul", 1], ["tongji", 1], ["avm", 1], ["in99_0614", 1]],
                                                [["seoul", 1], ["tongji", 1], ["avm", 0.5], ["in99_0614", 1]]]
        # param_pool_dict["train_data_ratio"] = [[["avm", 1]]]
        # param_pool_dict["val_data_ratio"] = [[["seoul", 1], ["tongji", 0]],
        #                                      [["seoul", 0], ["tongji", 1]]]
        # param_pool_dict["val_data_ratio"] = [[["seoul", 1], ["tongji", 0]]]
        # param_pool_dict["val_data_ratio"] = [[["seoul", 0], ["tongji", 1]]]
        param_pool_dict["val_data_ratio"] = [[["in99_0614", 1]]]
        param_pool_dict["save_every_epoch"] = [False]
        param_pool_dict["eval_freq"] = [10]
        # '0', '1', '2', '3', '4', '5', '6', '7'
        # device_used = ['6']
        # device_used = ['0', '0']
        # device_used = ['0_1_2_3'] # for one experiment
        # device_used = ['0', '1', '2', '3'] # for four experiments

        for id in range(start_id, start_id + len(device_used)):
            try: # cover exp
                bp = os.getcwd() + f'/experiments/{exp_name}'
                shutil.rmtree(f"{bp}/exp_{id}")
                shutil.rmtree(f"{bp}/tf_log/{exp_name}_exp_{id}")
            except:
                pass

    else:
        raise NotImplementedError

    launch_training_job(args.parent_dir, exp_name, session_name,
                        param_pool_dict, device_used, params, start_id)


if __name__ == "__main__":
    experiment()
