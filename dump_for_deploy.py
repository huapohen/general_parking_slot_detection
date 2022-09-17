"""dumps model and the inpur/output tensot and prepares for the deployment"""

import argparse
import logging
import os
import cv2
from tqdm import tqdm

import numpy as np
import torch
import torchvision.models as models

import dataset.data_loader as data_loader
import model.net as net

from common import utils
from common.manager import Manager
from dataset.process import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',
                    default='experiments/experiment_yolox_single_scale/exp_3',
                    help="Directory containing params.json")
parser.add_argument(
    '--restore_file',
    default='yolox_single_scale_model_latest.pth',  # yolox_single_scale_model_latest
    help="name of the file in --model_dir containing weights to load")
parser.add_argument('-gu',
                    '--gpu_used',
                    type=str,
                    default='0',
                    help='select the gpu for train or evaluation.')
parser.add_argument('-dp',
                    '--dump_path',
                    type=str,
                    default='0',
                    help='path containing the raw format input/output.')


def evaluate(manager):
    """
    Args:
        manager: a class instance that contains objects related to train and evaluate.
    """
    print("dump begin!")

    # loss status and eval status initial
    manager.model.eval()

    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch in manager.dataloaders[manager.params.eval_type]:

            # move to GPU if available
            data_batch = utils.tensor_gpu(data_batch)
            # compute model output
            image = data_batch["image"]
            output_batch = manager.model(image)
            # dump the input&output

            print("dump the input and output")
            b,c,h,w = image.shape
            with open(os.path.join(manager.params.dump_path, "in_tensor_{}_{}_{}_{}.raw".format(b,c,h,w)), 'wb') as fid:
                image.cpu().numpy().tofile(fid)
            b,c,h,w = output_batch.shape
            with open(os.path.join(manager.params.dump_path, "out_tensor_{}_{}_{}_{}.raw".format(b,c,h,w)), 'wb') as fid:
                output_batch.cpu().numpy().tofile(fid)

            print("dump the model")
            script_model = torch.jit.trace(manager.model, image)
            script_model.save(os.path.join(
                manager.params.dump_path, "slot_vertex_detect_v0.pt"))
            print("dump finish!")
            break


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # Only load model weights
    params.only_weights = True

    # Update args into params
    params.update(vars(args))

    # Use GPU if available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_used  # params.gpu_used
    params.cuda = False  # torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        model = torch.nn.DataParallel(model,
                                      device_ids=range(
                                          torch.cuda.device_count()))
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
    manager.load_checkpoints()

    # Test the model
    logger.info("Starting test")

    # check the dump path
    if manager.params.dump_path == "0":
        manager.params.dump_path = os.path.join(
            manager.params.model_dir, 'dump')
    if not os.path.exists(manager.params.dump_path):
        os.makedirs(manager.params.dump_path)

    # Evaluate
    evaluate(manager)
