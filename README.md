# General Parking Slot Detection

## Description
代码很挫，以后再整理
草稿形式的整理，后面有时间了，再慢慢整理，，这里面，介绍怎么使用，也写得不清楚，各位稍稍摸索下吧，提问后，我空了再来回答。
search_paramter.py是用来 开多窗口，一次训练跑多个实验的脚本。它调用train.py
train.py里面也可以修改，用train.py单独跑单个实验或者重启中断的实验
evaluate.py则为单独用来跑测试的。
代码里面，有太多跟本实验无关的脚本，我后续再来整理。
实验做的过程中，没进行git管理，代码改动得太多，原来的实验的权重，可能跑不起来，但建议可以重新跑一下，很快的，参数配置里width设为0.125，depth=1/3，跑得超级快
，轻量级时学习率0.01，0.001，收敛得快；非轻量级时可设得更低为0.0001。
在[**fullimg branch**](https://github.com/huapohen/general_parking_slot_detection/tree/fullimage)分支里，提供了一个权重文件，改动得比较小。

在跑分时，我把限制条件放宽了，跑出来的AP很高，不放宽的话，设严谨了，则会低一些。毕竟不是打榜，也不发论文，感觉90+差不多就行了。
这个代码版本和实验记录。是一路调参过程中累计的。
毕竟这个问题到手（车位检测），只花了半月去琢磨，搞出了这个模型，后面大部分时间都是在跟 数据集较劲和工程C++上跟踪增删查改较劲，所以实验记录都是一点点加的，一开始没考虑那么多，属草稿形式。只能是空了再来整理。连benchmark和评价标准的代码，也是后面补的。先将就用吧。

超轻量级的（指flops 几十M这种），推理出来的效果有点差强人意，把flops整大点，效果会非常好。
至于超轻量级flops 几十这种。是通过PPLCNet的banckbone来改的，这个快，最快。这个backbone优点很多。
而默认用YOLOX的darknet，同级别效果要好点。
哦，对了，端到端，无后处理。要的就是这种！而且结构简单


link:

[自动泊车之通用停车位检测算法（上篇）](https://zhuanlan.zhihu.com/p/521821002)

[自动泊车之通用停车位检测算法（下篇）](https://zhuanlan.zhihu.com/p/522630354)

supplementary material: [videos](https://pan.baidu.com/s/1iTVvIJQWhV1nC8cbsBN2Yg)  gpsd 

experiments records: [200 exps](https://kdocs.cn/l/cnqdZU59SRuX)  

one experiment pretrain weight 部分实验的预训练权重：https://pan.baidu.com/s/1CdqcPhMfPQMat9m3i51-Vw     提取码：gpsd 


 ![image](https://github.com/huapohen/general_parking_slot_detection/blob/master/dataset/pairable/5.jpg)
 ![image](https://github.com/huapohen/general_parking_slot_detection/blob/master/dataset/pairable/4.jpg)
 ![image](https://github.com/huapohen/general_parking_slot_detection/blob/master/dataset/pairable/1.jpg)
 ![image](https://github.com/huapohen/general_parking_slot_detection/blob/master/dataset/pairable/2.jpg)
 ![image](https://github.com/huapohen/general_parking_slot_detection/blob/master/dataset/pairable/6.jpg)
 ![image](https://github.com/huapohen/general_parking_slot_detection/blob/master/dataset/pairable/7.jpg)
 ![image](https://github.com/huapohen/general_parking_slot_detection/blob/master/dataset/pairable/8.jpg)
 ![image](https://github.com/huapohen/general_parking_slot_detection/blob/master/dataset/pairable/9.jpg)
 ![image](https://github.com/huapohen/general_parking_slot_detection/blob/master/dataset/pairable/3.jpg)
 the mAP is calculated by setting the threshold confidence to 0.
 However, I do not open the strict limited condition. Later, I open the limited condition in dataset/process.py->match_marking_points(),  the mAP is a obvious **decline/descend**.  What's more, we can transform the major metric from mAP to Recall.
 
## Other branch
**master** is the subimage version, and the **fullimage** is the fullimage version
 
 
## Experiments Manager
search_hyperparams.py
run more than 100+ experiments through configuration, very convenient to manage.

 
## Dataset
baidu netdisk:

1. tongji dataset

[ps2.0](https://pan.baidu.com/s/1uJJjECNBKVYrqw9-w5HcWQ)  gpsd

[ps2.0_convert](https://pan.baidu.com/s/1ayADXI5jfd7oKB_NGVCZjg)  gpsd

2. seoul dateset

[PIL_park](https://pan.baidu.com/s/1rBz8aDP6mg2mmeq6QRpISQ)  gpsd

3. diy

[benchmark](https://pan.baidu.com/s/14o2jO5k4Epm4mF_gmsGkQw)  gpsd

All in here:

[gpsd_datasets](https://pan.baidu.com/s/1uIycqAEaQRBLrh2BVuBWYw)  gpsd
 
## Requirements
- numpy
- opencv
- torch

## Data pre-processing
By now, this project use this pre-processing, and the previous version do not use.
1. You need download the raw dataset 'PIL-park' and 'ps2.0_convert' from https://github.com/dohoseok/context-based-parking-slot-detect/
2. Please refer to the file in this project: './dataset/pairable/pairable_dataset_maker_v2.py'
3. Modify the data path in the file and enable the function.
4. python ./dataset/pairable/pairable_dataset_maker_v2.py


## Train
You should create an tmux windows called 'exp' (as same as 'session_name' you have setted in file 'search_hyperparams.py' )       
`tmux new -s session_name`       
And then,       
`python search_hyperparams.py`
1. All the experiments will run in the tmux windows. After training or debug, you should close the subwindows which created by search_hyperparams.py.        
2. The parameters setting in search_hyperparams.py will be saved in json file and the train.py as the actual runner read corresponding json file to start the training.      
3. Tensorboard log file saved under the 'experiments/xxxx'. The 'xxxx' is corresponding to the 'name' which setting in search_hyperparams.py in fuction 'experiment'.
4. 'experiments/params.json' is the template configure file, you could change the data path and other hyperparams in it. 

If you want to run train.py, you need modify some params in the train.py.

## Test
`python evaluate.py`


## DataSet Path
use this
1. /home/data/lwb/pairable_parking_slot/seoul
2. /home/data/lwb/pairable_parking_slot/tongji
\
or
1. /home/data/lwb/data/ParkingSlot/public
 (include seoul+tongji, merge train and test)
2. /home/data/lwb/data/ParkingSlot/in99

## TensorboardX
use this command to start TensorboardX service    
`tensorboard --logdir_spec exp_1:add_1,exp_2:add_2 --bind_all --port xxxx`

