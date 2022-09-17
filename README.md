# General Parking Slot Detection
 
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

