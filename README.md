# motion_statistics
PyTorch implementation of [Self-supervised Spatio-temporal Representation Learning for Videos by Predicting Motion and Appearance Statistics](https://arxiv.org/abs/1904.03597).


## Prerequisites 
1. Clone the repo 
```bash
$ git clone https://github.com/carriex/motion_statistics.git
```

2. Install the python dependency packages 
```bash
$ pip install -r requirements.txt 
```

3. Download the UCF dataset and extract the optical flow data

Resources: 
[Download UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/)

[Extract optical flow on GPU](https://github.com/wizyoung/Optical-Flow-GPU-Docker)

4. Update `constant.py` to include the filepath

## Run

### Train

First train with motion statistics. Note: This implementation only uses the motion statistics (Session 3.2) mentioned in the paper, not the Appearance statistics (Session 3.3). 

```bash
python train.py 
```

### Finetune

Then finetune on the label.

```bash
python fineune.py 
```

### Eval

Evaluate on the UCF-101 test data.

```bash
python eval.py 
```


