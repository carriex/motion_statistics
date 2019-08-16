
# Ubuntu environment set-up receipt 

ssh user@137.189.62.103

## Install Conda 

0. Download the [installer](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
1. Run the installer script 

```bash 
$ bash Anaconda-latest-Linux-x86_64.sh
```

2. Create and activate conda envrionment 
```bash 
$ conda create  --name <name>
$ conda activate <name> 
```

## Install PyTorch 

0. Follow instruction in [page](https://pytorch.org/get-started/locally/) with approriate system / Python Version 

1. Verify that Torch has been installed with GPU available


```bash
$ python 
>>> import torch 
>>> torch.cuda.current_device()
0
>>> torch.cuda.get_device_name(0)
'TITAN RTX'
>>> torch.cuda.is_available()
True  
```


## Install Vision related packages 

### Install OpenCV for Python 

Installing OpenCV via conda might result in failure in dependency resolution. Alternatively we can install via pip. [link](https://pydeeplearning.com/opencv/install-opencv3-with-anaconda-python3-6-on-ubuntu-18-04/)

```bash
$ pip install opencv-python  
```

### Install OpenCV for C++ 

Follow the instruction in this [link](https://gist.github.com/raulqf/a3caa97db3f8760af33266a1475d0e5e)
For CUDA 10, need to add below option 

```bash 
-D BUILD_opencv_cudacodec=OFF 
```


## Install Tensorflow with GPU

0. Create env and download tensorflow with GPU via conda 
```bash 
$ conda create name tf_gpu tensorflow-gpu 
```

1. Verify that tensorflow has been installed on the conda environment. The below command should print the GPU device detected.

```bash
$ activate tf_gpu 
$ python 

>>> import tensorflow as tf 
>>> sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

## Mount external disk on the filesystem 

0. Check the filesystem 
```bash
$ df 
```

1. Check the disk entry 
```bash
$ sudo fdisk -l 
```

2. Mount the disk to the mount point desired (e.g. mount /dev/sda on /data1)
```bash 
$ sudo mkdir /data1
$ sudo mount /dev/sda1 /data1 
$ cd /data1 
$ df -Th | grep sda 

/dev/sda 	ext4 	7.3T	2.5G 	6.0T	1% 		/data1

```

3. Setup auto mount at boot time 
```bash
$ sudo vi /etc/fstab
```

Add below line with information on the filesystem type
```
 /dev/sda /data1 ext3 defaults 0 0 

```

## Set-up SSH for LAN

1. Install openssh-server with apt 
```bash
$ sudo apt update 
$ sudo apt install openssh-server 
```

2. Once the installation finished, the server should start automatically. Check the status of the SSH service (notice that the port number will be shown here).
```bash
$ sudo systemctl status ssh 
```

3. Find out the IP Address of the server 
```bash 
$ ip a 
```

4. Connect to the server via SSH from another machine in the same LAN

```bash 
$ ssh user@<ipaddress>
```

*You might want to try telnet to the listening point on the target IP to make sure that the host is reachable from the client machine.*







