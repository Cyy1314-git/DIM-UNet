# DIM-UNet
This is the official code repository for "DIM-UNet: Enhancing Medical Image Segmentation with Dual-Input Hybrid Attention and State Space Modeling". 

## Abstract
Medical image segmentation is crucial for diagnosis, treatment planning, and surgical preparation. However, traditional methods face challenges such as limited long-range dependency modeling by Convolutional Neural Networks (CNNs) and high computational complexity of Transformers. To address these issues, we propose DIM-UNet, a novel segmentation model that integrates State Space Models (SSMs) with CNNs. DIM-UNet employs a Dual Input Hybrid Attention (DIHA) module, which fuses features extracted by ResNet and Mamba encoders, deeply consolidating channel and spatial features. This approach effectively captures crucial information latent in both spatial and channel dimensions of medical images. Furthermore, we introduce a Log perturbation strategy based on category proportions to balance class distributions, enhancing model robustness and classification accuracy. Experimental results on the ISIC17, ISIC18, and Synapse datasets demonstrate that DIM-UNet consistently outperforms existing methods across multiple metrics, showcasing outstanding segmentation accuracy and computational efficiency.

## 0. Main Environments
```bash
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```
The .whl files of causal_conv1d and mamba_ssm could be found here. {[Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k)}

## 1. Prepare the dataset

### ISIC datasets
- The ISIC17 and ISIC18 datasets are split into a 7:3 ratio and can be downloaded from the following official links: ISIC2017 dataset: {[Official Link](https://challenge.isic-archive.com/data/#2017)}, ISIC2018 dataset: {[Official Link](https://challenge.isic-archive.com/data/#2018)}

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)
- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

### Synapse datasets
-- For the Synapse dataset, you could follow [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) to download the dataset

- After downloading the datasets, you are supposed to put them into './data/Synapse/', and the file format reference is as follows.

- './data/Synapse/'
  - lists
    - list_Synapse
      - all.lst
      - test_vol.txt
      - train.txt
  - test_vol_h5
    - casexxxx.npy.h5
  - train_npz
    - casexxxx_slicexxx.npz

## 2. Prepare the pre_trained weights
- The weights of the pre-trained VMamba could be downloaded [here](https://github.com/MzeroMiko/VMamba)  After that, the pre-trained weights should be stored in './pretrained_weights/'.

## 3. Train the VM-UNet
```bash
cd DIM-UNet
python train.py  # Train and test DIM-UNet on the ISIC17 or ISIC18 dataset.
python train_synapse.py  # Train and test DIM-UNet on the Synapse dataset.
```
## 4. Obtain the outputs
- After trianing, you could obtain the results in './results/'

## 5. Acknowledgments
- We thank the authors of VM-UNet and Swin-UNet for their open-source codes.
