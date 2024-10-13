# [NeurIPS 2024] All-in-One Image Coding for Joint Human-Machine Vision with Multi-Path Aggregation


## Introduction
This repository is the offical PyTorch implementation of [All-in-One Image Coding for Joint Human-Machine Vision with Multi-Path Aggregation (NeurIPS 2024)](https://www.arxiv.org/abs/2409.19660). 

**Abstract:**
Image coding for multi-task applications, catering to both human perception and machine vision, has been extensively investigated. Existing methods often rely on multiple task-specific encoder-decoder pairs, leading to high overhead of parameter and bitrate usage, or face challenges in multi-objective optimization under a unified representation, failing to achieve both performance and efficiency. To this end, we propose Multi-Path Aggregation (MPA) integrated into existing coding models for joint human-machine vision, unifying the feature representation with an all-in-one architecture. MPA employs a predictor to allocate latent features among task-specific paths based on feature importance varied across tasks, maximizing the utility of shared features while preserving task-specific features for subsequent refinement. Leveraging feature correlations, we develop a two-stage optimization strategy to alleviate multi-task performance degradation. Upon the reuse of shared features, as low as 1.89% parameters are further augmented and fine-tuned for a specific task, which completely avoids extensive optimization of the entire model. Experimental results show that MPA achieves performance comparable to state-of-the-art methods in both task-specific and multi-objective optimization across human viewing and machine analysis tasks. Moreover, our all-in-one design supports seamless transitions between human- and machine-oriented reconstruction, enabling task-controllable interpretation without altering the unified model.


## TODO List
This repository is still under active construction:
- [x] Release training and testing codes
- [x] Release pretrained models
- [ ] Release visualization tools


## Preparation
The experiments were conducted on a single NVIDIA RTX 3090 with PyTorch 2.2.1, CUDA 11.8 and CuDNN8 (in the [docker environment](https://hub.docker.com/layers/pytorch/pytorch/2.2.1-cuda11.8-cudnn8-devel/images/sha256-5a0af47e17cb894f2654ee5bca6b88e795073bc72bd3d3890a843da4d1e04436?context=explore)). Create the environment, clone the project and then run the following code to complete the setup:
```bash
git clone https://github.com/NJUVISION/MPA.git
cd MPA
pip install -U pip && pip install -e .
```


## Pretrained Models
The trained weights in step 3 (with 4 tasks included in our paper) can be downloaded from [Google Drive](https://drive.google.com/file/d/1RdVCGfNVXYPhfP98ZsDRlCK4Ul29sVoq/view?usp=sharing) and [Baidu Drive (access code:by3d)](https://pan.baidu.com/s/1VJ0vddH8NMwFK2anOTTjrQ).


## Training
The training is completed by the following steps:

Step1: Run the script for variable-rate compression without GAN training pipeline:
```bash
python examples/train_stage1_wo_gan.py -m mpa_enc -d /path/to/dataset/ --epochs 400 -lr 1e-4 --batch_size 8 --cuda --save
```

Step2: Run the script for variable-rate compression with GAN training pipeline:
```bash
python examples/train_stage1_w_gan.py -m mpa_enc -d /path/to/dataset/ --epochs 400 -lr 1e-4 -lrd 1e-4 --batch_size 8 --cuda --save --pretrained /path/to/step1/checkpoint.pth.tar
```

Step3: Run the script for multi-task coding applications:
```bash
# for low distortion
python examples/train_stage2_mse.py -m mpa --task_idx 0 -d /path/to/dataset/ --epochs 200 -lr 1e-4 --batch_size 8 --cuda --save --pretrained /path/to/step2/checkpoint.pth.tar

# for classification
python examples/train_stage2_cls.py -m mpa --task_idx 1 -d /path/to/imagenet-1k/ --epochs 4 -lr 1e-4 --batch_size 8 --cuda --save --pretrained /path/to/step2/checkpoint.pth.tar

# for semantic segmentation
python examples/train_stage2_seg.py -m mpa --task_idx 2 -a psp -d /path/to/ade20k/ --epochs 200 -lr 1e-4 --batch_size 8 --cuda --save --pretrained /path/to/step2/checkpoint.pth.tar
```

The training checkpoints will be generated in the "checkpoints" folder at the current directory. You can change the default folder by modifying the function "init()" in "expample/train.py".

For semantic segmentation, please download the checkpoint of PSPNet from [the official repo](https://github.com/hszhao/semseg) first, and save it to `checkpoints/pspnet/pspnet_train_epoch_100.pth`.


## Testing
An example to evaluate R-D performance:
```bash
# high realism
python -m compressai.utils.eval_var_model checkpoint /path/to/dataset/ -a mpa -p ./path/to/step3/checkpoint.pth.tar --cuda --task_idx 0 --q_task 1 --save /path/to/save_dir/

# low distortion
python -m compressai.utils.eval_var_model checkpoint /path/to/dataset/ -a mpa -p ./path/to/step3/checkpoint.pth.tar --cuda --task_idx 0 --q_task 8 --save /path/to/save_dir/
```

An example to evaluate classification performance:
```bash
python examples/eval_cls_real_bpp.py -m mpa --task_idx 1 --cls_model convnext_tiny -d /path/to/imagenet-1k/ --test_batch_size 16 --cuda --save --pretrained ./path/to/step3/checkpoint.pth.tar --q_task 8 --real_bpp
```

An example to evaluate semantic segmentation performance:
```bash
python examples/eval_seg_real_bpp.py -m mpa --task_idx 2 -a psp -d /path/to/ade20k/ --test_batch_size 16 --cuda --save --pretrained ./path/to/step3/checkpoint.pth.tar --q_task 8 --real_bpp
```


## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{zhang2024allinone,
    author = {Zhang, Xu and Guo, Peiyao and Lu, Ming and Ma, Zhan},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {All-in-One Image Coding for Joint Human-Machine Vision with Multi-Path Aggregation},
    year = {2024}
}
```


## Acknowledgements
Our code is based on [TinyLIC](https://github.com/lumingzzz/TinyLIC), [CompressAI](https://github.com/InterDigitalInc/CompressAI), [NATTEN](https://github.com/SHI-Labs/NATTEN), [DynamicViT](https://github.com/raoyongming/DynamicViT), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and [PSPNet](https://github.com/hszhao/semseg). We would like to acknowledge the valuable contributions of the authors for their outstanding works and the availability of their open-source codes, which significantly benefited our work.
