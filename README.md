# Dual pyramid network for salient object detection

by Xuemiao Xu^, Jiaxing Chen^, Huaidong Zhang\*, and Guoqiang Han\* (^ joint 1st author, * joint corresponding author)[[paper link](https://www.sciencedirect.com/science/article/pii/S0925231219313451?via%3Dihub)]

This implementation is written by Jiaxing Chen at the South China University of Technology.

## Citation

@article{xu2020dual, &nbsp;

​	title={Dual pyramid network for salient object detection},&nbsp;

​	author={Xu, Xuemiao and Chen, Jiaxing and Zhang, Huaidong and Han, Guoqiang},&nbsp;

​	journal={Neurocomputing},&nbsp;

​	volume={375},&nbsp;

​	pages={113--123},&nbsp;

​	year={2020},&nbsp;

​	publisher={Elsevier}&nbsp;
}

## Saliency Map

The results of DPNet on six RGB saliency datasets (ECSSD, HKU-IS, PASCAL-S, SOD, DUT-OMRON, DUTS-TE) and three RGB-D saliency  datasets (NLPR, NJUD, STEREO) can be found at [Google Drive]().

## Trained Model

You can download the trained model which is reported in our paper at  [Google Drive]().

## Requirement

- Python 2.7
- PyTorch 0.4.0
- torchvision
- numpy
- Cython
- pydensecrf ([here](https://github.com/Andrew-Qibin/dss_crf) to install)

## Training

1. Set the path of pretrained resnet model in resnet/config.py
2. Set the path of DUTS-TR dataset in config.py
3. Run by `python train.py`

*Hyper-parameters* of training were gathered at the beginning of *train.py* and you can conveniently change them as you need.

## Testing

1. Set the path of six benchmark datasets in config.py
2. Put the trained model in ckpt/dpnet
3. Run by `python infer.py`

*Settings* of testing were gathered at the beginning of *infer.py* and you can conveniently change them as you need.

## Dataset links

- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://sites.google.com/site/ligb86/hkuis), [PASCAL-S](http://cbi.gatech.edu/salobj/), [SOD](http://elderlab.yorku.ca/SOD/), [DUT-OMRON](http://ice.dlut.edu.cn/lu/DUT-OMRON/Homepage.htm), [DUTS](http://saliencydetection.net/duts/) : the six benchmark datasets