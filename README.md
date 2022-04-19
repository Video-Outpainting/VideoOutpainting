## Results

#Vertical to Horizontal
- [Standard](https://tinyurl.com/VtHxStandard)
- [Gao etal.](https://tinyurl.com/VtHxFGVC)
- [Ours without image shifting](https://tinyurl.com/VtHxOurs)
- [Ours + image shifting](https://tinyurl.com/VtHxShift)
- [Ours + image shifting + post processing](https://tinyurl.com/VtHxPost)
    
#Landscape to Ultrawide
- [Standard](https://tinyurl.com/LtUxStandard)
- [Gao etal.](https://tinyurl.com/LtUxFGVC)
- [Ours without image shifting](https://tinyurl.com/LtUxOurs)
- [Ours + image shifting](https://tinyurl.com/LtUxShift)
- [Ours + image shifting + post processing](https://tinyurl.com/LtUxPost)
- 
## Prerequisites
- Tested on python 3.6.13, ubuntu 18.04 
- Anaconda

## Installation 

```
conda create -n VideoOutpainting python=3.6.13
conda activate VideoOutpainting
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
pip install tensorflow-gpu==1.15
pip install imageio
pip install scikit-image
pip install imageio-ffmpeg
```

- download weights for image completion network from https://github.com/zsyzzsoft/co-mod-gan 
    ./co-mod-gan-places2-050000.pkl

- download weight for RAFT (optical flow) from https://github.com/princeton-vl/RAFT
     ./raft-things.pth

- download weights for COSNet (VOS) from https://github.com/carrierlxk/COSNet
     ./co_attention.pth

## Usage


- Video outpainting on a single video:
```bash
cd tool
python video_outpaint.py --path ../frames/ --outroot ../results/frames/ --Width 0.125 --replace
```
replace: remove and recomplete 0.125*the width of the video on each side.

no replace: extrapolate 0.125*the width of the video width on each side

- Run dataset:
```bash
cd tool
python runDataset.py --pathToDataset /home/user/Documents/DAVIS-data/DAVIS/JPEGImages/480p/ --outroot ../result/ --vertical
```
vertical: Vertical to horizontal video conversion (0.33)

no vertical: horizontal to ultra-wide video conversion (0.125)
## Acknowledgments
- Our code is based upon [FGVC](https://github.com/vt-vl-lab/FGVC/).
