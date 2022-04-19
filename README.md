## Results

### Vertical to Horizontal
- [Standard Completions](https://www.youtube.com/playlist?list=PLh5XtfDDhGgX1tj4yuPEU-BQUxZnj7KRv)
- [Gao etal.](https://www.youtube.com/playlist?list=PLh5XtfDDhGgVNLC1TfJTBdqYgYshIuyZi)
- [Ours without image shifting](https://www.youtube.com/playlist?list=PLh5XtfDDhGgXDxdyNL4jg8-SpzWCKgEQN)
- [Ours with image shifting](https://www.youtube.com/playlist?list=PLh5XtfDDhGgUsCCemgB8c6lejqQMIdqmf)
- [Ours with both image shifting and post processing](https://www.youtube.com/playlist?list=PLh5XtfDDhGgVonQ0PifxQfr_EG0UpNbul)
    
### Landscape to Ultrawide
- [Standard Completions](https://www.youtube.com/playlist?list=PLh5XtfDDhGgWCs_SpBV6lpcXBoYUmUtrV)
- [Gao etal.](https://www.youtube.com/playlist?list=PLh5XtfDDhGgWF627x6FrpdWzorSZd7yhm)
- [Ours without image shifting](https://www.youtube.com/playlist?list=PLh5XtfDDhGgXgHrKEPTNTpg2dWi8DnB5B)
- [Ours with image shifting](https://www.youtube.com/playlist?list=PLh5XtfDDhGgU8aHSfddl3oXSOwz6yMsB-)
- [Ours with image shifting and post processing](https://www.youtube.com/playlist?list=PLh5XtfDDhGgWBokNQ2bJCV8UK7NeJ7OqB)

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
