# AI Deinterlacing functions for Vapoursynth
A collection of three temporally aware deep learning deinterlacers.  
This will double the frame rate, for example from 30i to 60p.  

## Requirements
* pip install numpy
* pip install positional_encodings (optional, only for DeF)
* pip install -U openmim (optional, only for DfConv_EkSA)
* mim install mmcv (optional, only for DfConv_EkSA)
* [pytorch](https://pytorch.org/)

## Setup
Drop the entire "vs_deepdeinterlace" folder to where you typically load scripts from.

## Deinterlacers
| Deinterlacer | Quality | Speed     | Hardware | Paper                                                                     | Code 
| ------------ | ------- | --------- | ---------| ------------------------------------------------------------------------- | ----
| DfConvEkSA   | Higher  | ~5 fps    | RTX 4090 | [Link](https://arxiv.org/pdf/2404.13018)                                  | [Link](https://github.com/KUIS-AI-Tekalp-Research-Group/Video-Deinterlacing)
| DDD          | Lower   | ~40 fps   | RTX 4090 | [Link](https://studios.disneyresearch.com/2020/11/10/deep-deinterlacing/) | [Link](https://github.com/vincentvdschaft/Disney-Deep-Deinterlacing)
| DeF          | Lower   | ~3 fps    | RTX 4090 | [Link](https://link.springer.com/chapter/10.1007/978-981-99-8073-4_28)    | [Link](https://github.com/Anonymous2022-cv/DeT)

## Usage

    import vs_deepdeinterlace
    clip = vs_deepdeinterlace.DfConvEkSA(clip, tff=True, taa=False, device='cuda')
    clip = vs_deepdeinterlace.DDD(clip, tff=True, taa=False, device='cuda')
    clip = vs_deepdeinterlace.DeF(clip, tff=True, taa=False, device='cuda')

__*clip*__  
Interlaced clip. Not seperated into fields.

__*tff*__  
Top Field First if True. Bottom Field First if False.

__*taa*__  
Test-Time Augmentation. Increases quality a bit, but quadruples processing time.  
In addition to the normal deinterlacing, frames will be augmented by inverting, time reversing, and a combination of both. Then deinterlaced, augmentations reversed, and all 4 results averaged to get a more "true" result.

__*device*__  
Possible values are "cuda" to use with an Nvidia GPU, or "cpu". DDD is kind of usable on CPU, but DfConvEkSA and DeF are extremely slow.

## Tips
The deinterlacers work okay for detelecining, but fail to use all information from the correct field on fast motions.  

If you would like to finetune or improve the results, consider using one of these deinterlacers as "EdiExt"-clip in [QTGMC](https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/f11d79c98589c9dcb5b10beec35b631db68b495c/havsfunc/havsfunc.py#L1912).  

In my testing DeF seemed to perform similarly to DDD, but 10x slower. I have included it anyway in case it works better for someone else.
