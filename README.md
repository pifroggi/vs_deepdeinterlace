# AI Deinterlacing functions for Vapoursynth
A collection of three temporally aware deep learning deinterlacers.  
This will double the frame rate, for example from 30i to 60p.  

| Deinterlacer | Quality | Speed     | Test Resolution | Hardware | Paper                                                                     | Code 
| ------------ | ------- | --------- | ---------- | -------- | ------------------------------------------------------------------------- | ----
| DfConvEkSA   | Higher  | ~5 fps    | 720x480    | RTX 4090 | [Link](https://arxiv.org/pdf/2404.13018)                                  | [Link](https://github.com/KUIS-AI-Tekalp-Research-Group/Video-Deinterlacing)
| DeF          | Lower   | ~6 fps    | 720x480    | RTX 4090 | [Link](https://link.springer.com/chapter/10.1007/978-981-99-8073-4_28)    | [Link](https://github.com/Anonymous2022-cv/DeT)
| DDD          | Lower   | ~50 fps   | 720x480    | RTX 4090 | [Link](https://studios.disneyresearch.com/2020/11/10/deep-deinterlacing/) | [Link](https://github.com/vincentvdschaft/Disney-Deep-Deinterlacing)

## Requirements
* [pytorch](https://pytorch.org/)
* pip install numpy
* pip install positional_encodings (optional, only for DeF)
* pip install -U openmim && mim install mmcv (optional, only for DfConv_EkSA)


## Setup
Drop the entire "vs_deepdeinterlace" folder to where you typically load scripts from.

## Usage

    import vs_deepdeinterlace
    
    clip = vs_deepdeinterlace.DfConvEkSA(clip, tff=True, tta=False, device="cuda")
    
    clip = vs_deepdeinterlace.DeF(clip, tff=True, tta=False, device="cuda", fp16=True)

    clip = vs_deepdeinterlace.DDD(clip, tff=True, tta=False, device="cuda", fp16=True)

  
__*`clip`*__  
Interlaced clip, not seperated into fields. Must be in RGBS format.

__*`tff`*__  
Top Field First if True. Bottom Field First if False.

__*`tta`* (optional)__  
Test-Time Augmentation. Increases quality a bit, but quadruples processing time.  
In addition to the normal deinterlacing, frames will be augmented by inverting, time reversing, and a combination of both. Then deinterlaced, augmentations reversed, and all 4 results averaged to get a more "true" result.

__*`device`* (optional)__  
Possible values are "cuda" to use with an Nvidia GPU, or "cpu". DDD is usable on CPU, but DfConvEkSA and DeF are extremely slow.

__*`fp16`* (optional)__  
Up to doubles processing speed and halves VRAM usage. Strongly recommended if your GPU supports it. Does not work on CPU. DfConvEkSA does not support fp16 mode.


## Tips
* If you would like to finetune or improve the results, consider using one of these deinterlacers as "EdiExt"-clip in [QTGMC](https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/f11d79c98589c9dcb5b10beec35b631db68b495c/havsfunc/havsfunc.py#L1912).
* The deinterlacers work okay for animation, but fail to use all information from the correct field on fast motions. May still be useful for orphaned fields.
* In my testing DeF seemed to perform similarly to DDD, but 10x slower. I have included it anyway in case it works better for someone else.
