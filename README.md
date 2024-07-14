# AI Deinterlacing functions for Vapoursynth
A collection of three temporally aware deep learning deinterlacers.  
This will double the frame rate, for example from 30i to 60p.  

| Deinterlacer | Quality | Speed     | Test Resolution | Hardware | Paper                                                                     | Code 
| ------------ | ------- | --------- | --------------- | -------- | ------------------------------------------------------------------------- | ----
| DDD          | Lower   | ~50 fps   | 720x480         | RTX 4090 | [Link](https://studios.disneyresearch.com/2020/11/10/deep-deinterlacing/) | [Link](https://github.com/vincentvdschaft/Disney-Deep-Deinterlacing)
| DeF          | Middle  | ~6 fps    | 720x480         | RTX 4090 | [Link](https://link.springer.com/chapter/10.1007/978-981-99-8073-4_28)    | [Link](https://github.com/Anonymous2022-cv/DeT)
| DfConvEkSA   | Higher  | ~5 fps    | 720x480         | RTX 4090 | [Link](https://arxiv.org/pdf/2404.13018)                                  | [Link](https://github.com/KUIS-AI-Tekalp-Research-Group/Video-Deinterlacing)

## Requirements
* [pytorch](https://pytorch.org/)
* pip install numpy
* pip install positional_encodings (optional, only for DeF)
* pip install -U openmim && mim install "mmcv>=2.0.0" (optional, only for DfConvEkSA)

## Setup
Put the entire "vs_deepdeinterlace" folder into your scripts folder, or where you typically load scripts from."

## Usage

    import vs_deepdeinterlace

    clip = vs_deepdeinterlace.DDD(clip, tff=True, tta=False, device="cuda", fp16=True)

    clip = vs_deepdeinterlace.DeF(clip, tff=True, tta=False, device="cuda", fp16=True)

    clip = vs_deepdeinterlace.DfConvEkSA(clip, tff=True, tta=False, mocomp=False, device="cuda", fp16=False)

  
__*`clip`*__  
Interlaced clip, not seperated into fields. Must be in RGBS format.

__*`tff`*__  
Top Field First if True. Bottom Field First if False.

__*`tta`* (optional)__  
Test-Time Augmentation. Increases quality a bit, but quadruples processing time.  
In addition to the normal deinterlacing, frames will be augmented by inverting, time reversing, and a combination of both. Then deinterlaced, augmentations reversed, and all 4 results averaged to get a more "true" result.

__*`mocomp`* (optional)__  
Motion compensation is an unofficial feature exclusive to DfConvEkSA which improves deinterlacing quality further, specifically on fast camera movements. It works by first doing a quick deinterlacing pass with DDD as a reference to then motion compensate the input frames for DfConvEkSA.  
This will roughly half processing speed.

__*`device`* (optional)__  
Possible values are "cuda" to use with an Nvidia GPU, or "cpu". DDD is usable on CPU, but DfConvEkSA and DeF are extremely slow.

__*`fp16`* (optional)__  
Up to doubles processing speed and halves VRAM usage. Strongly recommended if your GPU supports it. Does not work on CPU. DfConvEkSA only benefits from fp16 when mocomp is active.


## Tips
* If you would like to finetune or improve the results, consider using one of these deinterlacers as "EdiExt"-clip in [QTGMC](https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/f11d79c98589c9dcb5b10beec35b631db68b495c/havsfunc/havsfunc.py#L1912).
* The deinterlacers work okay for animation, but fail to use all information from the correct field on fast motions. May still be useful for orphaned fields. This is improved with the mocomp parameter.
* In my testing DeF seemed to perform similarly to DDD in most areas, but many times slower. The main benefit of DeF is that is seems to be best at generating missing information when it is not present on the other input fields.
