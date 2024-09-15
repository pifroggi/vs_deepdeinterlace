# AI Deinterlacing functions for Vapoursynth
A collection of three temporally aware deep learning deinterlacers.  
This will double the frame rate, for example from 30i to 60p.  

| # | Deinterlacer | Quality | Speed     | Test Resolution | Hardware | Paper                                                                     | Code 
| - | ------------ | ------- | --------- | --------------- | -------- | ------------------------------------------------------------------------- | ----
| 1 | DDD          | Lower   | ~50 fps   | 720x480         | RTX 4090 | [Link](https://studios.disneyresearch.com/2020/11/10/deep-deinterlacing/) | [Link](https://github.com/vincentvdschaft/Disney-Deep-Deinterlacing)
| 2 | DeF          | Middle  | ~6 fps    | 720x480         | RTX 4090 | [Link](https://link.springer.com/chapter/10.1007/978-981-99-8073-4_28)    | [Link](https://github.com/Anonymous2022-cv/DeT)
| 3 | DfConvEkSA   | Higher  | ~5 fps    | 720x480         | RTX 4090 | [Link](https://arxiv.org/pdf/2404.13018)                                  | [Link](https://github.com/KUIS-AI-Tekalp-Research-Group/Video-Deinterlacing)

## Requirements
* [pytorch](https://pytorch.org/)  
* `pip install numpy`  
* `pip install positional_encodings` (optional, only for DeF)  
* `pip install -U openmim && pip install -U setuptools && python -m mim install "mmcv>=2.0.0"` (optional, only for DfConvEkSA)  

## Setup
Put the entire `vs_deepdeinterlace` folder into your vapoursynth scripts folder.  
Or install via pip: `pip install git+https://github.com/pifroggi/vs_deepdeinterlace.git`

## Usage
Applying these deinterlacers directly to YUV videos with chroma subsampling comes with some issues. This wrapper function works around this by deinterlacing luma and chroma seperately, keeping the subsampling intact. With this some parameters take two values: The first is for luma and the second for chroma.

    import vs_deepdeinterlace
    clip = vs_deepdeinterlace.YUV(clip, matrix_in_s="709", range_in_s="limited", deinterlacer=[3, 1], tff=True, tta=[False, False], mocomp=[False, False], device="cuda", fp16=True)

If your input clip is not subsampled, you can also use just one of the three deinterlacers directly:

    import vs_deepdeinterlace
    clip = vs_deepdeinterlace.DDD(clip, tff=True, tta=False, device="cuda", fp16=True)
    clip = vs_deepdeinterlace.DeF(clip, tff=True, tta=False, device="cuda", fp16=True)
    clip = vs_deepdeinterlace.DfConvEkSA(clip, tff=True, tta=False, mocomp=False, device="cuda", fp16=False)

__*`clip`*__  
Interlaced clip, not seperated into fields.  
Must be in YUV format for the YUV wrapper function.  
Must be in RGBS format when using the DDD, DeF, or DfConvEkSA function directly.

__*`matrix_in_s`, `range_in_s`*__  
The color matrix and color range of the input clip when using the YUV wrapper function, which also determines the output.

__*`deinterlacer`*__  
The deinterlacer to use. The first value is for luma and the second for chroma. See the table at the top for details.  
1 seems often good enough for chroma.

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

## Tips & Troubleshooting
* If installing the DfConvEkSA requirements does not work, try doing the following. On some systems a wheel needs to be build, which may take up to half an hour. If the driver, pytorch, or python is updated, reinstalling may be necessary, which can be done with the same commands:
  ```
  python -m mim uinstall mmcv
  pip install -U setuptools
  pip cache purge
  python -m mim install "mmcv>=2.0.0"
  ```
* If you would like to finetune or improve the results, consider using one of these deinterlacers as "EdiExt"-clip in [QTGMC](https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/f11d79c98589c9dcb5b10beec35b631db68b495c/havsfunc/havsfunc.py#L1912). This helps to remove some remaining temporal shimmer.
* The deinterlacers work okay for animation, but fail to use all information from the correct field on fast motions. May still be useful for orphaned fields. This is improved with the mocomp parameter.
* In my testing DeF seemed to perform similarly to DDD in most areas, but many times slower. The main benefit of DeF is that is seems to be best at generating missing information when it is not present on the other input fields.

