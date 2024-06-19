
# Original paper https://arxiv.org/pdf/2404.13018
# Original code https://github.com/KUIS-AI-Tekalp-Research-Group/Video-Deinterlacing

# Vapoursynth implementation by pifroggi
# or tepete on the "Enhance Everything!" Discord Server

import vapoursynth as vs
import os
import torch
import numpy as np
import functools
from .DfConvEkSA_files.DfConvEkSA_arch import DfConvEkSA_arch

core = vs.core

def frame_to_array(frame: vs.VideoFrame) -> np.ndarray:
    array = np.empty((frame.height, frame.width, 3), dtype=np.float32)
    for p in range(frame.format.num_planes):
        array[..., p] = np.asarray(frame[p], dtype=np.float32)
    #remove the added border
    half_height = array.shape[0] // 2
    return array[:half_height]

def array_to_frame(img: np.ndarray, frame: vs.VideoFrame):
    for p in range(frame.format.num_planes):
        pls = frame[p]
        frame_arr = np.asarray(pls)
        np.copyto(frame_arr, img[:, :, p])

def tensor_to_array(tensor, min_max=(0, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    img_np = tensor.detach().numpy()
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    return img_np.astype(np.float32)

def apply_augmentations(arrays, inverse=False, reverse=False):
    augmented_arrays = arrays.copy()
    if inverse:
        augmented_arrays = [1 - arr for arr in augmented_arrays]
    if reverse:
        augmented_arrays = augmented_arrays[::-1]
    return augmented_arrays

def reverse_augmentations(arr, inverse=False, reverse=False):
    reversed_arr = arr.copy()
    if inverse:
        reversed_arr = 1 - reversed_arr
    return reversed_arr

#mirror frames at start or end
def mirror_index(n: int, i: int, max_frames: int) -> int:
    index = n + i
    if index < 0:
        index = -index
    elif index >= max_frames:
        index = 2 * (max_frames - 1) - index
    return index

def inference(arrays, device, model):
    arrays = [arr[:, :, [2, 1, 0]] for arr in arrays]
    imgs_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(np.stack(arrays, axis=0), (0, 3, 1, 2)))).float().to(device)

    with torch.no_grad():
        c, h, w = imgs_tensor.shape[-3:]
        parity_indicator = torch.zeros((1, c, h, w), device=device)
        imgs_LQ = torch.cat([imgs_tensor, parity_indicator], dim=0)
        imgs_LQ = imgs_LQ.unsqueeze(0)
        output = model(imgs_LQ)
        return output

def process_frame(n: int, f: vs.VideoFrame, clip: vs.VideoNode, device, model, tff=False, tta=False) -> vs.VideoFrame:
    frames = [clip.get_frame(mirror_index(n, i, clip.num_frames)) for i in range(-2, 3)]
    arrays = [frame_to_array(frame) for frame in frames]
    
    flip = (n % 2 != 0) if tff else (n % 2 == 0)
    
    if flip:
        arrays = [np.flipud(arr) for arr in arrays]
    
    if tta:
        augmentations = [
            {'inverse': False, 'reverse': False},
            {'inverse': True, 'reverse': False},
            {'inverse': False, 'reverse': True},
            {'inverse': True, 'reverse': True}
        ]
        
        augmented_results = []
        for aug in augmentations:
            augmented_arrays = apply_augmentations(arrays, **aug)
            output_tensor = inference(augmented_arrays, device, model)
            output_img = tensor_to_array(output_tensor)
            reversed_img = reverse_augmentations(output_img, **aug)
            augmented_results.append(reversed_img)
        
        output_img = np.mean(augmented_results, axis=0)
    else:
        output_tensor = inference(arrays, device, model)
        output_img = tensor_to_array(output_tensor)
    
    if flip:
        output_img = np.flipud(output_img)

    output_frame = f.copy()
    array_to_frame(output_img, output_frame)
    return output_frame

def DfConvEkSA(clip: vs.VideoNode, tff=False, tta=False, device='cuda', fp16=False) -> vs.VideoNode:

    #checks
    if clip.format.id not in [vs.RGBS]:
        raise ValueError("Input clip must be in RGBS format.")

    if fp16 is True:
        raise ValueError("DfConvEkSA does not work with fp16.")

    device = torch.device(device)
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'DfConvEkSA_files', 'DfConvEkSA_dim64k50_trainonYOUKU_150000_G.pth')
    model = DfConvEkSA_arch(64, 5, 8, 5, 7)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval().to(device)
    
    clip = core.std.SeparateFields(clip, tff=tff)
    #double frame height because for ModifyFrame input and output frames must have same dimensions
    clip = core.std.AddBorders(clip, bottom=clip.height)
    return clip.std.ModifyFrame(clips=[clip], selector=functools.partial(process_frame, clip=clip, device=device, model=model, tff=tff, tta=tta))
