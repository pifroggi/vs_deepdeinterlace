import vapoursynth as vs
import os
import torch
import numpy as np
import functools
from .DDD_files.model_base_arch import ModelConfig, _MODEL_FROM_TAG
from .DDD_files import DDD_arch

core = vs.core

def frame_to_array(frame: vs.VideoFrame) -> np.ndarray:
    full_image = np.dstack([np.asarray(frame[p]) for p in range(frame.format.num_planes)])
    #remove the added border
    half_width = full_image.shape[1] // 2
    return full_image[:, :half_width]

def array_to_frame(img: np.ndarray, frame: vs.VideoFrame):
    for p in range(frame.format.num_planes):
        pls = frame[p]
        frame_arr = np.asarray(pls)
        np.copyto(frame_arr, img[:, :, p])

def tensor_to_array(tensor, min_max=(0, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    img_np = tensor.detach().cpu().numpy()
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

def mirror_index(n: int, i: int, max_frames: int) -> int:
    index = n + i
    if index < 0:
        index = -index
    elif index >= max_frames:
        index = 2 * (max_frames - 1) - index
    return index

def inference(arrays, model, device):
    imgs_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(np.stack(arrays, axis=0), (0, 3, 1, 2)))).float().to(device)
    with torch.no_grad():
        known_fields = torch.cat([imgs_tensor[0], imgs_tensor[1], imgs_tensor[2]], dim=0).to(device)
        output = model.forward(known_fields.unsqueeze(0)).detach().cpu()
        torch.clamp_(output, 0, 1)
        im_network = model.sample_to_im((known_fields.unsqueeze(0).cpu(), output), num_color_channels=3)
        return im_network

def process_frame(n: int, f: vs.VideoFrame, clip: vs.VideoNode, model, device, tff=False, taa=False) -> vs.VideoFrame:
    frames = [clip.get_frame(mirror_index(n, i, clip.num_frames)) for i in range(-1, 2)]
    arrays = [frame_to_array(frame) for frame in frames]
    
    flip = (n % 2 == 0) if tff else (n % 2 != 0)
    if flip:
        arrays = [np.fliplr(arr) for arr in arrays]

    if taa:
        augmentations = [
            {'inverse': False, 'reverse': False},
            {'inverse': True, 'reverse': False},
            {'inverse': False, 'reverse': True},
            {'inverse': True, 'reverse': True}
        ]
        
        augmented_results = []
        for aug in augmentations:
            augmented_arrays = apply_augmentations(arrays, **aug)
            output_img = inference(augmented_arrays, model, device)
            reversed_img = reverse_augmentations(output_img, **aug)
            augmented_results.append(reversed_img)
        
        output_img = np.mean(augmented_results, axis=0)
    else:
        output_img = inference(arrays, model, device)
    
    if flip:
        output_img = np.fliplr(output_img)
    
    output_frame = f.copy()
    array_to_frame(output_img, output_frame)
    return output_frame

def DDD(clip: vs.VideoNode, tff=False, taa=False, device='cuda') -> vs.VideoNode:
    device = torch.device(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_config_path = os.path.join(current_dir, 'DDD_files', 'model_config.yaml')
    config = model_config_path
    config_object = ModelConfig(config)
    config_object.config_path = str(config)
    model_class = _MODEL_FROM_TAG[config_object.model_class.lower()]
    model = model_class(config_object, device=device)

    clip = core.std.SeparateFields(clip, tff=tff)
    #transpose because model was trained on vertical interlacing, then double frame width because for ModifyFrame input and output frames must have same dimensions
    clip = core.std.Transpose(clip)
    clip = core.std.AddBorders(clip, right=clip.width)
    clip = clip.std.ModifyFrame(clips=[clip], selector=functools.partial(process_frame, clip=clip, model=model, device=device, tff=tff, taa=taa))
    clip = core.std.Transpose(clip)
    return clip
    
    
    
    
    
    
    
    
    
    
