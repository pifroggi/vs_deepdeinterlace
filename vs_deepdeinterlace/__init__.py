import vapoursynth as vs

#deinterlacers
from .DeF import DeF
from .DfConvEkSA import DfConvEkSA
from .DDD import DDD

core = vs.core

#combined deinterlacer uses two of the ones in this package to first deinterlace, then for every frame, align the surrounding frames to it, then deinterlaces again
#this makes deinterlacing on fast movements much better, but is also slower

def combined(clip, tff=False, tta=False, device='cuda', fp16=True):
    import vs_align
    
    #checks
    if clip.format.id not in [vs.RGBS]:
        raise ValueError("Input clip must be in RGBS format.")

    #deinterlace quickly as reference for alignment
    quick_deinterlaced = DDD(clip, tff=tff, tta=False, device=device, fp16=fp16)

    #create superclip with segments like this: 0-4, 1-5, 2-6, 3-7,... so a segment has the middle original frame with the next two and previous two needed for deinterlacing
    def create_superclip(clip, segment_length=5):
        segments = []
        for i in range(clip.num_frames - segment_length + 1):
            segments.append(clip[i:i + segment_length])
        return core.std.Splice(segments)
    superclip = create_superclip((quick_deinterlaced[:2]+quick_deinterlaced), segment_length=5)

    #in a cycle of 5, replace every frame with the middle one; this will be used as reference for alignment
    middle_frame = superclip.std.SelectEvery(cycle=5, offsets=2)
    middle_frame = core.std.Interleave([middle_frame] * 5)

    #align superclip to middle_frame clip; now each original frame has 2 following and 2 previous frames aligned to it
    aligned_to_midframe = vs_align.spatial(superclip, middle_frame, precision=2, iterations=1, blur_strength=0, device=device)
    aligned_to_midframe = vs_align.spatial(aligned_to_midframe, middle_frame, precision=4, iterations=1, blur_strength=0, device=device)

    #get original fields and weave
    aligned_to_midframe = core.std.SeparateFields(aligned_to_midframe, tff=tff)
    aligned_to_midframe = core.std.SelectEvery(aligned_to_midframe, cycle=4, offsets=[0, 3])
    aligned_to_midframe = core.std.DoubleWeave(aligned_to_midframe, tff=tff)
    aligned_to_midframe = core.std.SelectEvery(aligned_to_midframe, 2, 0, modify_duration=False)

    #deinterlace again, now that fields are better aligned
    improved_deinterlace = DfConvEkSA(aligned_to_midframe, tff=tff, tta=tta, device=device)

    #select middle frames from deinterlaced superclip 
    improved_deinterlace = core.std.SelectEvery(improved_deinterlace, cycle=5, offsets=2)

    return improved_deinterlace
