from decord import VideoReader, cpu
from PIL import Image
from typing import List, Iterator

def stream_video_blocks(video_path: str, sample_fps: float, block_size: int) -> Iterator[List[Image.Image]]:
    """
    A memory-efficient generator that opens a video and yields blocks of frames
    directly from the file without loading the entire video into RAM.

    Args:
        video_path (str): Path to the video file.
        sample_fps (float): The target frames per second to sample.
        block_size (int): The number of frames per block to yield.

    Yields:
        Iterator[List[Image.Image]]: A block of frames as a list of PIL Images.
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        video_fps = vr.get_avg_fps()

        if sample_fps <= 0:
            # If sample_fps is invalid, fall back to video's native FPS
            sample_fps = video_fps
            
        step = max(1, round(video_fps / sample_fps))
        indices = list(range(0, len(vr), step))
        
        for i in range(0, len(indices), block_size):
            batch_indices = indices[i:i + block_size]
            if not batch_indices:
                continue
            
            frames_array = vr.get_batch(batch_indices).asnumpy()
            yield [Image.fromarray(frame) for frame in frames_array]
            
    except Exception as e:
        raise IOError(f"Failed to stream or process video at '{video_path}': {e}")


# We keep the old function for backwards compatibility or other uses if needed,
# but our main pipeline will no longer use it.
def load_video_as_frames(video_path: str, sample_fps: float = 1.0) -> List[Image.Image]:
    """
    [DEPRECATED FOR STREAMING] Loads ALL sampled video frames into memory at once.
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        video_fps = vr.get_avg_fps()

        if sample_fps <= 0:
            step = 1
        else:
            step = max(1, round(video_fps / sample_fps))
        indices = list(range(0, len(vr), step))
        
        frames_array = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames_array]
    except Exception as e:
        raise IOError(f"Failed to load or process video at '{video_path}': {e}")
