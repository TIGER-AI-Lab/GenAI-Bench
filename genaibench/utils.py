import requests
import numpy as np
import av
from pathlib import Path
video_cache_dir = Path(__file__).parent / "video_cache"

def load_template(task, template):
    with open(Path(__file__).parent / "templates" / task / f"{template}.txt") as f:
        return f.read()
    
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def download_video(url, file_path):
    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return file_path



def process_video_into_frames(video_path, max_num_frames):
    if video_path.startswith("http"):
        video_path = download_video(video_path, video_cache_dir / Path(video_path).name)
    else:
        video_path = Path(video_path)
    container = av.open(video_path)
    num_frames = len(container.streams.video[0])
    if num_frames > max_num_frames:
        indices = np.arange(0, num_frames, num_frames / max_num_frames).astype(int)
    else:
        indices = np.arange(num_frames)
        
    frames = read_video_pyav(container, indices)
    return frames
