import cv2
from pathlib import Path
import os
from tqdm import tqdm


def F2V(fdir_path, vpath, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(str(vpath), fourcc, fps, (width, height))
    for file in tqdm(sorted(Path(fdir_path).iterdir(), key=lambda img: int(img.stem))):
        frame = cv2.imread(str(file))
        frame = cv2.resize(frame, dsize=(width, height),
                           interpolation=cv2.INTER_CUBIC)
        video.write(frame)
    video.release()


if __name__ == "__main__":
    F2V("/mnt/c/Sample/result_video/kQVydSe/Yes",
        "/mnt/c/Sample/result_video/kQVydSe/Yes.mp4", 30, 512, 256)
