import cv2
from pathlib import Path


def V2F(video: cv2.VideoCapture, ipath: Path) -> None:
    if not ipath.exists():
        ipath.mkdir()
    ret, frame = video.read()
    index = 0
    while ret:
        cv2.imwrite(str(ipath / f"{index}.png"), frame)
        ret, frame = video.read()
        index += 1
    video.release()

    return


if __name__ == "__main__":
    from sys import argv
    for video_path in list(Path(argv[1]).iterdir()):
        if not video_path.is_dir():
            video = cv2.VideoCapture(str(video_path))
            V2F(video, video_path.parent / video_path.stem)
