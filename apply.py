import cv2
import subprocess
import multiprocessing as mp
import argparse
from pathlib import Path
from video2frame import V2F
from frame2video import F2V
from inference_frame import main as style_transfer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", help="input video directory", required=True)
    parser.add_argument(
        "--style", "-s", help="style image directory", required=True)
    parser.add_argument(
        "--output", "-o", help="output video root directory", required=True)
    args = parser.parse_args()

    for video_path in Path(args.input).iterdir():
        if video_path.is_dir():
            continue
        input_video = cv2.VideoCapture(str(video_path))
        fps, width, height = input_video.get(cv2.CAP_PROP_FPS), input_video.get(
            cv2.CAP_PROP_FRAME_WIDTH), input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        V2F(input_video, video_path.parent / video_path.stem)

        style_transfer(video_path, Path(args.style), Path(args.output))
        subprocess.run(["rm", "-rf", f"{video_path.parent / video_path.stem}"])
        subprocess.run(
            ["ffmpeg", "-i", f"{video_path}", "-map", "0:a",
                f"{video_path.parent / video_path.stem}.m4a"]
        )

        for style_folder in Path(args.output).iterdir():
            F2V(style_folder / video_path.stem,
                style_folder / f"temp_{video_path.name}", fps, int(width), int(height))
            subprocess.run(["rm", "-rf", f"{style_folder / video_path.stem}"])
            subprocess.run(
                ["ffmpeg", "-i", str(style_folder / f"temp_{video_path.name}"), "-i",
                    f"{video_path.parent / video_path.stem}.m4a", "-c:v", "copy", "-c:a", "aac", f"{style_folder / video_path.name}"]
            )
            subprocess.run(
                ["rm", "-rf", str(style_folder / f"temp_{video_path.name}")])
        subprocess.run(
            ["rm", "-rf", f"{video_path.parent / video_path.stem}.m4a"])
