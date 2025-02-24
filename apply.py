import cv2
import subprocess
import argparse
from pathlib import Path
from sys import argv
from video2frame import V2F
from frame2video import F2V
# from inference_frame import main as style_transfer
from style_transfer import main as style_transfer


# Requires ffmpeg to run

if __name__ == "__main__":
    default_input_folder = Path(argv[0]).parent / "input"
    default_style_folder = Path(argv[0]).parent / "style"
    default_output_folder = Path(argv[0]).parent / "result"
    temp_root = Path(argv[0]).parent / "temp"
    if not default_input_folder.exists():
        default_style_folder.mkdir()
    if not default_style_folder.exists():
        default_style_folder.mkdir()
    if not default_output_folder.exists():
        default_output_folder.mkdir()
    if not temp_root.exists():
        temp_root.mkdir()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", help="input video directory", nargs="?", default=f"{str(default_input_folder)}")
    parser.add_argument(
        "--style", "-s", help="style image directory", nargs="?", default=f"{str(default_style_folder)}")
    parser.add_argument(
        "--output", "-o", help="output video root directory", nargs="?", default=f"{str(default_output_folder)}")
    parser.add_argument(
        "--not_remove", "-nr", action="store_true", help="not remove stylized frame")
    args = parser.parse_args()

    for video_path in Path(args.input).iterdir():
        # Get video
        if video_path.is_dir():
            continue
        input_video = cv2.VideoCapture(str(video_path))
        fps, width, height = input_video.get(cv2.CAP_PROP_FPS), input_video.get(
            cv2.CAP_PROP_FRAME_WIDTH), input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Make directory for the video to store its origin frame
        video_temp = temp_root / video_path.stem
        if not video_temp.exists():
            video_temp.mkdir()
        # Transfer video into frames
            V2F(input_video, video_temp)

        # Style-transfer the frames
        styled_video_folder = Path(args.output) / video_path.stem
        if not styled_video_folder.exists():
            styled_video_folder.mkdir()
        print(f"Now processing: {video_path.stem}")
        style_transfer(video_temp, Path(args.style), styled_video_folder)

        # Get the audio file
        audio_path = temp_root / f"{video_path.stem}.m4a"
        if not audio_path.exists():
            subprocess.run(
                ["ffmpeg", "-i", f"{video_path}", "-map", "0:a",
                    f"{audio_path}"]
            )

        # Turn the style-transfered frames into video
        for style_folder in styled_video_folder.iterdir():
            if not style_folder.is_dir():
                continue
            try:
                F2V(style_folder,
                    styled_video_folder / f"temp_{video_path.name}", fps, int(width), int(height))
            except FileNotFoundError:
                pass
            if args.not_remove is False:
                subprocess.run(
                    ["rm", "-rf", f"{style_folder / video_path.stem}"])
        # Combine the output video with audio
            styled_video_name = style_folder.name + "_" + video_path.name
            if not (styled_video_folder / styled_video_name).exists():
                subprocess.run(
                    ["ffmpeg", "-i", str(styled_video_folder / f"temp_{video_path.name}"), "-i",
                        f"{audio_path}", "-c:v", "copy", "-c:a", "aac", f"{styled_video_folder / styled_video_name}"]
                )
            subprocess.run(
                ["rm", "-rf", str(styled_video_folder /
                                  f"temp_{video_path.name}")]
            )
        # subprocess.run(
        #     ["rm", "-rf", f"{video_path.parent / video_path.stem}.m4a"])
