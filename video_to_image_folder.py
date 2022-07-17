import os
import cv2 as cv

from tqdm import tqdm
from random import random
from typing import Tuple


class VideoCaptureWrapper:
    def __init__(self, cap: cv.VideoCapture):
        self.cap = cap

    def __enter__(self):
        return self.cap

    def __exit__(self, *_):
        self.cap.release()


def extract_frames(video_filename: str, output_dir: str, 
        region_of_interest: Tuple[int, int, int, int], 
        /, 
        train_split: float = 0.8, 
        output_image_size=64) -> None:

    with VideoCaptureWrapper(cv.VideoCapture(fname)) as cap:
        divisor = int(cap.get(cv.CAP_PROP_FPS))
        if not cap.isOpened():
            raise Exception('error opeing file')

        success = True
        for i in tqdm(range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))):
            success, frame = cap.read()
            if success and i % divisor == 0:
                _set = 'train' if random() < train_split else 'test'
                output_name = os.path.join(output_dir, _set, f'{os.path.split(video_filename)[-1]}_{i // divisor}.png')
                t, b, l, r = region_of_interest
                cv.imwrite(output_name, cv.resize(frame[t:b, l:r], (output_image_size, output_image_size)))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Converts an image to a series of images with one image every second')
    parser.add_argument('--output-size', default=64, type=int, help='Output image size in pixels (output images are square)')
    parser.add_argument('--output-dir', default='out', type=str, help='Output directory for the images. Two folders "train" and "test" will be created in the output folder if they don\'t already exist')
    parser.add_argument('--train-split', default=0.8, type=float, help='Portion of video frames to be used for training')
    parser.add_argument('--top', default=0, type=int, help='Input region of interest top')
    parser.add_argument('--bottom', default=-1, type=int, help='Input region of interest bottom')
    parser.add_argument('--left', default=0, type=int, help='Input region of interest left')
    parser.add_argument('--right', default=-1, type=int, help='Input region of interest right')
    parser.add_argument('input_files', metavar='input-files', nargs='+', type=str, help='Input files to be processed')

    args = parser.parse_args()
    print(args)

    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)

    for fname in args.input_files:
        extract_frames(fname, args.output_dir, 
                       (args.top, args.bottom, args.left, args.right),
                       args.train_split,
                       args.output_size)

