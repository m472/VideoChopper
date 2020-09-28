#!/usr/bin/python3

import os
from typing import Type, Dict, Union
from enum import Enum

from chopper import chop, VideoWriter, PaletteWriter, OpenCVVideoWriter, ImageWriter, SingleFileNumpyWriter, \
    PerFrameNumpyWriter
from train_classifier import create_model
from classify import classify


class VideoWriterChoices(Enum):
    Video: VideoWriter = OpenCVVideoWriter
    Images: VideoWriter = ImageWriter


class PaletteWriterChoices(Enum):
    OneFilePerClip: PaletteWriter = SingleFileNumpyWriter
    OneFilePerFrame: PaletteWriter = PerFrameNumpyWriter


def pair(arg: str) -> (str, str):
    file_filter, name, _ = arg.split(':')
    return file_filter, name


def split_wrapper(args):
    output_file_pattern = os.path.join(args.output_directory, args.output_file_prefix)
    chop([f.name for f in args.input_files],
         args.left,
         args.top,
         args.right,
         args.bottom,
         getattr(VideoWriterChoices, args.video_writer).value(output_file_pattern),
         getattr(PaletteWriterChoices, args.palette_writer).value(output_file_pattern),
         args.debug)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Combine videos of the same kayaker based on equipment color')
    parser.add_argument('--debug', '-d', action='store_true', help='print debug information and show images')
    subparsers = parser.add_subparsers(help='sub-command help')
    subparsers.required = True

    parser_split = subparsers.add_parser('split', help='splits videos into clips')
    parser_split.add_argument('--left', '-l', type=int, default=1920 // 4)
    parser_split.add_argument('--top', '-t', type=int, default=1080 // 4)
    parser_split.add_argument('--right', '-r', type=int, default=3 * 1920 // 4)
    parser_split.add_argument('--bottom', '-b', type=int, default=3 * 1080 // 4)
    parser_split.add_argument('--output_directory', '-o', type=str, default='.')
    parser_split.add_argument('--output_file_prefix', '-p', type=str, default='out_')
    parser_split.add_argument('--video_writer', choices=[member.name for member in VideoWriterChoices],
                              default=VideoWriterChoices.Images.name)
    parser_split.add_argument('--palette_writer', choices=[member.name for member in PaletteWriterChoices],
                              default=PaletteWriterChoices.OneFilePerFrame.name)
    parser_split.add_argument('input_files', type=argparse.FileType('r'), nargs='+')
    parser_split.set_defaults(func=split_wrapper)

    parser_train = subparsers.add_parser('train', help='train neural network')
    parser_train.add_argument('key_value_pairs', nargs='+')
    parser_train.set_defaults(func=create_model)

    parser_combine = subparsers.add_parser('combine', help='combine clips using pre-trained neural network')
    parser_combine.add_argument('output directory')
    parser_combine.set_defaults(func=classify)

    args = parser.parse_args()
    args.func(args)
