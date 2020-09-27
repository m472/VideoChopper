#!/usr/bin/python3


def pair(arg: str) -> (str, str):
    file_filter, name, _ = arg.split(':')
    return file_filter, name


def test(*args, **kwargs):
    print(args, kwargs)


if __name__ == '__main__':
    import argparse
    from chopper import chop
    from train_classifier import create_model
    from classify import classify

    parser = argparse.ArgumentParser(description='Combine videos of the same kayaker based on equipment color')
    subparsers = parser.add_subparsers(help='sub-command help')
    subparsers.required = True

    parser_split = subparsers.add_parser('split', help='splits videos into clips')
    parser_split.add_argument('--left', '-l', type=int, default=1920//4)
    parser_split.add_argument('--top', '-t', type=int, default=1080//4)
    parser_split.add_argument('--right', '-r', type=int, default=3*1920//4)
    parser_split.add_argument('--bottom', '-b', type=int, default=3*1080//4)
    parser_split.add_argument('--output_directory', '-o', type=str, default='.')
    parser_split.add_argument('--output_file_prefix', '-p', type=str, default='out_')
    parser_split.add_argument('input_files', type=argparse.FileType('r'), nargs='+')
    parser_split.set_defaults(func=chop)

    parser_train = subparsers.add_parser('train', help='train neural network')
    parser_train.add_argument('key_value_pairs', nargs='+')
    parser_train.set_defaults(func=create_model)

    parser_combine = subparsers.add_parser('combine', help='combine clips using pre-trained neural network')
    parser_combine.add_argument('output directory')
    parser_combine.set_defaults(func=classify)

    try:
        args = parser.parse_args()
        args.func(args)
    except TypeError:
        parser.print_help()


