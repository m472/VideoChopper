if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Combine videos of the same kayaker based on equipment color')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_split = subparsers.add_parser('split', help='splits videos into clips')

