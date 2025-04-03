import argparse
from train_segmentation import LitUnsupervisedSegmenter

def parse_args():
    parser = argparse.ArgumentParser(description='Load pretrained model cfg')
    parser.add_argument('--path', help='checkpoint file path',
                        default=None, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args.path)

    model = LitUnsupervisedSegmenter.load_from_checkpoint(args.path)
    print(model.cfg)


if __name__ == '__main__':
    main()
    