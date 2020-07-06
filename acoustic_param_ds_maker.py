from acoustic_utils import *
import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='Data preparation')
    
    parser.add_argument(
        '-rirDir', '--rirDir',
        type=str, required = True,
        help='rir directory.'
    )
    
    parser.add_argument(
        '-outDir', '--outDir',
        type=str, required = True,
        help='output folder.'
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    if not os.path.exists(args.outDir):
        os.mkdir(args.outDir)

    print("Computing t60s...")
    rir2t60(args.rirDir, args.outDir)
    print("Computing c50...")
    rir2clarity(args.rirDir, args.outDir,50)
    print("Computing c80...")
    rir2clarity(args.rirDir, args.outDir,80)
    print("Computing drr...")
    rir2drr(args.rirDir, args.outDir)