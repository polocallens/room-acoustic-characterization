import compute_mfcc acoustic_param_ds_maker convolute 

import subprocess
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='MakeConvDataset')
    
    parser.add_argument(
        '-rirDir', '--rirDir',
        type=str, default=None, required=True,
        help='Impulse response directory'
    )
    
    parser.add_argument(
        '-audioDir', '--audioDir',
        type=str, default=None, required=True,
        help='Audio directory'
    )

    parser.add_argument(
        '-trim', '--trim',
        type=int, default=4, 
        help='Audio length in seconds'
    )
    
    parser.add_argument(
        '-outDir', '--outDir',
        type=str, default=None, required=True,
        help='mfcc output directory'
    )
        
    return parser.parse_args()

    
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    subprocess.run(["python", "acoustic_param_ds_maker.py", "-rirDir", args.rirDir, "-outDir", args.outDir + 'acoustic_true_vals/'])
    
    subprocess.run(["python", "convolute.py", "-audioDir", args.audioDir, ])