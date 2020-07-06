#Generates reverberant music and speech
import glob
import pickle
from tqdm import tqdm
from scipy.io import wavfile
from normalize import *
from mfcc import *
import os
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='MakeConvDataset')
    
    parser.add_argument(
        '-rev_dir', '--rev_dir',
        type=str, default=None, required=True,
        help='Music directory.'
    )
    return parser.parse_args()

    
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    music_dir = args.rev_dir
    
    for room in tqdm(glob.glob(music_dir + '*')):
        for audio_sample_path in tqdm(glob.glob(room + '/*.wav')):
            sr_audio, audio = wavfile.read(audio_sample_path)
            mfcc = compute_norm_mfcc_3(audio,sr_audio)
            with open(os.path.splitext(audio_sample_path)[0] + '.pkl','wb') as pkl_file:
                pickle.dump(mfcc,pkl_file)