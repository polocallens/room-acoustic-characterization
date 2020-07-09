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
        '-revDir', '--revDir',
        type=str, default=None, required=True,
        help='Reveberant signals directory.'
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
        
    for room in tqdm(glob.glob(args.revDir + '*')):
        room_name = os.path.split(room)[1]
        if not os.exists(args.outDir + room_name):
            os.makedirs(args.outDir + room_name)
            
        for audio_sample_path in tqdm(glob.glob(room + '/*.wav')):
            sr_audio, audio = wavfile.read(audio_sample_path)
            mfcc = compute_norm_mfcc_3(audio,sr_audio)
            
            audio_name = os.path.split(os.path.splitext(audio_sample_path)[0])[1]
            
            with open(args.outDir + room_name + '/' + audio_name + '.pkl','wb') as pkl_file:
                pickle.dump(mfcc,pkl_file)