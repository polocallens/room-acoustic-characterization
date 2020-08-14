#custom imports
from utils.resample import *
from utils.mfcc import *

#Generates reverberant music and speech
import glob
import pickle
from tqdm import tqdm
from scipy.io import wavfile
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
    
    parser.add_argument(
        '-trim', '--trim',
        type=int, default=8,
        help='audio length in seconds'
    )
        
    return parser.parse_args()

    
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
        
    for room in tqdm(glob.glob(os.path.join(args.revDir, '*'))):
    
        room_name = os.path.split(room)[1]
        room = resample_audio_dir(room,trim=args.trim)
        
        if not os.path.exists(os.path.join(args.outDir, room_name)):
            os.makedirs(os.path.join(args.outDir,room_name))
            
        for audio_sample_path in glob.glob(os.path.join(room,'*.wav')):
            sr_audio, audio_rev = wavfile.read(audio_sample_path)
            audio_rev = audio_rev / np.max(np.abs(audio_rev))
            
            mfcc = compute_norm_mfcc(audio_rev,sr_audio)
            
            audio_name = os.path.split(os.path.splitext(audio_sample_path)[0])[1]
            
            with open(os.path.join(args.outDir, room_name, audio_name + '.pkl'),'wb') as pkl_file:
                pickle.dump(mfcc,pkl_file)