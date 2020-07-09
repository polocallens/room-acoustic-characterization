# Generates reverberant music and speech
import glob
import tqdm
from scipy.io import wavfile
import os
from argparse import ArgumentParser
import numpy as np
from scipy.signal import fftconvolve

# Custom imports
from utils.normalize import *

def parse_args():
    parser = ArgumentParser(description='MakeConvDataset')
    
    parser.add_argument(
        '-audioDir', '--audioDir',
        type=str, default=None, required=True,
        help='Music directory.'
    )
    
    parser.add_argument(
        '-rirDir', '--rirDir',
        type=str, default=None, required = True,
        help='rir directory'
    )
    
    parser.add_argument(
        '-outDir', '--outDir',
        type=str, default=None, required = True,
        help='output directory'
    )
    
    parser.add_argument(
        '-trim', '--trim',
        type=int, default=None, 
        help='Audio length in seconds'
    )
    
    parser.add_argument('--norm', dest='norm', action='store_true')
    parser.set_defaults(norm=False)
    
    return parser.parse_args()
    
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    audioDir = args.audioDir
    rirDir = args.rirDir
    outDir = args.outDir


    #resample audio first 
    
    if args.norm :
        print('---norm music+rir directories---')
        audioDir = resample_audio_dir(audioDir,trim=args.trim)
        rirDir = resample_audio_dir(rirDir)
    else :
        print("Skipping normalization, make sure you selected already normed samples")
    
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    print('---Convolving...---')
    for music_file in tqdm.tqdm(glob.glob(audioDir + '*')):

        m_sr, music_sig = wavfile.read(music_file)
        music_sig = (music_sig / np.max(np.abs(music_sig))).flatten()
        
        music_name = os.path.splitext(os.path.basename(music_file))[0]
        #print(f'zik : {music_name}')

        for rir_file in glob.glob(rirDir + '*'):
            rir_name = os.path.splitext(os.path.basename(rir_file))[0]

            if os.path.exists(outDir + rir_name + '/' + music_name + '.wav'):
                continue
                
            rir_sr, rir_sig = wavfile.read(rir_file)
            
            
            if not os.path.exists(outDir + rir_name):
                os.mkdir(outDir+rir_name)
            #print(f'rir : {rir_file}')
            
            rir_sig = (rir_sig / np.max(np.abs(rir_sig))).flatten()
                        
            music_rev = fftconvolve(music_sig, rir_sig, mode="full")
            music_rev = music_rev[:len(music_sig)]
            
            music_rev = music_rev / np.max(np.abs(music_rev))

            wavfile.write(outDir + rir_name + '/' + music_name + '.wav', m_sr, music_rev)