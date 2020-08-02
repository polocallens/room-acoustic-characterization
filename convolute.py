# Generates reverberant music and speech
import glob
import tqdm
from scipy.io import wavfile
import os
from argparse import ArgumentParser
import numpy as np
from scipy.signal import fftconvolve
import pickle
import shutil

# Custom imports
from utils.resample import *
from utils.mfcc import compute_norm_mfcc
from utils.noise import get_white_noise, get_noise_from_sound

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
        type=int, default=None, required = True,
        help='Audio length in seconds'
    )
    
    parser.add_argument(
        '-outFormat', '--outFormat',
        type=str, default='mfcc', 
        help='Output format --> mfcc or wavfile '
    )
    
    parser.add_argument(
        '-noiseSNR', '--noiseSNR',
        type=int, default=None, 
        help='Add pink noise to rev signal at the specified SNR'
    )
    
    parser.add_argument(
        '-noiseType', '--noiseType',
        type=str, default='white', 
        help='Type of noise --> white or real. if real, please specify file with noiseFile argument'
    )
    
    parser.add_argument(
        '-noiseFile', '--noiseFile',
        type=str, default=None, 
        help='Path to real noise file'
    )
    
    return parser.parse_args()
    
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    audioDir = args.audioDir
    rirDir = args.rirDir
    outDir = args.outDir

    #resample audio first 
    print('--resampling and trimming music--')
    audioDir = resample_audio_dir(audioDir,trim=args.trim)
    
    print('trimming start of rir directory')
    rirDir = trim_silence_dir(rirDir)
    
    print('resampling RIR directory')
    rirDir = resample_audio_dir(rirDir)
    
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    print('---Convolving...---')
    for music_file in tqdm.tqdm(glob.glob(audioDir + '*')):

        m_sr, music_sig = wavfile.read(music_file)
        music_sig = (music_sig / np.max(np.abs(music_sig))).flatten()
        
        music_name = os.path.splitext(os.path.basename(music_file))[0]
        
        if m_sr != 16000 :
            raise ValueError(f'wrong sampling rate with {music_file}')

        #print(f'zik : {music_name}')

        for rir_file in glob.glob(rirDir + '*'):
            rir_name = os.path.splitext(os.path.basename(rir_file))[0]

            if os.path.exists(os.path.join(outDir, rir_name, music_name + '.wav')):
                continue
            
            #Read rir
            rir_sr, rir_sig = wavfile.read(rir_file)
            
            #Make RIR directory
            if not os.path.exists(os.path.join(outDir, rir_name)):
                os.mkdir(os.path.join(outDir,rir_name))
                    
            rir_sig = (rir_sig / np.max(np.abs(rir_sig))).flatten()
                        
            music_rev = fftconvolve(music_sig, rir_sig, mode="full")
            
            #Normalize reverberant signal
            music_rev = music_rev / np.max(np.abs(music_rev))
            
            #Add noise
            if args.noiseSNR is not None:
                if args.noiseType == 'white':
                    noise = get_white_noise(music_rev,args.noiseSNR)
                elif args.noiseType == 'real':
                    if noiseFile is None:
                        print("Please specify noiseFile argument")
                        sys.exit(1)
                    noiseFile = resample_file(args.noiseFile)
                    noise_sr, real_noise = wavfile.read(noiseFile)
                    
                    #if noise is longer than reverberant signal, cut its tail
                    #in the contrary, repeats until music and noise sizes are equal
                    real_noise = np.resize(real_noise,music_rev.shape) 
                    
                    #normalize
                    real_noise = real_noise / np.max(np.abs(real_noise))
                    
                    noise = get_noise_from_sound(music_rev,real_noise,args.noiseSNR)
                else:
                    print("please specify noiseType")
                    sys.exit(1)
                    
                music_rev = music_rev + noise
                music_rev = music_rev / np.max(np.abs(music_rev))
                
            
            if args.outFormat == 'mfcc':
                mfcc = compute_norm_mfcc(music_rev,m_sr)
                with open(os.path.join(outDir, rir_name, music_name + '.pkl'),'wb') as pkl_file:
                    pickle.dump(mfcc,pkl_file)
                    
            elif args.outFormat == 'wavfile':
                wavfile.write(os.path.join(outDir, rir_name, music_name + '.wav'), m_sr, music_rev)