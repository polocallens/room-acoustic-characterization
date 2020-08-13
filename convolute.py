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
import sys

# Custom imports
from utils.resample import *
from utils.mfcc import compute_norm_mfcc, compute_melspectrogram, compute_logspectrogram2, compute_logspectrogram3
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
        nargs = '+', type=str, required = True,
        help='output format (mfcc mel wavfile)'
    )
    
    parser.add_argument(
        '-noiseSNR', '--noiseSNR',
        type=int, default=None, 
        help='Add noise to rev signal at the specified SNR'
    )
    
    parser.add_argument(
        '-noiseType', '--noiseType',
        type=str, default='white', 
        help='Type of noise --> white, random or real. if real, please specify file with noiseFile argument'
    )
    
    parser.add_argument(
        '-noiseFile', '--noiseFile',
        type=str, default=None, 
        help='Path to real noise file'
    )
    
    parser.add_argument('--no-rir-resampling', dest='rir_resampling', action='store_false')
    parser.set_defaults(rir_resampling=True)
    
    parser.add_argument('--no-audio-resampling', dest='audio_resampling', action='store_false')
    parser.set_defaults(audio_resampling=True)
    
    return parser.parse_args()
    
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    audioDir = args.audioDir
    rirDir = args.rirDir
    outDir = args.outDir

    #resample audio first 
    if args.audio_resampling:
        print('--resampling and trimming music--')
        audioDir = resample_audio_dir(audioDir,trim=args.trim)
    
    if (args.rir_resampling):
        print('resampling RIR directory')
        rirDir = resample_audio_dir(rirDir)
    
        print('trimming start of rir directory')
        rirDir = trim_silence_dir(rirDir)
    else :
        print('skipping rir resampling')
    
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    print('---Convolving...---')
    for music_file in tqdm.tqdm(glob.glob(os.path.join(audioDir,'*.wav'))):

        m_sr, music_sig = wavfile.read(music_file)
        music_sig = (music_sig / np.max(np.abs(music_sig))).flatten()
        
        music_name = os.path.splitext(os.path.basename(music_file))[0]
        
        if m_sr != 16000 :
            raise ValueError(f'wrong sampling rate with {music_file}')

        #print(f'zik : {music_name}')

        for rir_file in glob.glob(os.path.join(rirDir,'*.wav')):
            rir_name = os.path.splitext(os.path.basename(rir_file))[0]

            #pass if already computed
            if args.outFormat in ['mfcc','mel','logspectrogram'] and os.path.isfile(os.path.join(outDir, rir_name, music_name + '.pkl')):
                continue
            if args.outFormat == 'wavfile' and os.path.exists(os.path.join(outDir, rir_name, music_name + '.wav')):
                continue
            
            #Read rir
            rir_sr, rir_sig = wavfile.read(rir_file)
            
            #make output directories 
            for param in args.outFormat:
                if not os.path.exists(os.path.join(outDir,param)):
                        os.makedirs(os.path.join(outDir,param))
                #Make RIR directory
                if not os.path.exists(os.path.join(outDir, param,rir_name)):
                    os.makedirs(os.path.join(outDir,param,rir_name))
                    
            rir_sig = (rir_sig / np.max(np.abs(rir_sig))).flatten()
                        
            music_rev = fftconvolve(music_sig, rir_sig, mode="full")
            
            #Normalize reverberant signal
            music_rev = music_rev / np.max(np.abs(music_rev))
            
            #fix length
            music_rev = np.pad(music_rev,(0,m_sr*args.trim))[:m_sr*args.trim]
            
            #Add noise
            if args.noiseSNR is not None:
                if args.noiseType == 'white':
                    noise = get_white_noise(music_rev,args.noiseSNR)
                elif args.noiseType == 'random':
                    noiseSNR = np.random.randint(low=0,high=15)
                    noise = get_white_noise(music_rev,noiseSNR)
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
                    
                #Add noise
                music_rev = music_rev + noise
                
                #normalize and convert back to float32
                music_rev = np.float32(music_rev / np.max(np.abs(music_rev)))
                
            
            if 'mfcc' in args.outFormat:
                mfcc = compute_norm_mfcc(music_rev,m_sr)
                with open(os.path.join(outDir,'mfcc', rir_name, music_name + '.pkl'),'wb') as pkl_file:
                    pickle.dump(mfcc,pkl_file)
                    
            if 'mel' in args.outFormat:
                mel = compute_melspectrogram(music_rev,m_sr)
                with open(os.path.join(outDir, 'mel',rir_name, music_name + '.pkl'),'wb') as pkl_file:
                    pickle.dump(mel,pkl_file)
                    
            if 'logspectrogram' in args.outFormat:
                spec = compute_logspectrogram3(music_rev,m_sr)
                with open(os.path.join(outDir, 'logspectrogram',rir_name, music_name + '.pkl'),'wb') as pkl_file:
                    pickle.dump(spec,pkl_file)
                    
            if 'wavfile' in args.outFormat:
                wavfile.write(os.path.join(outDir, 'wavfile',rir_name, music_name + '.wav'), m_sr, music_rev)
        
    #þrint shape for network input
    if 'logspectrogram' in args.outFormat:
        print(f'logspectrogram shape is {spec.shape}')
    if 'mfcc' in args.outFormat:
        print(f'mfcc shape is {mfcc.shape}')
    if 'mel' in args.outFormat:
        print(f'mel shape is {mel.shape}')