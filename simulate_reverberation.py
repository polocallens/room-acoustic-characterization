import glob
import tqdm
from scipy.io import wavfile
import os
from argparse import ArgumentParser
import numpy as np
from scipy.signal import fftconvolve

# Custom imports
from utils.resample import *
from utils.noise import get_white_noise, get_noise_from_sound

def parse_args():
    parser = ArgumentParser(description='Simulate reverberation')
    
    parser.add_argument(
        '-audio', '--audio',
        type=str, default=None, required=True,
        help='Audio file'
    )
    
    parser.add_argument(
        '-rir', '--rir',
        type=str, default=None, required = True,
        help='RIR file'
    )
    
    parser.add_argument(
        '-trim', '--trim',
        type=int, default=None, required = True,
        help='Audio length in seconds'
    )
    
    parser.add_argument(
        '-noiseSNR', '--noiseSNR',
        type=int, default=None, 
        help='Add noise to rev signal at the specified SNR'
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
    
    print('resampling and trimming music')
    audio_resampled = resample_trim_file(args.audio, trim=args.trim, sampling_rate=16000, channels = 1,)
    
    print('resampling RIR')
    rir_resampled = resample_file(args.rir, sampling_rate=16000, channels = 1)
    
    print('Simulating reverberation')
    #read audio
    audio_sr, audio_sig = wavfile.read(audio_resampled)
    #normalize audio
    audio_sig = (audio_sig / np.max(np.abs(audio_sig))).flatten()
    
    
    #read rir
    rir_sr, rir_sig = wavfile.read(rir_resampled)
    #normalize rir
    rir_sig = (rir_sig / np.max(np.abs(rir_sig))).flatten()
    
    #simulate reverberation
    audio_rev = fftconvolve(audio_sig, rir_sig, mode="full")
    #normalize reverberant signal
    audio_rev = audio_rev / np.max(np.abs(audio_rev))
    
    #fix length
    audio_rev = np.pad(audio_rev,(0,audio_sr*args.trim))[:audio_sr*args.trim]
    
    #Add noise
    if args.noiseSNR is not None:
        print(f'Adding {args.noiseType} noise with {args.noiseSNR}dB SNR')
        if args.noiseType == 'white':
            noise = get_white_noise(audio_rev,args.noiseSNR)
        elif args.noiseType == 'random':
            noiseSNR = np.random.randint(low=0,high=15)
            noise = get_white_noise(audio_rev,noiseSNR)
        elif args.noiseType == 'real':
            if noiseFile is None:
                print("Please specify noiseFile argument")
                sys.exit(1)
            noiseFile = resample_file(args.noiseFile)
            noise_sr, real_noise = wavfile.read(noiseFile)

            #if noise is longer than reverberant signal, cut its tail
            #in the contrary, repeats until music and noise sizes are equal
            real_noise = np.resize(real_noise,audio_rev.shape) 

            #normalize
            real_noise = real_noise / np.max(np.abs(real_noise))

            noise = get_noise_from_sound(audio_rev,real_noise,args.noiseSNR)
        else:
            print("please specify noiseType")
            sys.exit(1)

        #Add noise
        audio_rev = audio_rev + noise

        #normalize and convert back to float32
        audio_rev = np.float32(audio_rev / np.max(np.abs(audio_rev)))
        
    #save audio file
    out_dir, audio_name = os.path.split(args.audio)
    #remove extension
    audio_name = os.path.splitext(audio_name)[0]
    
    out_path = os.path.join(out_dir, audio_name + '_reverberant.wav')
    print(f'Saving reverberant audio file as : {out_path}')
    wavfile.write(out_path, audio_sr, audio_rev)