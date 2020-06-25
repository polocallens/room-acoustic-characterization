#Generates reverberant music and speech
import glob
import tqdm
from scipy.io import wavfile
from normalize import *
import os
from argparse import ArgumentParser
from scipy import signal

def parse_args():
    parser = ArgumentParser(description='MakeConvDataset')
    
    parser.add_argument(
        '-music_dir', '--music_dir',
        type=str, default=None, required=True,
        help='Music directory.'
    )
    
    parser.add_argument(
        '-rir_dir', '--rir_dir',
        type=str, default=None, required = True,
        help='rir directory'
    )
    
    parser.add_argument(
        '-out_dir', '--out_dir',
        type=str, default=None, required = True,
        help='output directory'
    )
    
    parser.add_argument(
        '-trim', '--trim',
        type=int, default=None, 
        help='output directory'
    )
    
    return parser.parse_args()
    
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    music_dir = args.music_dir
    rir_dir = args.rir_dir
    out_dir = args.out_dir


    #resample audio first 
    print('---norm music+rir directories---')
    normed_music_dir = resample_audio_dir(music_dir,trim=args.trim)
    normed_rir_dir = resample_audio_dir(rir_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('---Convolving...---')
    for music_file in tqdm.tqdm(glob.glob(normed_music_dir + '*')):

        m_sr, music_sig = wavfile.read(music_file)
        music_sig = music_sig / np.max(np.abs(music_sig))
        
        music_name = os.path.splitext(os.path.basename(music_file))[0]
        
        for rir_file in glob.glob(normed_rir_dir + '*'):
            rir_sr, rir_sig = wavfile.read(rir_file)
            
            rir_sig = rir_sig / np.max(np.abs(rir_sig))
                        
            music_rev = signal.fftconvolve(music_sig, rir_sig, mode="full")
            music_rev = music_rev[:len(music_sig)]
            
            music_rev = music_rev / np.max(np.abs(music_rev))

            wavfile.write(out_dir + music_name + '-' + os.path.basename(rir_file), m_sr, music_rev)
            
    print('---Normalizing outputs---')
