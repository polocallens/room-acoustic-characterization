#Generates reverberant music and speech
from glob import glob
from tqdm import tqdm
from scipy.io import wavfile
from normalize import *
import os

music_dir =
rir_dir = 
out_dir = 


#normalize the samples first 
normed_music_dir = normalize_audio_dir(music_dir,trim=4)
normed_rir_dir = normalize_audio_dir(rir_dir)


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for music_file in tqdm(glob(normed_music_dir)):
    wavfile.read(music_file)
    for rir_file in tqdm(glob(normed_rir_dir)):
        