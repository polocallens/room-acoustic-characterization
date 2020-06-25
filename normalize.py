import glob
import tqdm
import os
from subprocess import call

def normalize_audio_dir(file_dir, sampling_rate = 16000, channels = 1, trim = None):
    
    print(f'------- Normalizing directory : {file_dir} -------')
    file_list = sorted(glob.glob(file_dir + '*'))

    if trim is not None :
        out_dir = file_dir.strip("/") + '_sr' + str(sampling_rate) + '_c_' + str(channels)  + '_' + str(trim) + 's'
    else :
        out_dir = file_dir.strip("/") + '_sr' + str(sampling_rate) + '_c_' + str(channels)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for file in tqdm.tqdm(file_list):
        filename = os.path.split(file)[1]
        if trim is not None:
            temp_filepath =  out_dir + '/' + 'temp_' + filename
            
            call('sox ' + file + ' ' + temp_filepath + ' trim 0 ' + str(trim),shell=True)
            call('sox -G ' +  temp_filepath + ' -r '+str(sampling_rate)+' -e float -c '+str(channels)+' -b 16 ' + out_dir + '/' + filename,shell=True)
            os.remove(temp_filepath)
        
        else:
            call('sox -G ' + file + ' -r '+str(sampling_rate)+' -e float -c '+str(channels)+' -b 16 ' + out_dir + '/' + filename,shell=True)
        
    return out_dir



def chunk_audio_files(music_dir,chunk_duration_s):
    print(f'------- Chunk audio directory : {music_dir} -------')
    music_list = sorted(glob.glob(music_dir + '*'))
    
    out_dir = music_dir.strip("/") + '_'+ str(chunk_duration_s) + 's/'
    
    try : 
        os.mkdir(out_dir)
    except:
        print('Output folder already exists')  
        
    for file in tqdm(music_list):
        filename = os.path.split(file)[1]
        call('sox ' + file + ' ' + out_dir + '/' + filename + ' trim 0 ' + str(chunk_duration_s),shell=True)