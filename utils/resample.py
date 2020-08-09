import glob
import tqdm
import os
from subprocess import call

def resample_file(file,sampling_rate = 16000, channels = 1):
    out_file = os.path.splitext(file)[0] + '_sr' + str(sampling_rate) + '_c_' + str(channels) + '.wav'
    call('sox -G ' + file + ' -r '+str(sampling_rate)+' -e float -c '+str(channels)+' ' + out_file + ' norm',shell=True)
    return out_file

#---------------------------------------------------------------------------------

def resample_trim_file(file, trim, sampling_rate = 16000, channels = 1):
    
    out_file = os.path.splitext(file)[0] + '_resampled_trimmed' + '.wav'
    
    out_dir, filename = os.path.split(file)
    
    temp_filepath = os.path.join(out_dir,'temp_' + filename)
    temp_filepath2 = os.path.join(out_dir,'temp2_' + filename)
      
    call('sox ' + file + ' ' + temp_filepath2 + ' pad 0 ' + str(trim),shell=True)
    call('sox ' + temp_filepath2 + ' ' + temp_filepath + ' trim 0 ' + str(trim),shell=True)
    call('sox -G ' +  temp_filepath + ' -r '+str(sampling_rate)+' -e float -c '+str(channels)+' ' + out_file + ' norm',shell=True)
    
    os.remove(temp_filepath)
    os.remove(temp_filepath2)
    
    return out_file
#---------------------------------------------------------------------------------
def resample_audio_dir(file_dir, sampling_rate = 16000, channels = 1, trim = None, trim_silence = False):
    
    
    print(f'------- Resampling directory : {file_dir} -------')
    

    if trim is not None :
        out_dir = file_dir.strip("/") + '_sr' + str(sampling_rate) + '_c_' + str(channels)  + '_' + str(trim) + 's'
    else :
        out_dir = file_dir.strip("/") + '_sr' + str(sampling_rate) + '_c_' + str(channels)
    
    file_list = sorted(glob.glob(os.path.join(file_dir, '*.wav')))
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        return(out_dir + '/')
    
    for file in tqdm.tqdm(file_list):
        filename = os.path.split(file)[1]
        if trim is not None:
            temp_filepath =  out_dir + '/' + 'temp_' + filename
            temp_filepath2 =  out_dir + '/' + 'temp2_' + filename
            
            call('sox ' + file + ' ' + temp_filepath2 + ' pad 0 ' + str(trim),shell=True)
            call('sox ' + temp_filepath2 + ' ' + temp_filepath + ' trim 0 ' + str(trim),shell=True)
            call('sox -G ' +  temp_filepath + ' -r '+str(sampling_rate)+' -e float -c '+str(channels)+' ' + out_dir + '/' + filename + ' norm',shell=True)
            os.remove(temp_filepath)
            os.remove(temp_filepath2)

        else:
            call('sox -G ' + file + ' -r '+str(sampling_rate)+' -e float -c '+str(channels)+' ' + out_dir + '/' + filename + ' norm',shell=True)
        
    return out_dir + '/'

#---------------------------------------------------------------------------------
def chunk_audio_files(music_dir,chunk_duration_s):
    print(f'------- Chunk audio directory : {music_dir} -------')
    music_list = sorted(glob.glob(os.path.join(music_dir, '*')))
    
    out_dir = music_dir.strip("/") + '_'+ str(chunk_duration_s) + 's/'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    for file in tqdm.tqdm(music_list):
        filename = os.path.split(file)[1]
        call('sox ' + file + ' ' + out_dir + '/' + filename + ' trim 0 ' + str(chunk_duration_s),shell=True)
    
#---------------------------------------------------------------------------------
def trim_silence_dir(dir):
    outdir = dir.strip("/") + '_trimmed'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    file_list = sorted(glob.glob(os.path.join(dir,'*.wav')))
    
    for file in tqdm.tqdm(file_list):
        outfile = os.path.join(outdir, os.path.basename(file))
        
        call('sox ' + file + ' ' + outfile + ' silence 1 0.01 0.05%',shell=True)
    
    return outdir
