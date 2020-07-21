import os
from argparse import ArgumentParser
import sys 
import pickle

# Custom imports
from utils.acoustic_utils import *


def parse_args():
    parser = ArgumentParser(description='Data preparation')
    
    parser.add_argument(
        '-rirDir', '--rirDir',
        type=str, required = True,
        help='rir directory.'
    )
    
    parser.add_argument(
        '-outDir', '--outDir',
        type=str, required = True,
        help='output folder.'
    )
    
    parser.add_argument(
        '-params', '--params',
        nargs = '+', type=str, required = True,
        help='parameters to be computed.'
    )
    
    parser.add_argument(
        '-bands', '--bands',
        nargs = '+', default = [125.,  250.,  500., 1000., 2000., 4000.],
        help='center bands for param computations'
    )
    
    args = parser.parse_args()
    
    #check args 
    for param in args.params:
        if param not in ['t60', 'c50', 'c80', 'drr', 'all']:
            parser.print_help()
            sys.exit(1)
    
    return args

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    if not os.path.exists(args.outDir):
        os.mkdir(args.outDir)

    """# V1 
    print("Computing t60s...")
    rir2t60(args.rirDir, args.outDir)
    print("Computing c50...")
    rir2clarity(args.rirDir, args.outDir,50)
    print("Computing c80...")
    rir2clarity(args.rirDir, args.outDir,80)
    print("Computing drr...")
    rir2drr(args.rirDir, args.outDir)"""
    
    
    
    
#For 'all' --> bands = [100,500,5000]
#Ouput shape 1x12: [RT-100,RT-500,RT-5k,
#                C50-100,C50-500,C50-5k,
#                c80-100,c80-500,c80-5k,
#                drr-100,drr-500,drr-5k]


    #V2
    
    rir_list = sorted(glob.glob(args.rirDir + '*.wav'))
    
    #declare array for all gathered params
    """if 'all' in arg.params:
        if not os.path.exists(args.outDir + param):
            os.mkdir(args.outDir + param)
            
        params = ['t60', 'c50', 'c80', 'drr']
        
        if bands is not None:
            all_params = np.empty((len(args.params)-1) * len(args.bands))
        else:
            all_params = np.empty((len(args.params)-1))
        
    else:
        params = args.params"""
    
    if 'all' in args.params:
        params = ['t60','c50','c80','drr','all']
    else :
        params = args.params

    #Create param directories
    for param in params:
        if not os.path.exists(args.outDir + param):
            os.mkdir(args.outDir + param)
    
    bands = np.array(args.bands)
    #print(f'bands = {bands}')
    
    
    for i,rir_file in enumerate(tqdm(rir_list)):
        #Read rir file
        try:
            rir_sr, rir = wavfile.read(rir_file)
        except:
            print("Error encountered while parsing file: ",rir_file)
            sys.exit(1)
    
        #compute parameters
        if 't60' in params:
            t60 = t60_impulse(rir,rir_sr, bands,rt='t30')
        if 'c50' in params:
            c50 = acoustics.room.clarity(50, rir, rir_sr, bands)
        if 'c80' in params:
            c80 = acoustics.room.clarity(80, rir, rir_sr, bands)
        if 'drr' in params:
            drr = drr_impulse(rir, rir_sr, bands)
        
        if 'all' in params:
            all = np.hstack((t60,c50,c80,drr))
            
        filename = os.path.splitext(os.path.basename(rir_file))[0]

        for param in params:
            with open(args.outDir + param + "/" + filename + ".pkl", "wb") as f:
                pickle.dump(eval(param), f)
            f.close()
        #debug
        #print(f't60 = {t60}')
        #print(f'c50 = {c50}')
        #print(f'c80 = {c80}')
        #print(f'drr = {drr}')
        #print(f'shape of all : {all_params}')
        
        