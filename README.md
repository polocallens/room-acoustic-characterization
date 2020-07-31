# music-parameters-estimation




## Instructions

You will need :
- RIR directory --> rirDir
- Music or speech directory --> musicDir 

## Datasets
#### RIR


### Convolve music dataset with RIRs

```console
python convolute.py -audioDir YOUR_AUDIO_DIR/ -rirDir YOUR_RIR_DIR/ -outDir WHERE_YOU_WANT_YOUR_CONVOLVED_MUSIC -trim DESIRED_AUDIO_LEN_IN_S
```

### Analyse RIR dataset 
To generate the true t60, c50, c80 and drr values from your RI dataset, run :
```console
python acoustic_param_ds_maker.py -rirDir YOUR_RIR_DIR/ -outDir WHERE_YOU_WANT_YOUR_TRUE_VALUES
```
It will create subdirectories for each parameter and pickle files with the name from the original rir.

