#!/usr/bin/env python
'''
File: preprocessing_ECAPA-TDNN.py
Author: Daniela Wiepert
Date: 10/2021
Sources: 
    https://gist.github.com/sotelo/be57571a1d582d44f3896710b56bc60d
Normalize, resample/convert channels, remove silences from audio files and then extract features for TIMIT
'''

# built-in
import os
import argparse
import pandas as pd 
import glob
import numpy as np
import shutil
from pathlib import Path

# third-party
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_leading_silence, detect_nonsilent
from python_speech_features import mfcc
from python_speech_features import delta
#from python_speech_features import logfbank
import scipy.io.wavfile as wav

# same directory
from silence import silence_removal
from get_ECAPA_TDNN_embeddings import generate_embeddings

## HELPER FUNCTIONS
def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def convert(sound, set_rate, set_channel):
    '''
    Resample audio and convert number of channels

    Input: sound - pydub AudioSegment
           set_rate - rate to resample to (int)
           set_channel - # of channels to convert to (int)
    Output: resampled/converted pydub AudioSegment
    '''
    #resample if specified
    if set_rate != sound.frame_rate:
        sound = sound.set_frame_rate(set_rate)

    # convert # channels
    if set_channel != sound.channels:
        sound = sound.set_channels(set_channel)

    return sound

### FROM: https://gist.github.com/sotelo/be57571a1d582d44f3896710b56bc60d
def remove_beg_end_silences(sound,silence_threshold=-50.0, chunk_size = 10):
    '''
    take pydub AudioSegment (sound), silence threshold in dB, and chunk size in ms and 
    find/trim audio to remove beginning and end silences

    Inputs: sound - pydub AudioSegment
            silence_threshold - float, dB, default = -30.0 dB - MUST BE A FLOAT
            chunk_size - int, ms, default = 10 ms
    Outputs: trimmed_sound, pydub AudioSegment with beginning and end silences removed
    '''
    # find where beginning silence ends
    start_trim = detect_leading_silence(sound,silence_threshold,chunk_size)
    # find where ending silence starts
    end_trim = detect_leading_silence(sound.reverse(),silence_threshold,chunk_size)

    # TRIM SOUND
    duration = len(sound)
    trimmed_sound = sound[start_trim:duration-end_trim]
    
    return trimmed_sound

def save_embedding(file, input_dir, feature_dir):
    '''
    Generate and save embedding as a .npy file
    '''
    embeddings = generate_embeddings([file]) 
    if embeddings[0] == []:
        print('No embedding')
    else:
        np.save(file.replace(input_dir, feature_dir).replace('.wav','.npy'), embeddings[0])
        print('Feature saved')

    return None

def audio_pipeline(to_process, input_dir, output_dir, feature_dir='', set_rate=16000, set_channel=1, remove_all=False, split=False, silence_threshold=-30.0, skip_features=False):
    '''
    Remove either beginning/ending silences or all silences from audio after resampling/converting channels and normalizing audio.
    Can specify which tasks to process from your input_dir

    Input: to_process - list of strings containing files to process
           input_dir - path to input dir (string)  
           output_dir - path to output dir (string)
           set_rate - sample rate to resample to (int)
           set_channel - channel to convert to (int)
           remove_all - boolean indicating whether to just remove beginning and ending silence or all silence
           split - boolean indicating whether to split on silence or remove and write to single file
           silence_threshold - threshold for silence in dB (float)
           skip_features - boolean indicating whether to skip feature generation 
           feature_dir - path to feature dir (string)
    Output: None, exports processed audio to output directory
    '''
    count = 0
    for f in to_process:

        count +=1
        output_f = f.replace(input_dir, output_dir).replace('.m4a','.wav')
        h= os.path.basename(f).split("_")[0]
        if not os.path.exists(output_f) and h != 'bkcywuk' and h != 'y3snbtl' and h != 'od47je7' and h != 'xdq7riq' and h != 'sy1al7n' and h != 'tmp.wav':
        #print(count)
    
        # read in audio data
            sound = AudioSegment.from_file(f) #,format="wav")

            #resample + channel conversion
            sound = convert(sound, set_rate, set_channel)

            #normalize
            sound = normalize(sound)

            if not remove_all:
                # remove beginning and end silences for all cases
                trimmed_sound = remove_beg_end_silences(sound,silence_threshold)

                #export AudioSegment to new wav file in output directory
                trimmed_sound.export(output_f, format="wav")

            else:
                # export the resampled/converted to mono/normalized sound to temporary file
                sound.export('.\\tmp.wav',format='wav')
                
                # remove silence
                silence_removal('tmp.wav', '.\\', os.path.split(output_f)[0], os.path.split(output_f)[1], False, split)

                # remove temporary file
                os.remove('tmp.wav')
        print('+ preprocessed: ' + f + ' (' + str(count) + ' of ' + str(len(to_process)) + ')')

        #f = f.replace('.m4a','.wav')
        if not skip_features and os.path.exists(output_f):
            assert feature_dir != '', 'feature_dir not given'
            feature_f = output_f.replace(output_dir, feature_dir).replace(".wav",".npy")
            if not os.path.exists(feature_f):
                save_embedding(output_f, output_dir, feature_dir)
            else: 
                print('feature already generated')
                    
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_dir", required= True, help="specifies input directory containing speech files")
    parser.add_argument("-o","--output_dir", required=True, help="specifies output directory for prepped files")
    parser.add_argument("-f","--feature_dir", default='', help="specifies directory to save features to")
    parser.add_argument("-t", "--to_process", default='', help="specify files to process if not wanting to use an entire dataset")
    parser.add_argument("-s", "--silence_threshold", type=float, default=-30.0, help="specify silence threshold for removing beginning and end silences")
    parser.add_argument("-sr", "--set_rate", default=16000, help="specify sample rate to resample to")
    parser.add_argument("-sc", "--set_channel", default=1, help="specify number of channels to convert to (1: mono, 2: stereo)")
    parser.add_argument("--remove_all", default=False, action='store_true',help="specify whether to remove all silences or just beggining and end silence")
    parser.add_argument("--split", default=False, action="store_true", help="specify whether to split on silence or keep as one file")
    parser.add_argument("--skip_features", default=False, action="store_true", help="specify whether to skip feature generation")
    args = parser.parse_args()

    if not args.skip_features:
        assert args.feature_dir != '', 'Feature dir not given'
    
    if args.to_process == '':
        if 'vox2' in args.input_dir:
            files = list(Path(args.input_dir).rglob("*.m4a")) 
        else:
            files = list(Path(args.input_dir).rglob("*.wav")) 
            files = [str(item) for item in files]

    else:
        with open(args.to_process, 'r') as output:
            files = output.readlines()
        files = [f.replace("\n","") for f in files]
        temp_files = [f.replace('.m4a','.wav') for f in files]

    # create output directory if it doesn't already exists
    if not os.path.exists(args.output_dir):
        shutil.copytree(args.input_dir, args.output_dir,ignore=ig_f)
        done_files = []
    else:
        done_files = list(Path(args.output_dir).rglob("*.wav")) 
        done_files = [str(item).replace(args.output_dir,args.input_dir) for item in done_files]

        # if 'vox2' in args.input_dir:
        #     done_files = [item.replace('.wav', '.m4a') for item in done_files]
        
    if not os.path.exists(args.feature_dir):
        shutil.copytree(args.input_dir, args.feature_dir,ignore=ig_f)
        done_feats = []
    else:
        done_feats = list(Path(args.feature_dir).rglob("*.npy")) 
        done_feats = [str(item).replace(args.feature_dir,args.input_dir).replace(".npy",".wav") for item in done_feats]

        # if 'vox2' in args.input_dir:
        #     done_feats = [item.replace('.wav', '.m4a') for item in done_feats]

    #once feature generated, check that done_feats also works
    if 'vox2' in args.input_dir:
        to_process = [f for f in temp_files if f not in done_files or f not in done_feats]
        to_process = [f.replace('.wav','.m4a') for f in to_process]
    else: 
        to_process = [f for f in files if f not in done_files or f not in done_feats] 

    if to_process != []:
    # run pre=processing 
        audio_pipeline(to_process, args.input_dir, args.output_dir, args.feature_dir, args.set_rate, args.set_channel, args.remove_all, args.split, args.silence_threshold, args.skip_features)
    else:
        print('All files already processed')
if __name__ == "__main__":
    main()