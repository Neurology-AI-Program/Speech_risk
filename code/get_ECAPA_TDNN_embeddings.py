#!/usr/bin/env python
'''
File: get_ECAPA-TDNN_embeddings.py
Author: Daniela Wiepert
Date: 9/2021
Sources: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

Get ECAPA-TDNN embeddings using speechbrain/spkrec-ecapa-voxceleb

Install dependencies:
    pip install speechbrain 
    pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
'''

#IMPORTS
import os
import argparse
import numpy as np
import pandas as pd
import shutil
import torchaudio
from pathlib import Path
from speechbrain.pretrained import EncoderClassifier

def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def save_embeddings(input_dir, save_dir, embeddings_df, data_type, mac):
    '''
    Save embeddings 
    Inputs: input_dir - path to input directory (str)
            save_dir - path to directory where you should save embeddings (str)
            embeddings_df - dataframe containing embeddings and corresponding files
            data_type - specify dataset ("timit", "speech") (string)
            mac - boolean indicating whether files should be treated as mac/linux or windows files 
    '''
    # new_path = os.path.join(save_dir,'ECAPA-TDNN_embeddings')
    if data_type == "timit":
        if not os.path.exists(save_dir):
            shutil.copytree(input_dir,save_dir,ignore=ig_f)
        
        embeddings = embeddings_df["embedding"].tolist()
        files = embeddings_df['file_name'].tolist()
        if mac:
            files = ["/".join(f.split("/")[-4:]) for f in files]
        else:
            files = ["\\".join(f.split("\\")[-4:]) for f in files]

        for i in range(len(files)):
            print("Saved " + str(i+1) + " of " + str(len(files)))
            p = os.path.join(save_dir,files[i])
            p = p.replace("wav","npy")
            np.save(p,embeddings[i])
    
    return None

def get_embedding(audio_file, classifier):
    '''
    get ECAPA-TDNN embedding for an audio file

    Input: audio_file - path to wav file (string)
           classifier - EncoderClassifier object 
    Output: embedding - ECAPA-TDNN embedding
    '''
    #load audio
    signal, fs = torchaudio.load(audio_file)
    
    if signal.shape[1] <= 500:
        return []
    #generate embedding
    embedding = classifier.encode_batch(signal)

    return embedding.detach().numpy().squeeze()

def generate_embeddings(data):
    '''
    generate ECAPA-TDNN embeddings for all paths in data using classifier from speechbrain

    Input: data - list of file paths to audio data (.wav) (list of string)
    Output: embeddings - list of embeddings (list of arrays)
    '''
    #instantiate classifier
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    
    #get embeddings
    embeddings = []
    print('generating embeddings')
    for i in range(len(data)):
        print(str(i) + ' of ' + str(len(data)))
        embeddings.append(get_embedding(data[i], classifier))
    return embeddings

def run_generation(files, input_dir, save, save_dir, mac = False):
    '''
    Generate embeddings for all data files and create a dataframe with file name, 
    embedding, and information on whether the file is part of the training or testing set

    Input: files - list of all files to generate embeddings for
           input_dir - path to input director (string)
           save - boolean indicating whether to save embeddings
           save_dir - string path where embeddings should be saved
    Output: embeddings_df - dataframe containing embedding information (pd dataframe)                                                     
       
    '''

    #generate embeddings
    all_embeddings = generate_embeddings(files)

    #create dataframe
    embeddings_df = {'file_name':files, 'embedding':all_embeddings}
    embeddings_df = pd.DataFrame(data = embeddings_df)

    if save:
        save_embeddings(input_dir, save_dir, embeddings_df, "timit", mac)
    return embeddings_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_dir", required= True, help="specifies input directory containing speech files")
    parser.add_argument("--save", default=False, action="store_true", help="specify whether to save embeddings")
    parser.add_argument("-s", "--save_dir", default='./', help="specify where to save embeddings")
    parser.add_argument("-d", "--data_type", required=True, choices=["timit"], help="specify which data is being used")
    parser.add_argument("--mac", default=False, action="store_true", help="specify whether files are in mac or windows format")
    args = parser.parse_args()


    result = list(Path(args.input_dir).rglob("*.wav")) # select all train wavs
    result_str=[str(item) for item in result]
    embeddings_df = run_generation(result_str, args.input_dir, args.save, args.save_dir, args.mac)
    

if __name__ == "__main__":
    main()



