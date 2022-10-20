#!/usr/bin/env python
'''
File: combined_train-test_split.py
Author: Daniela Wiepert
Date: 10/14/2021

Create a csv file containing file names, speaker ids, and whether the file is a training(enrollment) or testing file. 
Can specify how many unique speakers in each set, as well as the overlap between each set.
'''

#Imports
import os
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
import random

def read_txt(txt_file):
    '''
    Read in train/test/overlap data from txt 
    Inputs: txt_file - path to txt file (str)
    Outputs: train - array with type of sentence/task for train set
             test - array with type of sentence/task for test set
             overlap - array with type of sentence/task for overlap set
    '''
    train = []
    test = []
    overlap = []

    read = open(txt_file,"r")
    read = read.read().splitlines()
    for l in read: 
        s = l.split(" ")
        if s[0] == "train":
            train.extend(s[1:])
        if s[0] == "test":
            test.extend(s[1:])
        if s[0] == "overlap":
            overlap.extend(s[1:])

    # if overlap not specified, default to test
    if overlap == []:
        overlap = test

    return train, test, overlap

def prep_mayo_speech(tasks, all_files, speech_df, num_train_speakers, num_test_speakers, num_overlap, data):
    '''
    Prepare train/test ids and tasks to use for mayo speech data
    
    Inputs: tasks - path to txt file with task data (str)
            all_files - string array of all_files to use
            speech_df - dataframe of dataset description for speech data
            num_train_speakers - number of train speakers (int)
            num_test_speakers - number of test speakers (int)
            num_overlap - number of overlapping speakers (int)
    Outputs: train_task - string array for tasks to use in train
             test_task - string array for tasks to use in test
             overlap_task - string array for tasks to use in overlap
             train_id - string array of speaker IDs for train
             test_id - string array of speaker IDs for test
    '''
    train, test, overlap = read_txt(tasks)
    train_task = {}
    for t in train:
        temp = t.split("-")
        if temp[0] in train_task:
            temp2 = train_task.get(temp[0])
            temp2.append("-".join(temp[1:]))
            train_task[temp[0]] = temp2
        else:
            train_task[temp[0]] = ["-".join(temp[1:])]
    test_task = {}
    for t in test:
        temp = t.split("-")
        if temp[0] in test_task:
            temp2 = test_task.get(temp[0])
            temp2.append("-".join(temp[1:]))
            test_task[temp[0]] = temp2
        else:
            test_task[temp[0]] = ["-".join(temp[1:])]
    overlap_task = {}
    for t in overlap:
        temp = t.split("-")
        if temp[0] in overlap_task:
            temp2 = overlap_task.get(temp[0])
            temp2.append("-".join(temp[1:]))
            overlap_task[temp[0]] = temp2
        else:
            overlap_task[temp[0]] = ["-".join(temp[1:])]

    speech_train = []
    speech_test = []
    speech_overlap = []
    for s in list(all_files.get(data).keys()):
        task = list(set(speech_df.loc[speech_df["speakerID"] == int(s)]["task"].tolist()))
        if [t for t in task if check_task(train_task, task=t)]:
                speech_train.append(s)
        if [t for t in task if check_task(test_task, task=t)]:
            speech_test.append(s)
        if [t for t in task if check_task(overlap_task, task=t)]:
            speech_overlap.append(s)

    print('Max Overlap: ' + str(len([f for f in speech_overlap if f in speech_train])))
    all_id = list(set(speech_train + speech_test + speech_overlap))
    assert num_train_speakers + num_test_speakers - num_overlap <= len(all_id), "Not enough speakers available with current selected tasks in Mayo Speech"
    assert num_train_speakers <= len(speech_train), "Not enough speakers available with current selected tasks in Mayo Speech"
    assert num_test_speakers <= len(speech_test), "Not enough speakers available with current selected tasks in Mayo Speech"
    assert num_overlap <= len(speech_overlap),"Not enough speakers available with current selected tasks in Mayo Speech"
    

    test_id = []
    train_id = []
    overlap_id = []

    for id in set(speech_overlap):
        if id in speech_test and id in speech_train: 
            if len(overlap_id) < num_overlap:
                overlap_id.append(id)
    for id in set(speech_test):
        if id not in overlap_id:
            if len(test_id) < num_test_speakers-num_overlap:
                test_id.append(id)
    for id in set(speech_train):
        if id not in overlap_id and id not in test_id:
            if len(train_id) < num_train_speakers - num_overlap:
                train_id.append(id)
  


    assert num_overlap == len(overlap_id), 'error in id selection for mayo-speech'
    assert num_test_speakers-num_overlap == len(test_id), 'error in id selection for mayo-speech'
    assert num_train_speakers-num_overlap == len(train_id), 'error in id selection for mayo-speech'


    train_id = train_id + overlap_id
    test_id = test_id + overlap_id
    return train_task, test_task, overlap_task, train_id, test_id

def get_file_dict(input_dir, data, test_data, num_train_speakers, num_test_speakers, num_overlap, speech_df=''):
    '''
    Generate a file dictionary to store data:list of file paths per id 

    Inputs: input_dir - string array of paths input directories
            data - string array of data types
            test_data - string indicating data type for test set
            num_train_speakers - int array for number of speakers per data type in train set
            num_test_speakers - int for number of speakers in test set
            num_overlap - int for number of speakers overlapping train and test set
            speech_df - if using mayo-speech data, data frame for dataset description, default=''
    Outputs: all_files - dictionary of the following format {'data-type':{'id':['file_path',...]}}
    '''
    
    all_files = {} #dictionary will store files as {"data-type":{"id":["file_path",...]}}
    total = 0 
    for i in range(len(input_dir)):

        # GET ALL NPY_FILES IN INPUT_DIR
        # all_npy = list(Path(input_dir[i]).rglob("*.wav"))
        # if all_npy == []:
        print('loading files from ' + input_dir[i] + '...')
        all_npy = list(Path(input_dir[i]).rglob("*.npy"))
        print('finished')
        all_npy = [str(item) for item in all_npy] #convert from Path objects to string paths
    
        random.shuffle(all_npy) #shuffle the file order to promote different speakers in different train/test splits
        
        # GET ID PER FILE
        print('getting speaker ids...')
        if data[i] == "vox1" or data[i] == 'vox2' or data[i] == "vox" or data[i] == "vox-test": 
            all_id = [str(os.path.basename(os.path.dirname(os.path.dirname(all_npy[p])))) for p in range(len(all_npy))]
        elif data[i] == "mayo-speech" or "mayo-speech-split":
            assert isinstance(speech_df, pd.DataFrame), "Dataset not read in" #READ IN DATASET
            all_id = []
            for f in all_npy: 
                s = os.path.split(f)[1][:-4]
                s = s.split("_")[0]
                all_id.append(str(speech_df.loc[speech_df['file_name'] == s]["speakerID"].item()))
        print('finished')
        cur_total = len(set(all_id)) # amount of speakers from current data type
        
        if data[i] in test_data:
            assert (num_train_speakers[i] + num_test_speakers[test_data.index(data[i])] - num_overlap[test_data.index(data[i])]) <= cur_total, "Not enough speakers available for train/test split"
        else:
            assert num_train_speakers[i] <= cur_total, "Not enough speakers available for train/test split"

        # create {id: [files]} dictionary
        files = {}
        for j in range(len(all_npy)):
            id = all_id[j]
            if not id in files:
                files[id] = [all_npy[j]]
            else:
                temp = files.get(id)
                temp.append(all_npy[j])
                files[id] = temp
        all_files[data[i]] = files
        total += cur_total
    
    return all_files

def check_task(task_list, f_name='', speech_df='', task=''):
    '''
    Check whether a task is of the selected tasks for mayo speech data 
    
    Inputs: task_list - list of selected tasks (str)
            f_name - str file path (default '' if a task is already given)
            speech_df - if using mayo-speech data, data frame for dataset description, default=''
            task - task to check (str), default=''
    Output: True/False based on whether the task is a selected task
    '''
    if task == '':
        assert f_name != '' and isinstance(speech_df,pd.DataFrame)
        s = f_name[:-4].split("_")[0]
        task = speech_df.loc[speech_df["file_name"] == s]["task"].item().lower()
        #print(task)
    if '' in task_list:
        exclude = task_list.get('')
    else:
        exclude = {}

    s_task = task.split("-")
    if len(s_task) == 1:
        s_task.append("*")
    if s_task[0] in task_list:
        sub = task_list.get(s_task[0])
        if s_task[1] in sub:
            #print(task)
            return True 
        if sub == ['*']: 
            if task in exclude:
                return False
            #print(task)
            return True
    return False

def select_train(data, all_files, num_train_speakers, train_task=[], speech_train = [], speech_df = '',vox_total=[]):
    '''
    Select files for train set
    
    Inputs: data - string array of data types
            all_files - dictionary storing datatype: id:files associated with id
            num_train_speakers - int array for number of speakers per data type in train set
            train_task - tasks to use if using mayo-speech data (default = [])
            speech_train - train IDs to use if using mayo-speech data (default = [])
            speech_df - if using mayo-speech data, data frame for dataset description, default=''
            vox_total - how many videos to use for vox train/test/overlap
    Outputs: dictionary containing file list, array indicating whether file is part of train set, array indicating whether file is part of test set, speaker ID list, data type list
    '''
    files = []
    train = []
    test = []
    speaker_id = []
    data_type = []
    remaining_videos = {}

    # go through all files for each test speaker and randomly select 1 file from the specified set (sa, sx, si)
    #test_id = ids[args.num_train_speakers-args.num_overlap:args.num_train_speakers-args.num_overlap+args.num_test_speakers]
    
    for i in range(len(data)):
        d = data[i]
        curr_files = all_files.get(d)
        if d == "mayo-speech" or d == "mayo-speech-split":
            assert speech_train != [], 'Speaker IDs not given for Mayo Speech'
            train_id = speech_train
        elif d == "jrd":
            train_id = list(all_files.get(d).keys())
            if num_train_speakers[data.index('jrd')] == 1:
                train_ind = train_id.index('jrdn')
                train_id = [train_id[train_ind]]
        else:
            train_id = list(all_files.get(d).keys())
            random.shuffle(train_id)
            train_id = train_id[:num_train_speakers[i]]
            
            
            
        for t in train_id:
            f = curr_files.get(t)
            if "vox" in d:
                videos = list(set([str(os.path.basename(os.path.dirname(p))) for p in f]))
                random.shuffle(videos)
                rv = videos
                if vox_total[0] == "1":
                    videos = random.choices(videos, k=1)
                elif vox_total[0] == "half":
                    half = int(len(videos)/2)
                    videos = random.choices(videos, k=half)
        
                rv = [v for v in rv if v not in videos]
                
                if vox_total[1] == 'all' and vox_total[2] == 'all' and rv == []:
                    rv = videos
                elif vox_total[2] == '1' and rv == []:
                    rv= [videos[0]]
                    videos = videos[1:]
                
                remaining_videos[t] = rv
                

            for j in range(len(f)):
                f_name = os.path.split(f[j])[1]
                if d == "mayo-speech" or d == "mayo-speech-split":
                    assert train_task != [], 'train_task not given'
                    assert isinstance(speech_df, pd.DataFrame), 'speech_df not given'
        
                    if check_task(train_task, f_name, speech_df):
                            speaker_id.append(t)
                            files.append(f[j])
                            train.append(True)
                            test.append(False)
                            data_type.append(d)

                elif "vox" in d:
                    v = str(os.path.basename(os.path.dirname(f[j])))
                    if v in videos:
                        speaker_id.append(t)
                        files.append(f[j])
                        train.append(True)
                        test.append(False)
                        data_type.append(d)

                else:
                    speaker_id.append(t)
                    files.append(f[j])
                    train.append(True)
                    test.append(False)
                    data_type.append(d)
    
    print("# Train Speakers: " + str(len(set(speaker_id))))
    print('# Train Files: ' + str(len(files)))
    assert len(set(speaker_id)) == sum(num_train_speakers), 'error in number of speakers in training set'
    return {'files': files, 'speaker_id':speaker_id, 'train':train, 'test':test, 'data_type':data_type}, remaining_videos

def select_test(speaker_info, all_files, test_data, num_train_speakers, num_overlap, num_test_speakers, overlap_task = [], test_task = [], speech_test = [], speech_df = '', num_files=1, dif_speakers=False, vox_total=[],vox_all=False,remaining_videos=[]):
    '''
    Select files for test set
    Inputs: speaker_info - dictionary containing file list, array indicating whether file is part of train set, array indicating whether file is part of test set, speaker ID list, data type list
            all_files - dictionary storing datatype: id:files associated with id
            test_data - str indicating data type for test
            num_train_speakers - number of train speakers for test data (int)
            num_overlap - number of overlap train/test speakers (int)
            num_test_speakers - number of speakers in test set (int)
            overlap_task - tasks for overlap set if using mayo-speech (str list) (default = [])
            test_task - tasks for test set if using mayo-speech (str list) (default = [])
            speech_test - speaker ID list if using mayo-speech (str list) (default = [])
            speech_df - if using mayo-speech data, data frame for dataset description, default=''
            num_files - number of files to include in test-data
            dif_speakers - if num_files > 1, specify whether to treat each file as same speaker or different speakers
            vox_total - how many videos to use for vox 
            vox_all - specify whether to use all files for vox (boolean)
            remaining_videos - list of videos that have not been selected yet
    Outputs: updated dictionary containing file list, array indicating whether file is part of train set, array indicating whether file is part of test set, speaker ID list, data type list
    '''
    speaker_id = []
    files = []
    train = []
    test = []
    data_type = []

    if test_data == "mayo-speech" or test_data == "mayo-speech-split":
        assert speech_test != [], "speech_test not given"
        test_id = speech_test
    else:
        curr_id = list(set(speaker_info.get('speaker_id')))
        test_id = list(all_files.get(test_data).keys()) #[num_train_speakers-num_overlap:num_train_speakers-num_overlap+num_test_speakers]
        in_id = [t for t in test_id if t in curr_id]
        out_id = [t for t in test_id if t not in curr_id]
        test_id = in_id[:num_overlap] + out_id[:num_test_speakers-num_overlap]
    
    test_files = all_files.get(test_data)
    
    for t in test_id:
        f = test_files.get(t)
        random.shuffle(f)
        if "vox" in test_data:
            if t in remaining_videos:
                videos = remaining_videos.get(t) #if it's all 
                if vox_total[2] == "1":
                    videos = random.choices(videos, k=1)
            else:
                videos = list(set([str(os.path.basename(os.path.dirname(p))) for p in f]))
                if vox_total[1] == "1":
                    videos = random.choices(videos, k=1)
                if vox_total[1] == "half":
                    videos = random.choices(videos,k=int(len(videos)/2))
        count = 0 
        completed = False
        for i in range(len(f)):
            f_name = os.path.split(f[i])[1]
            if vox_all and "vox" in test_data:
                v = str(os.path.basename(os.path.dirname(f[i])))
                if v in videos:
                    speaker_id.append(t)
                    files.append(f[i])
                    train.append(False)
                    test.append(True)
                    data_type.append(test_data)
                    
            elif count < num_files: #not completed:         

                if test_data == "mayo-speech" or test_data == "mayo-speech-split":
                    assert overlap_task != [], 'overlap_task not given'
                    assert test_task != [], 'test_task not given'
                    if t in speaker_info.get('speaker_id'):
                        if check_task(overlap_task, f_name, speech_df):
                            if num_files > 1 and dif_speakers:
                                speaker_id.append(t+"-"+str(count))
                            else:
                                speaker_id.append(t)
                            files.append(f[i])
                            train.append(False)
                            test.append(True)
                            data_type.append(test_data)
                            count += 1
                            #completed = True

                    elif check_task(test_task, f_name, speech_df):
                        if num_files > 1 and dif_speakers:
                            speaker_id.append(t+"-"+str(count))
                        else:
                            speaker_id.append(t)
                        files.append(f[i])
                        train.append(False)
                        test.append(True)
                        data_type.append(test_data)
                        count += 1
                        #completed = True

                elif "vox" in test_data:
                    v = str(os.path.basename(os.path.dirname(f[i])))
                    if v in videos:
                        if num_files > 1 and dif_speakers:
                            speaker_id.append(t+"-"+str(count))
                        else:
                            speaker_id.append(t)
                        files.append(f[i])
                        train.append(False)
                        test.append(True)
                        data_type.append(test_data)
                        count += 1
                else:
                    if num_files > 1 and dif_speakers:
                        speaker_id.append(t+"-"+str(count))
                    else:
                        speaker_id.append(t)
                    files.append(f[i])
                    train.append(False)
                    test.append(True)
                    data_type.append(test_data)
                    count += 1
                    #completed = True
    if test_data != 'jrd':
        print('# Total Test speakers: ' + str(len(set(speaker_id))))
        print('# Overlap: ' + str(len([id for id in set(speaker_id) if id in set(speaker_info.get('speaker_id'))])))
        print('# Total files: ' + str(len(files)))
        assert len(set(speaker_id)) == num_test_speakers, "Error with selecting test speakers"
        assert len([id for id in set(speaker_id) if id in set(speaker_info.get('speaker_id'))]) == num_overlap, "Error in number of overlapping speakers"
    #temp = sum([id for id in set(speaker_id) if id in set(speaker_info.get('speaker_id'))])
    return {'files': speaker_info.get('files') + files, 'speaker_id':speaker_info.get('speaker_id') + speaker_id, 'train':speaker_info.get('train') + train, 'test': speaker_info.get('test') + test, 'data_type':speaker_info.get('data_type') + data_type}

def save_split(speaker_info, output_dir, data, total_train_speakers, num_test_speakers, num_overlap, dif_speakers, num_files,vox_total,vox_all):
    '''
    Save the train/test split as a csv file
    Inputs: speaker_info - dictionary containing file list, array indicating whether file is part of train set, array indicating whether file is part of test set, speaker ID list, data type list
            output_dir - path to output directory (str)
            data - list of all datatypes used (str list)
            total_train_speakers - total number of speakers in train set (int)
            num_test_speakers - number of speakers in test set (int)
            num_overlap - number of overlap train/test speakers (int)
            dif_speakers - boolean indicating whether files from same speaker in test were treated as different speakers
            num_files - files per speaker in test (int)
            vox_total - how many videos to use for vox 
            vox_all - specify whether to use all files for vox (boolean)
    Outputs: None
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    speaker_info_df = pd.DataFrame(speaker_info)
    new_data = []
    for d in data:
        if d == "vox1":
            new_data.append('v1')
        elif d == "vox2":
            new_data.append('v2')
        elif d == "vox":
            new_data.append('v')
        elif d == "vox-test":
            new_data.append('v-test')
        elif d == "mayo-speech":
            new_data.append('ms')
        else:
            new_data.append(d)

    if not vox_all:
        for i in range(len(vox_total)):
            if vox_total[i] == '1':
                vox_total[i] == '1s'

    csv_file = "_".join(new_data) 
    if vox_total != ['all','all','all']:
        csv_file += '_tr' + str(total_train_speakers) + "_" + vox_total[0] + '_te' + str(num_test_speakers) + '_' + vox_total[1] + '_o' + str(num_overlap)
    else:
        csv_file += '_tr' + str(total_train_speakers) + '_te' + str(num_test_speakers) + '_o' + str(num_overlap)
    if not vox_all:
        csv_file += '_s'
    if num_files >1:
        csv_file += "_f" + str(num_files)
    if dif_speakers:
        csv_file += "_diff"
    
    path = os.path.join(output_dir, csv_file)
    add_s = ''
    count = 0
    while os.path.exists(path+add_s+'.csv'):
        print('path already exists')
        count += 1
        add_s = '_' + str(count)
    path += add_s
    print('Output path: ' + path + '.csv')
    speaker_info_df.to_csv(path+'.csv',index=False)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", nargs="+", required=True, help="specifies input directory containing speech files")
    parser.add_argument("-o", "--output_dir", required=True, help="specify path to output directory")
    parser.add_argument("-x", "--num_train_speakers", nargs="+", default=[500,100,500], type=int, help="specify number of unique speakers to include in train set")
    parser.add_argument("-n", "--num_overlap",nargs="+", default=[20], type=int, help="specify number of speakers to have in both train and test")
    parser.add_argument("-y", "--num_test_speakers", nargs="+",default=[50], type=int, help="specify number of unique speakers to include in test set")
    parser.add_argument("--sentences",default='', help="specify txt file containing which sentence/task types to include in train and test set")
    parser.add_argument("--tasks", default='', help="specify txt file containing which tasks to use in train/test/overlap set")
    parser.add_argument("--dataset", default='', help="specify csv dataset description for mayo speech data")
    parser.add_argument("-d", "--data", nargs="+", required=True, help='specify type of data you are working with in order of the input dirs')
    parser.add_argument("-t", "--test_data", nargs="+", required=True, help="specify which data to use for test data")
    parser.add_argument("-f", "--num_files", default=1, type=int, help="specify number of files to include per speaker in test set")
    parser.add_argument("--dif_speakers", default=False, action='store_true', help="If num_files is > 1, specify whether to treat each file as a different speaker or as the same speaker in test")
    parser.add_argument("-v", "--vox_total", nargs="+",default=['all','1','1'], help="specify whether to use all,half,none for train/test with vox" )
    parser.add_argument("--vox_all", default=False,action='store_true')
    args = parser.parse_args()
    
    # CHECK THAT ALL ARGUMENTS ARE CORRECTLY INPUT
    assert len(args.input_dir) == len(args.data), "Data type not specified for each input dir"
    assert len(args.input_dir) == len(args.num_train_speakers), "Number of train speakers not specified for each input dir"
    assert (d in ['vox','vox1','vox2','vox-test','mayo-speech'] for d in args.data), "Invalid data type"
    assert len(args.test_data) == len(args.num_test_speakers), "Number of test speakers not specified for each test data"
    assert len(args.test_data) == len(args.num_overlap), "Number of overlap speakers not specified for each test data"
    assert len(args.vox_total) == 3, 'Did not give number of video selection for both vox train and test'
    v_in = [v for v in args.data if 'vox' in v]
    if v_in != []:
        assert (args.vox_total[0] in ['all','half','1'])
        assert (args.vox_total[1] in ['all','half','1'])
        assert (args.vox_total[2] in ['all','1'])
        
    #initialize
    train_task = []
    test_task = []
    overlap_task = []
    speech_df = ''
    speech_train = []
    speech_test = []
    speech_train2 = []
    speech_test2 = []

    if "mayo-speech" in args.data or "mayo-speech-split" in args.data:
        assert args.tasks != '' and args.dataset != '', "Task txt and dataset path not specified for Mayo Speech Data"
        # Read in dataset
        speech_df = pd.read_csv(args.dataset, keep_default_na=False)
        # Get dictionary with datatypes and files
    
    all_files = get_file_dict(args.input_dir, args.data, args.test_data, args.num_train_speakers, args.num_test_speakers, args.num_overlap, speech_df)
        # Read in tasks and get train/test ids
    if "mayo-speech" in args.data:
        train_task, test_task, overlap_task, speech_train, speech_test = prep_mayo_speech(args.tasks, all_files, speech_df, args.num_train_speakers[args.data.index("mayo-speech")], args.num_test_speakers[args.test_data.index("mayo-speech")], args.num_overlap[args.test_data.index("mayo-speech")], "mayo-speech") 
    if "mayo-speech-split" in args.data and not "mayo-speech" in args.data:
        # Read in tasks and get train/test ids
        train_task, test_task, overlap_task, speech_train, speech_test = prep_mayo_speech(args.tasks, all_files, speech_df, args.num_train_speakers[args.data.index("mayo-speech-split")], args.num_test_speakers[args.test_data.index("mayo-speech-split")], args.num_overlap[args.test_data.index("mayo-speech-split")],"mayo-speech-split")

    print('selecting train files...')
    speaker_info, remaining_videos = select_train(args.data, all_files, args.num_train_speakers, train_task, speech_train, speech_df, args.vox_total)
    print('finished')
    print('selecting test files...')

    if "mayo-speech-split" in args.test_data and len(args.num_test_speakers) > 1:
         i = args.test_data.index('mayo-speech-split')
         speaker_info = select_test(speaker_info, all_files, args.test_data[i], args.num_train_speakers[args.data.index(args.test_data[i])], args.num_overlap[i], args.num_test_speakers[i], overlap_task, test_task, speech_test, speech_df, args.num_files, args.dif_speakers, args.vox_total, args.vox_all, remaining_videos)
    else:
        for i in range(len(args.num_test_speakers)):
            speaker_info = select_test(speaker_info, all_files, args.test_data[i], args.num_train_speakers[args.data.index(args.test_data[i])], args.num_overlap[i], args.num_test_speakers[i], overlap_task, test_task, speech_test, speech_df, args.num_files, args.dif_speakers, args.vox_total,args.vox_all, remaining_videos)      
    print('finished')     
                
    # SAVE DATA
    print('saving data...')
    save_split(speaker_info, args.output_dir, args.data, sum(args.num_train_speakers), sum(args.num_test_speakers), sum(args.num_overlap), args.dif_speakers, args.num_files, args.vox_total,args.vox_all)
    print('finished')

if __name__ == "__main__":
    main()