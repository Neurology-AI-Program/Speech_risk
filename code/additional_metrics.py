#!/usr/bin/env python
'''
File: additional_metrics.py
Author: Daniela Wiepert
Date: 10/19/2022

Calculate additional metrics from PLDA output
'''

import argparse
import os
import json
import pandas as pd
import glob
import numpy as np 

from plda import prediction_breakdown

def load_data(file, th_func):
    '''
    Load PLDA data from a file

    Inputs: file - file name containing PLDA output
    
    Outputs: actual - matrix of actual matches
             predicted - matrix of original predicted matches
             scores - np matrix of PLDA scores
             modelset - np array containing list of all speakers the PLDA was trained on
             segset - np array containing list of all the speakers the PLDA was tested on 
             train_speakers - np array containing order of train speakers for indexing of actual/predicted matrices
             test_speakers - np array containing order of test speakers for indexing of actual/predicted matrices
             th - threshold for results

    '''
    obj = open(file)
    data = json.load(obj)
    obj.close()

    actual = np.asarray(data['actual_labels'])
    predicted = data['predicted_labels']
    if th_func == 'eer' or th_func == 'mindcf':
        predicted = predicted[th_func]
    predicted = np.asarray(predicted)
    plda_scores = data['plda_scores'][0]
    scores = np.asarray(plda_scores['scores'])
    segset = np.asarray(plda_scores['segset']) #test speakers 
    modelset = np.asarray(plda_scores['modelset']) #train speakers
    train_speakers = np.asarray(data['train_speakers'])
    test_speakers = np.asarray(data['test_speakers'])
        
    th = data['average_th']
    if th_func == 'eer' or th_func == 'mindcf':
        th = th[th_func]
    
    return actual, predicted, scores, modelset, segset, train_speakers, test_speakers, th

def score_to_predicted(matches,pred_new, modelset, segset, train_speakers, test_speakers):
    '''
        The order of speakers for indexing is not the same between the PLDA scores and the actual labels. This function
        translates from PLDA speaker order to actual speaker order so that actual and predicted arrays can be indexed 
        the same way

        Inputs: matches - np array with indices where there are accepted matches from the thresholded PLDA scores
                pred_new - an empty np matrix of equivalent size to the actual label matrix
                modelset - np array containing list of all speakers the PLDA was trained on
                segset - np array containing list of all the speakers the PLDA was tested on 
                train_speakers - np array containing order of train speakers for indexing of actual/predicted matrices
                test_speakers - np array containing order of test speakers for indexing of actual/predicted matrices
    ''' 
    train_m = matches[0]
    test_m = matches[1]
    for m in range(len(train_m)):
        #get IDs
        id1 = modelset[train_m[m]]
        id2 = segset[test_m[m]]

        # Assuming enrol_id and test_id are unique
        x = int(np.where(train_speakers == id1)[0][0])
        y = int(np.where(test_speakers == id2)[0][0])

        pred_new[x,y] = True
    
    return pred_new

def rank1_counts(files, wd, th_func):
    '''
    Get only rank 1 counts (best match per speaker that is above threshold)

    Inputs: files - list of all files containing PLDA outputs for a specific experiment
            wd - current working directory of the files (for saving out rank1 counts)
            th_func - string indicating which threshold function to get counts and precision for

    Outputs: None, creates a csv file with rank1 count information
    '''
    #initialize
    TA = []
    FA = []
    TR = []
    FR = []
    prec = []

    for i in range(len(files)):
        print(i)
        print(files[i])
        #load data
        actual, predicted, scores, modelset, segset, train_speakers, test_speakers, th = load_data(files[i], th_func)

        #threshold data
        result = (scores > th)

        #initialize empty matrix for rank1 results
        rank1 = np.full(np.shape(scores), False)
        pred_rank1 = np.full(np.shape(actual), False) 

        #for each test speaker
        for s in test_speakers:
            y = np.where(segset==s)
            col = np.squeeze(scores[:,y])
            x = np.argmax(col) #select the best match by finding the highest score in the column

            if result[x,y]: #if the best match is above threshold
                rank1[x,y] = True  #set true 
    

        #find predicted matches
        matches = np.where(rank1 == True)

        #translate to index mapping of actual/predicted arrays
        pred_rank1 = score_to_predicted(matches, pred_rank1, modelset, segset, train_speakers, test_speakers)
       
        # get confusion matrix counts
        ta, fa, tr, fr = prediction_breakdown(actual,pred_rank1)

        #calculate precision
        rank1_prec = ta / (ta+fa)

        #append
        TA.append(ta)
        FA.append(fa)
        TR.append(tr)
        FR.append(fr)
        prec.append(rank1_prec)

    #SAVE TO CSV
    out = pd.DataFrame({'files': files, 'TA': TA, 'FA': FA, 'TR': TR, "FR": FR, 'prec': prec})
    out_name = os.path.join(wd,"rank1_counts")
    out_name = out_name + "_"  + th_func + '.csv'
    out.to_csv(out_name)

def known_overlap_counts(files, wd, th_func, n=5):
    '''
    Get only the top N known overlap counts

    Inputs: files - list of all files containing PLDA outputs for a specific experiment
            wd - current working directory of the files (for saving out counts)
            th_func - string indicating which threshold function to get counts and precision for
            n - size of known overlap, default = 5

    Outputs: None, creates a csv file with known overlap count information
    '''

    #initialize
    TA = []
    FA = []
    TR = []
    FR = []
    prec = []

    for i in range(len(files)):
        print(i)
        print(files[i])
        
        #load data
        actual, predicted, scores, modelset, segset, train_speakers, test_speakers, th = load_data(files[i], th_func)

        #for known overlap, set all values below threshold to negative infinity
        scores[scores < th] = -np.inf
       
    
        #flatten scores to find the top N best matches across all scores
        scores_flat = scores.flatten()
        
        #initialize empty arrays for results
        results = np.full(np.shape(scores), False)
        pred_known = np.full(np.shape(scores), False)
        #scores_flat = new_scores.flatten()

        #get the top N best matches (largest scores)
        index = np.argpartition(scores_flat, -n)[-n:]

        #get mapping between flattened and original score indicies
        index_array = np.arange(0,scores_flat.shape[0])
        ind_array = np.reshape(index_array, scores.shape)
        
        #mark True in the empty results array
        for ind in index:
            if scores_flat[ind] != 0:
                t = np.where(ind_array == ind)
                x = t[0]
                y = t[1]
                results[x,y] = True
        
        #find where the matches are 
        matches = np.where(results == True)

        #translate to index mapping of actual/predicted arrays
        pred_known = score_to_predicted(matches, pred_known, modelset, segset, train_speakers, test_speakers)

        #get confusion matrix counts
        ta, fa, tr, fr = prediction_breakdown(actual,pred_known)
        #calculate precision
        rank1_prec = ta / (ta+fa)

        #add to list
        TA.append(ta)
        FA.append(fa)
        TR.append(tr)
        FR.append(fr)
        prec.append(rank1_prec)

    #SAVE TO CSV
    out = pd.DataFrame({'files': files, 'TA': TA, 'FA': FA, 'TR': TR, "FR": FR, 'prec': prec})
    out_name = os.path.join(wd, str(n) + "_known_overlap_counts")
    out_name = out_name + "_"  + th_func + '.csv'
    out.to_csv(out_name)

def acceptance_overlap(files, wd, th_func):
    '''
    Count the number of false acceptances that resulted from non-overlapping speakers vs overlapping speakers. 

    Inputs: files - list of all files containing PLDA outputs for a specific experiment
            wd - current working directory of the files (for saving out counts)
            th_func - string indicating which threshold function to get counts and precision for

    Outputs: None, creates a csv file with non-overlap acceptance information
    '''
    total_matches = []
    total_FA = []
    FA_non_o = []
    FA_o = []

    for f in files:
        print(f)
        actual, predicted, scores, modelset, segset, train_speakers, test_speakers, th = load_data(f, th_func)
        
        #actual = np.array([item for sublist in actual for item in sublist])
        #predicted = np.array([item for sublist in predicted for item in sublist])
        
         # Total Acceptances (any speaker match accepted by the PLDA)
        matches = np.argwhere(predicted) #find ind of predicted acceptances 
        total_matches.append(len(matches)) #total acceptances

        # Total FA (any false speaker match accepted by the PLDA)
        total_FA.append(np.sum(np.logical_and(np.invert(actual), predicted)))
        
        #FA non overlapping speaker (speaker without a true match is matched to a speaker)
        non_overlap_speakers = [i for i in range(len(test_speakers)) if test_speakers[i] not in train_speakers]
        predicted_non_overlap = predicted[:, non_overlap_speakers]
        actual_non_overlap = actual[:, non_overlap_speakers]
        FA_non_o.append(np.sum(np.logical_and(np.invert(actual_non_overlap), predicted_non_overlap)))
       
        #FA overlapping speakers (speaker with a true match is matched to the wrong speaker)
        overlap_speakers = [i for i in range(len(test_speakers)) if test_speakers[i] in train_speakers]
        predicted_overlap = predicted[:, non_overlap_speakers]
        actual_overlap = actual[:, non_overlap_speakers]
        FA_o.append(np.sum(np.logical_and(np.invert(actual_overlap), predicted_overlap)))

    #SAVE TO CSV
    out = pd.DataFrame({"files": files, "total_matches": total_matches, "total_FA": total_FA, "FA_non_overlap":FA_non_o, "FA_overlap":FA_o})
    out_name = os.path.join(wd, "non_overlap_acceptances")
    out_name = out_name + "_"  + th_func + '.csv'
    out.to_csv(out_name)

def main():
    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser()
    #required arguments
    parser.add_argument("-d", "--data_dirs", nargs="+", default =['train2/1000', 'train2/4000', 'test/150', 'train2/7205', 'test/500', 'test/1000'], help="specify all directories containing PLDA outputs you want additional metrics for")
    parser.add_argument("-r", "--root_dir", default='/Users/m144443/Documents/mayo-speech_data/outputs/', help="specify root directory of all data dirs")
    parser.add_argument("-e", "--reg_exp", default="v_tr*_all_te*_1_o5_s_*r500_s100data.json", help="specify a regular expression to use for selecting which files to calculate additional metrics for")
    #optional arguments
    parser.add_argument("-t", "--th_func", default="mindcf", choices=["mindcf", "eer", "eer_only", "mindcf_only", "manual"], help="specify which threshold function for loading data")
    parser.add_argument("-c", "--to_calculate",nargs='+', default=['rank1', 'known','overlap'], choices=['rank1','known','overlap'],  help="specify which metrics to calculate [rank1','known','overlap']")
    parser.add_argument("-n", "--known_overlap", default=5, help="specify known overlap")
    args = parser.parse_args()

    assert args.th_func in ["mindcf", "eer", "eer_only", "mindcf_only", "manual"], 'invalid threshold function'
    assert all(x in ['rank1','known','overlap'] for x in args.to_calculate), 'invalid calculation selected'

    for d in args.data_dirs:
        wd = os.path.join(args.root_dir, d)
        assert os.path.exists(wd), 'current directory does not exist: ' + wd
        print('Current working directory: ' + wd)


        files = glob.glob(os.path.join(wd,args.reg_exp))
        
        if 'rank1' in args.to_calculate:
            rank1_counts(files, wd, args.th_func)
        
        if 'known' in args.to_calculate:
            known_overlap_counts(files, wd, args.th_func, args.known_overlap)
        
        if 'overlap' in args.to_calculate:
            acceptance_overlap(files, wd, args.th_func)

if __name__ == "__main__":
    main()
    