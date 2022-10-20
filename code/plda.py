#!/usr/bin/env python
'''
File: plda.py
Author: Daniela Wiepert, Hugo Botha
Date: 10/10/2022
Sources: 
    VoxCeleb PLDA speaker verification: https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/speaker_verification_plda.py
    Speech Brain fast_PLDA_scoring example: https://speechbrain.readthedocs.io/en/latest/API/speechbrain.processing.PLDA_LDA.html#module-speechbrain.processing.PLDA_LDA

Run PLDA speaker verification on TIMIT using ECAPA-TDNN embeddings 
'''

## IMPORTS
#built-in
import os
import argparse
import random
import json 
import csv
import time
#3rd party
import torch
import numpy as np
import pandas as pd
from speechbrain.processing.PLDA_LDA import *
from speechbrain.utils.metric_stats import *
#files
from get_ECAPA_TDNN_embeddings import run_generation

## HELPER FUNCTIONS
def accuracy(true_pos, false_pos, false_neg, true_neg):
    '''
    Calculate accuracy of PLDA scoring

    Inputs: true_pos - number of true positives (int)
            false_pos - number of false positives (int)
            false_neg - number of false negatives (int)
            true_neg - number of true negatives (int)
    Output: calculated accuracy (float)
    '''
    return (true_pos+true_neg)/(true_pos+false_pos+false_neg+true_neg)

def average_xvectors(speakers, sentence_speakers, xv):
    '''
    Average all embeddings for each speaker
    
    Inputs: speakers - list of unique speakers (list of strings of len # speakers)
            sentence_speakers - list of speaker for each sentence (list of strings of len # sentences)
            xv - numpy array containing all embeddings (# files x embedding dim (192))
    Outputs: numpy array containing averaged embeddings (# speakers x embedding dim (192))
    '''
    xv2 = []
    for s in speakers:
        indices = np.array([i for i, x in enumerate(sentence_speakers) if x == s]) #find index for all embeddings for a given speaker
        slice = xv[indices] #get embeddings for a given speaker
        xv2.append(np.average(slice,axis=0)) #average embeddings
    return np.stack(xv2)

def fbeta_measure(prec, rec, beta=1):
    '''
    Calculate f-beta score (default f1-score)
    Inputs: prec - precision (float)
            rec - recall (float)
            beta - default 1
    Outputs: calculated f-beta score
    '''
    return ((1 + beta**2) * prec * rec) / ((beta**2 * prec) + rec)


def matthews_cor_coef(true_pos, false_neg, false_pos,true_neg):
    '''
    Calculate Mathhews Correlation Coefficient
    Inputs: true_pos - number of true positives (int)
            false_pos - number of false positives (int)
            false_neg - number of false negatives (int)
            true_neg - number of true negatives (int)
    Output: calculated matthews correlation coefficient (float)
    '''
    return ((true_pos*true_neg)-(false_pos*false_neg))/np.sqrt((true_pos+false_pos)*(true_pos+false_neg)*(true_neg+false_pos)*(true_neg+false_neg))

def performance_metrics(actual, predicted, b=1):
    '''
    Calculate the different performance PLDA metrics (precision, recall, f-score, matthews correlation coefficient, accuracy)
    Inputs: actual - boolean array of actual labels
            predicted - boolean array of predicted labels from scoring
            b - beta for f-score, default=1
    Outputs: prec - calculated precision (float)
             rec - calculated precision (float)
             f_score - calculated f-beta score (float)
             confusion - confusion matrix (2x2 np array)
             mcc - calculated matthews correlation coefficient (float)
             acc - calculated accuracy (float)
    '''

    #get confusion matrix
    true_pos, false_pos, true_neg, false_neg = prediction_breakdown(actual, predicted) 
    confusion = np.array([[true_pos, false_neg],[false_pos, true_neg]])

    prec = precision(true_pos, false_pos) #calculate precision
    rec = recall(true_pos, false_neg) #calculate recall
    f_score = fbeta_measure(prec, rec, b) #calculate f-beta score
    mcc = matthews_cor_coef(true_pos, false_neg, false_pos, true_neg) #calculate matthews correlation coefficient
    acc = accuracy(true_pos, false_pos, false_neg, true_neg) #calculate accuracy
    fp_rate = false_pos / (false_pos + true_neg)
    return prec, rec, f_score, confusion, mcc, acc, fp_rate

def precision(true_pos, false_pos):
    '''
    Calculate precision
    Inputs: true_pos - number of true positives (int)
            false_pos - number of false positives (int)
    Outputs: calculated precision (float)
    '''
    return true_pos / (true_pos + false_pos)

def prediction_breakdown(actual, predicted):
    '''
    Get true/false positive and negatives
    Inputs: actual - boolean array of actual labels
            predicted - boolean array of predicted labels
    Outputs: true_pos - number of true positives (int)
             false_pos - number of false positives (int)
             false_neg - number of false negatives (int)
             true_neg - number of true negatives (int)
    '''
    true_pos = np.sum(np.logical_and(actual, predicted))
    false_pos = np.sum(np.logical_and(np.invert(actual), predicted))
    true_neg = np.sum(np.logical_and(np.invert(actual), np.invert(predicted)))
    false_neg = np.sum(np.logical_and(actual, np.invert(predicted)))

    return true_pos, false_pos, true_neg, false_neg

def random_xvectors(speakers, sentence_speakers, xv):
    ''''
    Get a random embedding for each speaker based on index x

    Inputs: speakers - list of unique speakers (list of strings of len # speakers)
            sentence_speakers - list of speaker for each sentence (list of strings of len # sentences)
            xv - numpy array containing all embeddings (# files x embedding dim (192))
    Outputs: numpy array containing randomly selected embedding (# speakers x embedding dim (192))
    '''
    xv2 = []
    for s in speakers:
        indices = np.array([i for i, x in enumerate(sentence_speakers) if x == s]) #find index for all embeddings for a given speaker
        slice = xv[indices] #get embeddings for a given speaker
        xv2.append(random.choice(slice)) #randomly select embedding
    return np.stack(xv2)

def recall(true_pos, false_neg):
    '''
    Calculate recall
    Inputs: true_pos - number of true positives (int)
            false_neg - number of false negatives (int)
    Outputs: calculated recall
    '''
    return true_pos / (true_pos + false_neg)

def select_xvectors(speakers, sentence_speakers, xv, x):
    ''''
    Select an embedding for each speaker based on index x

    Inputs: speakers - list of unique speakers (list of strings of len # speakers)
            sentence_speakers - list of speaker for each sentence (list of strings of len # sentences)
            xv - numpy array containing all embeddings (# files x embedding dim (192))
            x - index to select (int)
    Outputs: numpy array containing selected embedding (# speakers x embedding dim (192))
    '''
    xv2 = []
    for s in speakers:
        indices = np.array([i for i, x in enumerate(sentence_speakers) if x == s]) #find index for all embeddings for a given speaker
        slice = xv[indices] #get embeddings for a given speaker
        xv2.append(slice[x]) #randomly select embedding
    return np.stack(xv2)

def threshold_matrix(speaker1, speaker2, score, th, actual):
    '''
    Threshold matrix to determine predicted labels 
    Inputs: speaker1 - string array of speaker set 1
            speaker2 - string array of speaker set 2 
            score - scored plda 
            th - threshold
            actual - boolean array of actual labels
    Outputs: predicted - boolean array of predicted labels
    '''

    result = (score.scoremat > th)
    predicted = np.full(actual.shape, False)
    for i in range(actual.shape[0]):
        for j in range(actual.shape[1]):
            id1 = speaker1[i]
            id2 = speaker2[j]

        # Assuming enrol_id and test_id are unique
            x = int(numpy.where(score.modelset == id1)[0][0])
            y = int(numpy.where(score.segset == id2)[0][0])

            if result[x,y]:
                predicted[i,j] = True
                 
    return predicted
    
def get_embeddings(paths):
    '''
    Get ECAPA-TDNN embeddings for all files in 
    
    Inputs: paths - list of all file paths (list of strings)
    Outputs: xv - numpy array containing all embeddings (# files x embedding dim (192))
    '''
    #load the necessary embeddings
    xv = []
    for i in range(len(paths)):
        if (i % 1000) == 0:
            print(str(f"Done with {i} out of {len(paths)}"))

        temp = np.load(paths[i],allow_pickle=True) #load embedding
        xv.append(temp.squeeze())
    xv = np.stack(xv)

    return xv

def get_pos_neg_score(actual, speaker1, speaker2, score):
    '''
    Get positive and negative scores
    Inputs: actual - boolean array of actual labels
            speaker1 - string array of speaker set 1
            speaker2 - string array of speaker set 2 
            score - scored plda 
    Outputs: positive_scores - array containing all scores where actual label is true
             negative_scores - array containing all scores where actual label is false
    '''

    pos_ind = np.where(actual==True)
    positive_scores = get_score(pos_ind, speaker1,speaker2, score)

    neg_ind = np.where(actual==False)
    negative_scores = get_score(neg_ind, speaker1, speaker2, score)

    return positive_scores, negative_scores

def get_ref_database(speakers1, speakers2):
    '''
    Get reference database of actual labels
    Inputs: speaker1 - string array of speaker set 1
            speaker2 - string array of speaker set 2 
    Outputs: actual - boolean array of actual labels
    '''
    speakers1 = np.repeat(np.expand_dims(np.asarray(speakers1, dtype=object), axis=1), len(speakers2),axis=1)
    speakers2 = [s.split("-")[0] for s in speakers2]
    speakers2 = np.asarray(speakers2, dtype=object)
    actual = np.equal(speakers1,speakers2)

    return actual

def get_score(ind, speaker1, speaker2, score):
    '''
    Get score for given speaker1/speaker2 pairs
    Inputs: ind - list of i,j index pairs 
            speaker1 - string array of speaker set 1
            speaker2 - string array of speaker set 2 
            score - scored plda
    Outputs: scores - array of scores for speaker1/speaker 2 pairs
    '''
    scores = []
    for i in range(len(ind[0])):
        id1 = speaker1[ind[0][i]]
        id2 = speaker2[ind[1][i]]

        x = np.where(score.modelset == id1)[0][0]
        y = np.where(score.segset == id2)[0][0]
        scores.append(score.scoremat[x,y])
    return scores

def get_stat_ob(speakers, xv, sentences=[]):
    '''
    Generate a StatObject for PLDA 

    Inputs: speakers - list of speakers (list of strings)
            xv - numpy array containing all embeddings (# files x embedding dim (192))
            sentences - list of sentence identifiers (list of strings) (default = [])
    Outputs: StatObject
    '''
    N = len(speakers)
    modelset = np.array(speakers, dtype="|O")
    s = np.array([None] * N)
    stat0 = np.array([[1.0]] * N)
    if sentences != []:
        segset = np.array(sentences, dtype="|O")
        return StatObject_SB(modelset=modelset, segset=segset, start=s, stop=s, stat0=stat0, stat1=xv), modelset
    else:
        return StatObject_SB(modelset=modelset, segset=modelset, start=s, stop=s, stat0=stat0, stat1=xv), modelset

def get_threshold(speakers, sentence_speakers, xv, plda, n, subset,c_miss=1.0, c_fa=1.0, p_target=1.0, p_k=1):
    '''
    Get an average threshold for trained PLDA 

    Inputs: speakers - list of unique speakers (list of strings of len # speakers)
            sentence_speakers - list of speaker for each sentence (list of strings of len # sentences)
            xv - numpy array containing all embeddings (# files x embedding dim (192))
            plda - trained PLDA object
            n - number of times to run threshold calculation (int)
            subset - size of subset (int)
            c_miss - (minDCF specifc) cost assigned to a missing error (default 1.0)
            c_fa - (minDCF specifc) cost assigned to a false alarm (default 1.0).
            p_target - (minDCF specifc) prior probability of having a target (default 0.01).
            p_k - (fast_PLDA_Scoring specific) probability of having a known speaker for open-setidentification case (=1 for the verification task and =0 for the closed-set case)
    Outputs: avg_eer_th - avg calculated threshold for eer (float)
             threshold_eer - list of eer thresholds for each subset (list of floats)
             eer - list of eer results for each subset (list of floats)
             avg_eer - avg eer result (float)
             avg_mindcf_th - avg calculated threshold for minDCF (float)
             threshold_mindcf - list of minDCF thresholds for each subset (list of floats)
             mindcf - list of minDCF results for each subset (list of floats)
             avg_mindcf - avg minDCF result (float)
    '''
    #subset = int(len(speakers)/100)
    # random_xv1 = average_xvectors(speakers, sentence_speakers, xv) #select random x-vector to get one vector per speakers (# speakers x embedding dim)
    # random_xv2 = average_xvectors(speakers, sentence_speakers, xv) #select rancom x-vector to get one vector per speakers (# speakers x embedding dim)

    random_xv1 = random_xvectors(speakers, sentence_speakers, xv) #select random x-vector to get one vector per speakers (# speakers x embedding dim)
    random_xv2 = random_xvectors(speakers, sentence_speakers, xv) #select rancom x-vector to get one vector per speakers (# speakers x embedding dim)

    threshold_eer = []
    threshold_mindcf = []
    eer = []
    avg_eer = None
    min_dcf = []
    avg_min_dcf = None
    for j in range(n):
        print('Run ' + str(j) + ' of ' + str(n))
        inds1 = random.sample(range(1, len(speakers)), subset)
        inds2 = random.sample(range(1, len(speakers)), subset)
        stat1, set1 = get_stat_ob(np.array([speakers[i] for i in inds1]), np.array([random_xv1[i,:] for i in inds1]))
        stat2, set2 =  get_stat_ob(np.array([speakers[i] for i in inds2]), np.array([random_xv2[i,:] for i in inds2]))
        actual = get_ref_database([speakers[i] for i in inds1], [speakers[i] for i in inds2])
        ndx = Ndx(models=set1, testsegs=set2)

        scores_plda = fast_PLDA_scoring(stat1, stat2, ndx, plda.mean, plda.F, plda.Sigma,p_known=p_k) #run plda scoring

        positive_scores, negative_scores = get_pos_neg_score(actual,[speakers[i] for i in inds1], [speakers[i] for i in inds2], scores_plda) #separate scores for true/false comparisons
        
        #run threshold function
        eer1, eer_th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
        eer.append(eer1)
        print('EER: ' + str(eer1))
        print('Threshold: ' + str(eer_th))
        min_dcf1, mindcf_th = minDCF(torch.tensor(positive_scores), torch.tensor(negative_scores), c_miss, c_fa, p_target)
        min_dcf.append(min_dcf1)
        print('Min DCF: ' + str(min_dcf1))
        print('Threshold: ' + str(mindcf_th))
        threshold_eer.append(eer_th)
        threshold_mindcf.append(mindcf_th)
    
    threshold_eer = np.array(threshold_eer)
    threshold_mindcf = np.array(threshold_mindcf)
    # average threshold
    eer = np.array(eer)
    eer_ind = np.invert(np.isnan(eer))
    eer = eer[eer_ind]
    avg_eer = np.average(eer)
    threshold_eer = threshold_eer[eer_ind]

    min_dcf = np.array(min_dcf)
    min_dcf_ind = np.invert(np.isnan(min_dcf))
    min_dcf = min_dcf[min_dcf_ind]
    avg_min_dcf = np.average(min_dcf)
    threshold_mindcf = threshold_mindcf[min_dcf_ind]

    return np.average(threshold_eer),np.average(threshold_mindcf),threshold_eer, threshold_mindcf, eer, avg_eer, min_dcf, avg_min_dcf


def get_threshold_single(th_func, speakers, sentence_speakers, xv, plda, n, subset,c_miss=1.0, c_fa=1.0, p_target=1.0, p_k=1):
    '''
    Get an average threshold for trained PLDA 

    Inputs: th_func - string indicating which threshold function to use (EER or minDCF)
            speakers - list of unique speakers (list of strings of len # speakers)
            sentence_speakers - list of speaker for each sentence (list of strings of len # sentences)
            xv - numpy array containing all embeddings (# files x embedding dim (192))
            plda - trained PLDA object
            n - number of times to run threshold calculation (int)
            subset - size of subset (int)
            c_miss - (minDCF specifc) cost assigned to a missing error (default 1.0)
            c_fa - (minDCF specifc) cost assigned to a false alarm (default 1.0).
            p_target - (minDCF specifc) prior probability of having a target (default 0.01).
            p_k - (fast_PLDA_Scoring specific) probability of having a known speaker for open-setidentification case (=1 for the verification task and =0 for the closed-set case)
    Outputs: avg_th - avg calculated threshold (float)
             threshold_list - list of thresholds for each subset (list of floats)
             calc - list of eer/minDCF results for each subset (list of floats)
             avg_calc - avg eer/minDCF result (float)
    '''
    #subset = int(len(speakers)/100)
    # random_xv1 = average_xvectors(speakers, sentence_speakers, xv) #select random x-vector to get one vector per speakers (# speakers x embedding dim)
    # random_xv2 = average_xvectors(speakers, sentence_speakers, xv) #select rancom x-vector to get one vector per speakers (# speakers x embedding dim)

    random_xv1 = random_xvectors(speakers, sentence_speakers, xv) #select random x-vector to get one vector per speakers (# speakers x embedding dim)
    random_xv2 = random_xvectors(speakers, sentence_speakers, xv) #select rancom x-vector to get one vector per speakers (# speakers x embedding dim)

    threshold= []
    calc = []
    avg_calc = None
    for j in range(n):
        print('Run ' + str(j) + ' of ' + str(n))
        inds1 = random.sample(range(1, len(speakers)), subset)
        inds2 = random.sample(range(1, len(speakers)), subset)
        stat1, set1 = get_stat_ob(np.array([speakers[i] for i in inds1]), np.array([random_xv1[i,:] for i in inds1]))
        stat2, set2 =  get_stat_ob(np.array([speakers[i] for i in inds2]), np.array([random_xv2[i,:] for i in inds2]))
        actual = get_ref_database([speakers[i] for i in inds1], [speakers[i] for i in inds2])
        ndx = Ndx(models=set1, testsegs=set2)

        scores_plda = fast_PLDA_scoring(stat1, stat2, ndx, plda.mean, plda.F, plda.Sigma,p_known=p_k) #run plda scoring

        positive_scores, negative_scores = get_pos_neg_score(actual,[speakers[i] for i in inds1], [speakers[i] for i in inds2], scores_plda) #separate scores for true/false comparisons
        
        #run threshold function
        if th_func == 'eer':
            calc1, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
            calc.append(calc1)
            print('EER: ' + str(calc1))
            print('Threshold: ' + str(th))
            threshold.append(th)
        else:
            calc1, th = minDCF(torch.tensor(positive_scores), torch.tensor(negative_scores), c_miss, c_fa, p_target)
            calc.append(calc1)
            print('Min DCF: ' + str(calc1))
            print('Threshold: ' + str(th))
            threshold.append(th)
    
    # average threshold
    calc = np.array(calc)
    calc_ind = np.invert(np.isnan(calc))
    calc = calc[calc_ind]
    avg_calc = np.average(calc)
    threshold = threshold[calc_ind]


    return np.average(threshold),threshold, calc, avg_calc
                                                                                                                                                                                         
def main():
    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser()
    #required arguments
    parser.add_argument("-d", "--dataset", required=True, help="specify path to dataset description - make sure description is in proper format")
    parser.add_argument("-o", "--output_dir", required=True, help="specifies output directory for saving data")
    parser.add_argument("-n", "--overlap", required=True, type=int, help="specify known overlap")
    parser.add_argument("-e", "--experiment_dir", required=True, help = "specify where to save experiment csv")
    #optional arguments
    parser.add_argument("-b", "--beta", default=1.0, type=float, help="specify beta for fbeta measure")
    parser.add_argument("-p", "--plda_rank", default=100,type=int, help="specify rank for plda")
    parser.add_argument("-r", "--run_threshold", default=500, type=int, help="specify existing threshold")
    parser.add_argument("-s", "--subset", default=100, type=int,help="specify size of subset for threshold calculation")
    parser.add_argument("-th", "--th", default='',help="specify manual threshold - if specified, threshold calculation will be skipped")

    parser.add_argument("-t", "--th_func", default="both", choices=['eer','mindcf','both'], help="specify whether to calculate threshold with eer, minDCF, or both")
    parser.add_argument("-cf", "--c_fa", default=10.0, type=float, help="Specify minDCF argument c_fa: cost assigned to false acceptance")
    parser.add_argument("-cm", "--c_miss", default=0.1, type=float, help="Specify minDCF argument c_miss: cost assigned to false rejection")
    parser.add_argument("-pk", "--p_known", default=1, type=float, help="Specify PLDA scoring argument p_known: probability of having a known speaker for open-set identification case (=1 for verification task and =0 for closed-set case)")
    parser.add_argument("-pt", "--p_target", default=0.001, type=float, help="Specify minDCF argument p_target: prior probability of having a target")
    args = parser.parse_args()
    
    assert args.th_func in ['eer','mindcf','both'], 'invalid choice'

    # make sure output directory exists
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # SET DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    # READ IN DATASET
    dataset = pd.read_csv(args.dataset, keep_default_na=False)

    train = dataset.loc[dataset["train"]==True]
    train_paths = train["files"].tolist()
    for f in train_paths:
        assert os.path.exists(f), f'{f} not exists'
    train_speakers = [str(s) for s in train["speaker_id"].tolist()]

    test = dataset.loc[dataset["test"]==True]
    test_paths = test["files"].tolist()
    for f in test_paths:
        assert os.path.exists(f), f'{f} not exists'
    test_speakers = [str(s) for s in test["speaker_id"].tolist()]


    # GET FILES AND SPEAKER INFORMATION

    # GET SENTENCE INFO
    #train_sentences=[str(os.path.basename(train_paths[p])).replace(".wav","") for p in range(len(train_paths))] #get train sentence labels
    train_sentence_speakers = train_speakers #save list of speakers of len (# sentences)
    train_speakers = list(set(train_speakers)) #get list of unique train speakers

    test_sentence_speakers = test_speakers #save list of speakers of len (# sentences)
    test_speakers = list(set(test_speakers)) #get list of unique test speakers

    # GET EMBEDDINGS
    print("Getting Embeddings")
    train_xv = get_embeddings(train_paths)
    test_xv = get_embeddings(test_paths)
    print("completed")

    # TRAIN PLDA (approx 2 minutes with 600000~files)
    xvectors_stat, modelset = get_stat_ob(train_sentence_speakers, train_xv) #, train_sentences)
    plda = PLDA(rank_f=args.plda_rank)
    plda.plda(xvectors_stat)
    print("PLDA Trained")

    #  GET THRESHOLD
    print("Getting threshold")
    if args.th_func == 'both' and args.th == '':
        eer_th, mindcf_th,eer_th_list, mindcf_th_list, eer, avg_eer, min_dcf, avg_min_dcf = get_threshold(train_speakers, train_sentence_speakers, train_xv, plda, args.run_threshold, args.subset, args.c_miss, args.c_fa, args.p_target)
        print('Average EER threshold: ' + str(eer_th)) 
        print('Average EER: ' + str(avg_eer))
        print('Average MinDCF threshold: ' + str(mindcf_th)) 
        print('Average Min DCF: ' + str(avg_min_dcf))
        print("completed")
    elif (args.th_func == 'eer' or args.th_func == 'mindcf') and args.th == '':
        th, th_list, calc, avg_calc = get_threshold_single(args.th_func, train_speakers, train_sentence_speakers, train_xv, plda, args.run_threshold, args.subset, args.c_miss, args.c_fa, args.p_target)

    # SCORE TEST DATA 
    enrol_xv = average_xvectors(train_speakers, train_sentence_speakers,train_xv) #average x-vectors to get one vector per speakers (# speakers x embedding dim)
    en_stat, en_set = get_stat_ob(train_speakers, enrol_xv)

    test_xv = average_xvectors(test_speakers, test_sentence_speakers, test_xv) 
    te_stat, te_set = get_stat_ob(test_speakers, test_xv)

    actual = get_ref_database(train_speakers, test_speakers)
    ndx = Ndx(models=en_set, testsegs=te_set)

    scores_plda = fast_PLDA_scoring(en_stat, te_stat, ndx, plda.mean, plda.F, plda.Sigma,args.p_known)

    # THRESHOLD MATRIX 
    if args.th != '' or args.th_func == 'eer' or args.th_func == 'mindcf':
        if args.th != '':
            th = int(args.th)
        predicted = threshold_matrix(train_speakers, test_speakers, scores_plda,th, actual)
        prec, rec, fscore, confusion, mcc, acc, fp_rate = performance_metrics(actual, predicted, args.beta)

    else:
        predicted_eer = threshold_matrix(train_speakers, test_speakers, scores_plda, eer_th, actual)
        predicted_mindcf = threshold_matrix(train_speakers, test_speakers, scores_plda, mindcf_th, actual)

        # PERFORMANCE METRICS - EER
        prec_eer, rec_eer, f_score_eer, confusion_eer, mcc_eer, acc_eer, fprate_eer = performance_metrics(actual, predicted_eer, args.beta)

        # PERFORMANCE METRICS - mindcf
        prec_mindcf, rec_mindcf, f_score_mindcf, confusion_mindcf, mcc_mindcf, acc_mindcf, fprate_mindcf = performance_metrics(actual, predicted_mindcf, args.beta)

    if args.th != '' or args.th_func == 'eer' or args.th_func == 'mindcf':
        if args.th != '':
            print('MANUAL THRESHOLD')
        elif args.th_func == 'eer':
            print('EER THRESHOLD')
        elif args.th_func == 'mindcf':
            print('MinDCF THRESHOLD')
        print('Confusion Matrix:')
        print(confusion)
        print('False Positive Rate: ' + str(fp_rate))
        print('Precision: ' + str(prec))
        print('Recall: ' + str(rec))
        print('F-score: ' + str(fscore))
        print('Matthews Correlation Coefficient: ' + str(mcc))
        print('Accuracy: ' + str(acc))

    else:
        print('EER THRESHOLD')
        print('Confusion Matrix:')
        print(confusion_eer)
        print('False Positive Rate: ' + str(fprate_eer))
        print('Precision: ' + str(prec_eer))
        print('Recall: ' + str(rec_eer))
        print('F-score: ' + str(f_score_eer))
        print('Matthews Correlation Coefficient: ' + str(mcc_eer))
        print('Accuracy: ' + str(acc_eer))

        print('MinDCF THRESHOLD')
        print('Confusion Matrix:')
        print(confusion_mindcf)
        print('False Positive Rate: ' + str(fprate_mindcf))
        print('Precision: ' + str(prec_mindcf))
        print('Recall: ' + str(rec_mindcf))
        print('F-score: ' + str(f_score_mindcf))
        print('Matthews Correlation Coefficient: ' + str(mcc_mindcf))
        print('Accuracy: ' + str(acc_mindcf))

    print('Run threshold: ' + str(args.run_threshold))
    print('Subset: ' + str(args.subset))
    print('Plda rank: ' + str(args.plda_rank))

    # SAVE DATA
    data = {}
    metrics = {}

    f_name = os.path.basename(args.dataset).replace('speaker_info','plda') 
    f_name = f_name[:-4] + '_r' + str(args.run_threshold) + '_s' + str(args.subset)


    if args.th != '' or args.th_func == 'eer' or args.th_func == 'mindcf':
        data['actual_labels'] = actual.tolist()
        data['predicted_labels'] = predicted.tolist()
        data['train_speakers'] = train_speakers
        data['test_speakers'] = test_speakers
        data['plda_scores'] = [{'scores': scores_plda.scoremat.tolist(), 'modelset':scores_plda.modelset.tolist(), 'segset':scores_plda.segset.tolist()}]
        data['args'] = [{'run_thres':args.run_threshold, 'subset':args.subset, 'plda_rank':args.plda_rank}]
        if args.th != '':
            data['average_th'] = args.th
        else:
            data['average_th'] = th
            data['threshold_list'] = th_list
            data['calc'] = calc
            data['avg_calc'] = avg_calc

        metrics['precision'] = prec
        metrics['confusion_matrix'] = confusion
        metrics['recall'] = rec
        metrics['mcc'] = mcc
        metrics['fscore'] = fscore
        metrics['accuracy'] = acc
        metrics['fp_rate'] = fp_rate
         
        if args.th != '':
            f_name = 'manual_'+ f_name
        elif args.th_func == 'eer':
            f_name = 'eer_'+ f_name
        elif args.th_func == 'mindcf':
            f_name = 'mindcf_'+ f_name

    else:
        data['actual_labels'] = actual.tolist()
        data['predicted_labels'] = {'eer':predicted_eer.tolist(),'mindcf':predicted_mindcf.tolist()}
        data['train_speakers'] = train_speakers
        data['test_speakers'] = test_speakers
        data['plda_scores'] = [{'scores': scores_plda.scoremat.tolist(), 'modelset':scores_plda.modelset.tolist(), 'segset':scores_plda.segset.tolist()}]
        data['args'] = [{'run_thres':args.run_threshold, 'subset':args.subset, 'plda_rank':args.plda_rank}]
        data['eer'] = eer.tolist()
        data['avg_eer'] = avg_eer
        data['min_dcf'] = min_dcf.tolist()
        data['min_dcf_avg'] = avg_min_dcf
        data['average_th'] = {'eer':eer_th,'mindcf':mindcf_th}
        data['threshold_list'] = {'eer':eer_th_list.tolist(), 'mindcf':mindcf_th_list.tolist()}

        metrics['precision'] = {'eer':prec_eer,'mindcf':prec_mindcf}
        metrics['confusion_matrix'] = {'eer':confusion_eer.tolist(),'mindcf':confusion_mindcf.tolist()}
        metrics['recall'] = {'eer':rec_eer,'mindcf':rec_mindcf}
        metrics['mcc'] = {'eer':mcc_eer,'mindcf':mcc_mindcf}
        metrics['fscore'] = {'eer':f_score_eer,'mindcf':f_score_mindcf}
        metrics['accuracy'] = {'eer':acc_eer,'mindcf':acc_mindcf}
        metrics['fp_rate'] = {'eer':fprate_eer,'mindcf':fprate_mindcf}

        f_name = 'both_'+ f_name
    
    
    f_name_data = f_name + 'data'
    f_name_metrics = f_name + 'metrics'
    path_data = os.path.join(args.output_dir,f_name_data)
    path_metrics = os.path.join(args.output_dir,f_name_metrics)
    
    add_s = ''
    count = 0
    while os.path.exists(path_data+add_s+'.json'):
        print('path already exists')
        count += 1
        add_s = '_' + str(count)

    path_data = path_data + add_s + '.json'
    path_metrics = path_metrics + add_s + '.json'
    print('Output file data: ' + path_data)
    print('Output file metrics: ' + path_metrics)
    json_data = json.dumps(data, indent=4)
    with open(path_data, 'w') as outfile:
        outfile.write(json_data)

    json_metrics = json.dumps(metrics, indent=4)
    with open(path_metrics, 'w') as outfile:
        outfile.write(json_metrics)

    if args.th != '' or args.th_func == 'eer' or args.th_func == 'mindcf':
        row = [args.dataset, path_data,path_metrics] #initial dataset, path to data, path to metrics
        row += [fp_rate, prec,rec,mcc,fscore,acc] #precision, recall, mcc, fscore, acc
        row += [confusion, th]
        if args.th_func == 'eer' or args.th_func == 'mindcf':
            row += [th_list, calc, avg_calc]
        if args.th != '':
            exp_name = 'manual_th_experiments.csv'
        elif args.th_func == 'eer':
            exp_name = 'eer_th_experiments.csv'
        elif args.th_func == 'mindcf':
            exp_name = 'mindcf_th_experiments.csv'
        csv_f = os.path.join(args.experiment_dir,exp_name)

        if not os.path.exists(csv_f):
            f = open(csv_f, 'w')
            writer = csv.writer(f)
            header = ['dataset','output_data','output_metrics']
            header += ['fp_rate','precision','recall','mcc','fscore','accuracy']
            header += ['confusion','threshold']
            if args.th_func == 'eer' or args.th_func == 'mindcf':
                header += ['th_list', 'calc (eer or mindcf)', 'avg_calc (eer or mindcf)']
            writer.writerow(header)
            writer.writerow(row)
            f.close()
        else:
            f = open(csv_f,'a')
            writer=csv.writer(f)
            writer.writerow(row)
            f.close()
    
    else:
        row = [args.dataset, path_data,path_metrics] #initial dataset, path to data, path to metrics
        row += [fprate_eer, prec_eer,rec_eer,mcc_eer,f_score_eer,acc_eer] #precision, recall, mcc, fscore, acc
        row += [fprate_mindcf, prec_mindcf,rec_mindcf,mcc_mindcf,f_score_mindcf,acc_mindcf] #precision, recall, mcc, fscore, acc
        row += [confusion_eer, eer_th, avg_eer, confusion_mindcf, mindcf_th, avg_min_dcf]
        csv_f = os.path.join(args.experiment_dir,'experiments.csv')

        if not os.path.exists(csv_f):
            f = open(csv_f, 'w')
            writer = csv.writer(f)
            header = ['dataset','output_data','output_metrics']
            header += ['eer_fp_rate','eer_precision','eer_recall','eer_mcc','eer_fscore','eer_accuracy']
            header += ['mindcf_fp_rate','mindcf_precision','mindcf_recall','mindcf_mcc','mindcf_fscore','mindcf_accuracy']
            header += ['eer_confusion','eer_threshold','eer_average','mindcf_confusion','mindcf_threshold','mindcf_average']
            writer.writerow(header)
            writer.writerow(row)
            f.close()
        else:
            f = open(csv_f,'a')
            writer=csv.writer(f)
            writer.writerow(row)
            f.close()

if __name__ == "__main__": 
    main()