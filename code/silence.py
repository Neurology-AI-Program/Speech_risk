#!/usr/bin/env python

'''
File: silence.py
Author: Daniela Wiepert
Date: 8/2021
Sources:
    https://maelfabien.github.io/project/Speech_proj/#high-level-overview
    
Detect voice sections and split on silence or remove silence from an audio file
'''

import os
import argparse
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import numpy as np


## HELPER FUNCTIONS FROM: https://maelfabien.github.io/project/Speech_proj/#high-level-overview
def _read_wav(wav_file):
    '''
    Read in a wav file and return the audio data and sample rate

    Input: wav_file - path to audio file (string)
    Output: data - audio data
            rate - sample rate in Hz (int)
    '''
    rate, data = wf.read(wav_file)
    channels = len(data.shape)
    
    assert channels == 1, "Channels must be 1 (mono)"

    return data, rate

def _calculate_frequencies(audio, rate):
    '''
    Compute range of possible frequencies at the sample rate for the given audio data

    Input: audio - audio data (int)
           rate - sample rate in Hz (int)
    Output: data_freq - possible frequencies (float)
    '''
    data_freq = np.fft.fftfreq(len(audio),1.0/rate)
    data_freq = data_freq[1:]
    return data_freq

def _calculate_energy(audio):
    '''
    Calculate the energy of the audio data

    Input: audio - audio data (int)
    Output: data_ampl_2 - square of the amplitude (float)
    '''   
    data_ampl = np.abs(np.fft.fft(audio))
    data_ampl = data_ampl[1:]
    data_ampl_2 = data_ampl ** 2
    return data_ampl_2

def _connect_energy_with_frequencies(data, rate):
    '''
    Create dictionary linking frequency with corresponding energy

    Input: data - audio data (int)
           rate - sample rate in Hz (int)
    Output: energy_freq - dictionary with frequency as keys and corresponding energy as values
    '''
    data_freq = _calculate_frequencies(data,rate)
    data_energy = _calculate_energy(data)

    energy_freq = {}
    for (i,freq) in enumerate(data_freq):
        if abs(freq) not in energy_freq:
            energy_freq[abs(freq)] = data_energy[i] * 2
    
    return energy_freq

def _sum_energy_in_band(energy_frequencies, speech_start_band, speech_end_band):
    '''
    Sum the energy corresponding to frequencies in human voice range in time window
    
    Input: energy_frequencies - dict containing frequencies as keys and corresponding energy as values
           speech_start_band - starting frequency for human voice in Hz (int)
           speech_end_band - ending frequency for human voice in Hz (int)
    Output: sum_energy - summed voice energy
    '''
    sum_energy = 0
    for f in energy_frequencies.keys():
        if speech_start_band < f < speech_end_band:
            sum_energy += energy_frequencies[f]
    return sum_energy

def _median_filter(x, k):
    '''
    Median filter for smoothing

    Input: x - input to smooth (1dim list)
           k - median filter (int)
    Output: median energy
    '''
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k-1) //2

    y = np.zeros((len(x),k), dtype=x.dtype)
    y[:,k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]

    return np.median(y,axis=1)

def _smooth_speech_detection(detected_voice, speech_window=0.5):
    '''
    apply smoothing to a region of detected speech

    Input: detected_voice - list containing speech ratios of detected voice
    Output: median_energy - output of median filter on detected_voice
    '''
    window =  0.02 
    median_window = int(speech_window/window)
    if median_window % 2 == 0:
        median_window = median_window - 1
    median_energy = _median_filter(detected_voice, median_window)

    return median_energy

## PLOTTING FROM: https://maelfabien.github.io/project/Speech_proj/#high-level-overview
def plot_signal(audio):
    '''
    Make basic plot of audio signal
    '''
    plt.figure(figsize=(12,8))
    plt.plot(np.arange(len(audio)),audio)
    plt.title("Raw audio signal")
    plt.show()

def plot_ratio(speech_ratio_list, threshold):
    '''
    Plot speech ratio versus threshold for speech
    '''
    plt.figure(figsize=(12,8))
    plt.plot(speech_ratio_list)
    plt.axhline(threshold, c = 'r')
    plt.title("Speech ratio list vs. threshold")
    plt.show()

def plot_detected_comparison(mean_data, detected_voice):
    '''
    Plot regions of detected speech vs. non-detected regions
    '''
    plt.figure(figsize=(12,8))
    plt.plot(np.array(mean_data), alpha=0.4, label="Not detected")
    plt.plot(np.array(detected_voice) * np.array(mean_data), label="Detected")
    plt.legend()
    plt.title("Detected vs. non-detected region")
    plt.show()

def plot_detected(detected_voice):
    '''
    Plot regions of detected speech
    '''
    plt.figure(figsize=(12,8))
    plt.plot(np.array(detected_voice), label="Detected")
    plt.legend()
    plt.title("Detected voice")
    plt.show()

## SPEECH DETECTION AND SILENCE REMOVAL FUNCTIONS

### From: https://maelfabien.github.io/project/Speech_proj/#high-level-overview
def detect_speech(audio, rate, speech_start_band, speech_end_band, sample_window, sample_overlap, threshold):
    '''
    Detect regions of speech using a rolling window algorithm

    Inout: audio - audio data (int)
           rate - sample rate in Hz (int)
           speech_start_band - starting frequency of human voice in Hz (int)
           speech_end_band - ending frequency of human voice in Hz (int)
           sample_window - window size in s (float)
           sample_overlap - amount to shift window at each step in s (float)
           threshold - threshold for energy ratio under which a sound is not tagged as voice (float)
    Output: speech_ratio_list - list containing speech ratios for each window
            detected_voice - list containing true or false depending on whether speech ratios are greater than the threshold
            mean_data - list containing means of audio data in each window
    '''
    # starting index
    sample_start = 0

    # initialize lists
    speech_ratio_list = []
    detected_voice = []
    mean_data = []

    # Go through audio data in increments based on the sample window
    while (sample_start < (len(audio) - sample_window*rate)):

        # Select only the region of the data in the window
        sample_end = int(sample_start + sample_window*rate)
        if sample_end >= len(audio):
            sample_end = len(audio)-1
        
        data_window = audio[sample_start:sample_end]
        mean_data.append(np.mean(data_window))

        # Full energy
        energy_freq = _connect_energy_with_frequencies(data_window,rate)
        sum_full_energy = sum(energy_freq.values())

        # voice energy
        sum_voice_energy = _sum_energy_in_band(energy_freq,speech_start_band,speech_end_band)

        # Speech ratio
        speech_ratio = sum_voice_energy/sum_full_energy
        speech_ratio_list.append(speech_ratio)
        detected_voice.append(speech_ratio > threshold)

        # Increment
        sample_start += int(sample_overlap*rate)

    return speech_ratio_list, detected_voice, mean_data

def merge_voice(detected_voice, min_silence, sample_overlap):
    '''
    Combine speech regions if the space between two voiced region is less than min_silence 

    Input: detected_voice - list of booleans, where a True value indicates the frame contains voice
           min_silence - length in ms to consider something silence (int)
           sample_overlap - amount to shift window at each step in s (float)
    Output: detected_voice - list of booleans, where a True value indicates the frame contains voice
    '''
    min_silence_i = min_silence//int(sample_overlap*1000)
    silent_segments = []
    ranges = []
    for i in range(len(detected_voice)):
        if i == 0:
            if not detected_voice[i]:
                ranges.append(i)
        else:
            if detected_voice[i] and not detected_voice[i-1]:
                silent_segments.append(ranges)
                ranges = []
            elif not detected_voice[i]:
                ranges.append(i)
            
    for v in silent_segments:
        if len(v) < min_silence_i:
            for j in v:
                detected_voice[j] = 1.0
    
    return detected_voice

def remove_excess_voice(detected_voice, min_voice, sample_overlap):
    '''
    Remove a speech region if the entire section of speech is less than min_voice

    Input: detected_voice - list of booleans, where a True value indicates the frame contains voice
           min_voice - length in ms to consider something a voiced region (int)
           sample_overlap - amount to shift window at each step in s (float)
    Output: detected_voice - list of booleans, where a True value indicates the frame contains voice
    '''    
    min_voice_i = min_voice//int(sample_overlap*1000)
    voiced_segments = []
    ranges = []
    for i in range(len(detected_voice)):
        if i == 0:
            if detected_voice[i]:
                ranges.append(i)
        else:
            if not detected_voice[i] and detected_voice[i-1]:
                voiced_segments.append(ranges)
                ranges = []
            elif detected_voice[i]:
                ranges.append(i)
            
    for v in voiced_segments:
        if len(v) < min_voice_i:
            for j in v:
                detected_voice[j] = 0.0

    return detected_voice

def split_on_silence(wav_file, output_path, audio, rate, detected_voice, sample_window, sample_overlap):
    '''
    Given the list of which frames had detected silence, split the audio on silence and export each voiced section as a wav file

    Input: wav_file - path to wav (string)
           output_path - path to output directory (string)
           audio - audio data (int)
           rate - sample rate (int)
           detected_voice - list of booleans, where a True value indicates the frame contains voice
           sample_window - window size in s (float)
           sample_overlap - amount to shift window at each step in s (float)
    Output: None, exports an audio file
    '''

    sample_start = 0 #starting index for audio
    index = 0 #starting index for detected_voice
    save_audio = np.array([])  #empty array to store audio data
    f_counter = 0  #file counter for number of voiced sections


    # Go through audio in frames
    while (sample_start < (len(audio) - sample_window*rate)):

        # Select only the region of the data in the window
        sample_end = int(sample_start + sample_window*rate)
        if sample_end >= len(audio):
            sample_end = len(audio)-1
        
        # if voice is not detected
        if not detected_voice[index]:
            # if not the starting index
            if index != 0:
                # if the audio list is not empty
                if save_audio.size != 0:
                    fname = wav_file[:-4] + "_" + str(f_counter) + ".wav" # add file counter to file name
                    save_audio = save_audio.astype(np.int16) # convert to int16 
                    wf.write(os.path.join(output_path,fname), rate, save_audio) # write wav file in output directory with specified rate and audio data
                    
                    f_counter += 1 #increment file counter
                    save_audio = np.array([]) #re-initialize audio array
        
        else:
            # if starting index
            if index == 0: 
                save_audio = np.append(save_audio,audio[sample_start:sample_end])

            # if not starting index and voice is detected
            else:
                # if previous index does not have voice
                if not detected_voice[index-1]:
                    save_audio = np.append(save_audio,audio[sample_start:sample_end])

                # if previous index does have voice
                else:
                    temp_start = sample_start + int(sample_overlap*rate)
                    save_audio = np.append(save_audio,audio[temp_start:sample_end])
        
        index += 1 # increment detected_voice index
        sample_start += int(sample_overlap*rate) # increment audio index (need to consider frames/window overlap)

        if index == len(detected_voice):
            if save_audio.size != 0:
                    fname = wav_file[:-4] + "_" + str(f_counter) + ".wav" # add file counter to file name
                    save_audio = save_audio.astype(np.int16) # convert to int16 
                    wf.write(os.path.join(output_path,fname), rate, save_audio) # write wav file in output directory with specified rate and audio data
                    
                    f_counter += 1 #increment file counter
                    save_audio = np.array([]) #re-initialize audio array
    return None

def remove_silence(wav_file, output_path, audio, rate, detected_voice, sample_window, sample_overlap):
    '''
    Given the list of which frames had detected silence, remove the silent regions from the audio and export as one new wav file

    Input: wav_file - path to wav (string)
           output_path - path to output directory (string)
           audio - audio data (int)
           rate - sample rate (int)
           detected_voice - list of booleans, where a True value indicates the frame contains voice
           sample_window - window size in s (float)
           sample_overlap - amount to shift window at each step in s (float)
    Output: None, exports an audio file
    '''
    sample_start = 0 #starting index for audio
    index = 0 #starting index for detected_voice
    save_audio = np.array([])  #empty array to store audio data

    # Go through audio data
    while (sample_start < (len(audio) - sample_window*rate)):

        # Select only the region of the data in the window
        sample_end = int(sample_start + sample_window*rate)
        if sample_end >= len(audio):
            sample_end = len(audio)-1
        
        # if voice detected
        if detected_voice[index]: 
            # if starting index
            if index == 0: 
                save_audio = np.append(save_audio,audio[sample_start:sample_end])

            # if not starting index and voice is detected
            else:
                # if previous index does not have voice
                if not detected_voice[index-1]:
                    save_audio = np.append(save_audio,audio[sample_start:sample_end])

                # if previous index does have voice
                else:
                    temp_start = sample_start + int(sample_overlap*rate)
                    save_audio = np.append(save_audio,audio[temp_start:sample_end])

        index += 1 # increment detected_voice index
        sample_start += int(sample_overlap*rate) # increment audio index (need to consider frames/window overlap)
  
    fname = wav_file[:-4] + "_combined" + ".wav" #specify file name
    save_audio = save_audio.astype(np.int16) # convert to int16
    wf.write(os.path.join(output_path,fname),rate,save_audio) # write new wav file

    return None
           
## MAIN FUNCTION
def silence_removal(wav_file, input_path, output_path, save_name='', plot=False, split=False, min_voice=500, min_silence=500, speech_start_band=300, speech_end_band=3000, sample_window=0.02, sample_overlap=0.01, threshold=0.25, speech_window=0.5):
    '''
    Remove silences by determining where there is no speech detected in a given wav file
    
    Input: wav_file - path to wav file to read in (string)
           save_name - file name for writing new wav file in output (string)
           input_path - path to input directory (string)
           output_path - path to output directory (string)
           plot - boolean indicating whether to plot data (default: False)
           split - boolean indicating whether to split on silence or just remove and export as one file (default: False)
           min_voice - length in ms to consider something a voiced region (int, default: 500ms)
           min_silence - length in ms to consider something silence (int, default: 500ms)
           speech_start_band - starting frequency of human voice in Hz (int, default: 300 Hz)
           speech_end_band - ending frequency of human voice in Hz (int, default: 3000 Hz)
           sample_window - window size in s (float, default: 0.02s, 20ms)
           sample_overlap - amount to shift window at each step in s (float, default: 0.01s, 10ms)
           threshold - threshold for energy ratio under which a sound is not tagged as voice (float, default: 0.25)
           speech_window - (float, default: 0.5)
    Output: None, exports new wav file with silence removed
    '''
    if save_name == '':
        save_name = wav_file

    # read in audio data
    read_file, rate = _read_wav(os.path.join(input_path,wav_file))

    # if plotting, plot the audio signal
    if plot:
        plot_signal(read_file)

    #detect speech
    speech_ratio_list, detected_voice, mean_data = detect_speech(read_file, rate, speech_start_band, speech_end_band, sample_window, sample_overlap, threshold)
    
    #if plotting, plot speech ratios and comparison of detected voice/non-voice sections before smoothing
    if plot:
        plot_ratio(speech_ratio_list,threshold)
        plot_detected_comparison(mean_data, detected_voice)
    
    # smoothing with median filter
    detected_voice = _smooth_speech_detection(np.array(detected_voice),speech_window)

    # if plotting, plot detected voice and comparison of detected voice/non-voice sections after smoothing
    if plot:
        plot_detected(detected_voice)
        plot_detected_comparison(mean_data,detected_voice)
    
    # merge voiced sections if silence is less than min_silence
    detected_voice = merge_voice(detected_voice, min_silence, sample_overlap)

    # if plotting, plot detected voice
    if plot:
        plot_detected(detected_voice)

    # remove voiced sections if voicing is less than min_voice
    detected_voice = remove_excess_voice(detected_voice, min_voice, sample_overlap)

    # if plotting, plot detected voice
    if plot:
        plot_detected(detected_voice)

    # remove sections without voice detected, either by splitting on silence or removing normally, these functions export wav files
    if split:
        split_on_silence(save_name, output_path, read_file, rate, detected_voice, sample_window, sample_overlap)
    else:
        remove_silence(save_name, output_path, read_file, rate, detected_voice ,sample_window, sample_overlap)

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--wav_file", required=True, help="specify wav file to remove silence from")
    parser.add_argument("-i", "--input_path", required=True, help="specify path to directory where input wav file is located")
    parser.add_argument("-p", "--output_path", default=".\\", help="specify output path for file")
    parser.add_argument("-s", "--speech_start_band", type=int, default=300, help="specify minimum frequency in Hz corresponding to human voice")
    parser.add_argument("-e", "--speech_end_band", type=int, default=3000, help="specify maximum frequency in Hz corresponding to human voice")
    parser.add_argument("-w", "--sample_window", default=0.02, help="specify window size in s (0.02=20ms)")
    parser.add_argument("-o", "--sample_overlap", default=0.01, help="specify amount of window overlap in s (0.01=10ms)")
    parser.add_argument("-t", "--threshold", default=0.22, help="specify energy threshold under which a sound is not tagged as voice")
    parser.add_argument("-mv", "--min_voice", type=int, default=350, help="specify minimum length for a region to count as voice in ms (500 ms)")
    parser.add_argument("-ms", "--min_silence", type = int, default=500, help="specify minimum length for a region to count as silence in ms (500 ms)")
    parser.add_argument("-sw","--speech_window", default=0.5, help="specify what counts as window of speech in s (0.5s=50ms)")
    parser.add_argument("--plot", default=False, action="store_true", help="specify whether to plot signal and audio removal")
    parser.add_argument("--split", default=False, action="store_true", help="specify whether to split audio on silence or just remove it")
    args = parser.parse_args()

    # make sure output directory exists
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # complete silence removal using arguments specified in command line
    silence_removal(args.wav_file, args.input_path, args.output_path, '', args.plot, args.split, args.min_voice, args.min_silence, args.speech_start_band, args.speech_end_band, args.sample_window, args.sample_overlap, args.threshold, args.speech_window)

if __name__ == "__main__":
    main()

