# Risk of re-identification for shared clinical speech recordings
Daniela A. Wiepert, Bradley A. Malin, Joseph R. Duffy, Rene L. Utianski, John L. Stricker, David T. Jones, and Hugo Botha

preprint available at: https://doi.org/10.48550/arXiv.2210.09975

## Abstract
Large, curated datasets are required to leverage speech-based tools in healthcare. These are costly to produce, resulting in increased interest in data sharing. As speech can potentially identify speakers (i.e., voiceprints), sharing recordings raises privacy concerns. We examine the re-identification risk for speech recordings, without reference to demographic or metadata, using a state-of-the-art speaker recognition system. We demonstrate that the risk is inversely related to the number of comparisons an adversary must consider, i.e., the search space. Risk is high for a small search space but drops as the search space grows (precision > 0.85 for < 1x10^6 comparisons, precision < 0.5 for > 3x10^6 comparisons). Next, we show that the nature of a speech recording influences re-identification risk, with non-connected speech (e.g., vowel prolongation) being harder to identify. Our findings suggest that speaker recognition systems can be used to re-identify participants in specific circumstances, but in practice, the re-identification risk appears low.

## Datasets
VoxCeleb 1 & 2: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

Mayo clinical speech recordings dataset
![mayo_breakdown](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/images/mayo_table.png)


## Architecture
![Model Architecture](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/images/architecture.png)
**Figure 1. Speaker Recognition System Architecture.** During training (top section) only the recordings from
known speakers are used. These are fed into a pretrained speaker identification model, which outputs the
predicted speaker out of a closed set. The prediction is not used, however, and instead the activations from the
penultimate fully connected layer are extracted for each recording. These represent a low dimensional, latent
representation for each recording that is enriched for speaker-identifying features (x-vectors). We used these
x-vectors for known speakers to train a PLDA classifier, which scores the pairwise similarity for all recordings and
proposes a threshold which minimizes the cost function (minDCF). This was done over several subsets and the
threshold averaged to obtain the final training threshold. During testing (bottom section), x-vectors are extracted
for recordings from unknown speakers in the same was as before. These are then fed into the trained PLDA but
pairwise comparisons are done with the known speaker x-vectors. The training threshold is applied, and the net
result is a set of matches (or no matches) for each recording. Those matches are used to calculate our metrics of
interested (FAs, TAs, FAR, etc.).

## Results
### Experiments
Find table of all different experiments run at [output_vox.csv](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/examples/output_vox.csv) and [output_mayo.csv](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/examples/output_mayo.csv).

### Plots
* Plots were manually generated using [plots.Rmd](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/code/plots.Rmd). The resulting plots can be found at [plots](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/images/plots/)

TODO: add figure captions
![figure2](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/images/plots/vox_counts.png)
**Figure 2. Number of true and false acceptances for the speaker recognition model in a realistic scenario with VoxCeleb.** **(a)** shows the counts when varying the 
number of known speakers while keeping the number of unknown speakers static, **(b)** shows the counts when varying the number of unknown speakers 
while keeping the number of known speakers static, and **(c)** shows the overall trend in terms of number of comparisons made (i.e.,
the search space size = known ∗ unknown speakers). All plots **(a-c)** include the Pearson’s correlation coefficient
and corresponding significance for false acceptances and number of speakers/comparisons. Each run is plotted as
a single circle, with red horizontal lines indicating the mean number of false acceptances and green horizontal lines
indicating the mean number of true acceptances.

![figure3](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/images/plots/vox_prec.png)
**Figure 3. Precision and False Acceptance Rates for the speaker recognition model in a realistic scenario with VoxCeleb.** Precision **(a)** and false acceptance rates **(b)**
are shown as a function of the number of comparisons. For both plots, each run is represented by a circle, and the mean is represented by a horizontal black line.

![figure4](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/images/plots/vox_worstcase.png)
**Figure 4. Results for our speaker recognition model in worst-case scenarios with VoxCeleb.** **(a)** shows the true and
false acceptance counts for a known overlap scenario (limited to N = 5 best matches), while **(b)** shows the
corresponding precision as a function of the number of comparisons (search space size). **(c)** and **(d)** show the false
and true acceptance counts for a full overlap scenario, where all unknown speakers are present in the known
speaker set as a function of the number of comparisons (search space size). **(a)** and **(c)** also shows the Pearson’s
correlation coefficient and corresponding significance between false acceptances and number of comparisons. Each
run is plotted as a single circle, with red horizontal lines indicating the mean number of false acceptances, green
horizontal lines indicating the mean number of true acceptances, and black horizontal lines indicating the mean
precision.

![figure5](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/images/plots/mayo_counts.png)
**Figure 5. Results for our speaker recognition model with the Mayo clinical speech dataset.** **(a)** and **(b)** show cross-task results,
where recordings for known speakers are always sentence repetition but the task for unknown speaker recordings
varies. The baseline is when sentence repetitions are in both the known and unknown set. Pooling is when all
recordings for an unknown speaker are linked together across all tasks. **(a)** show the breakdown of counts for this
case while (b) is the corresponding precision. **(c)** and **(d)** show within-task results, where tasks for known and
unknown speakers are always the same. **(c)** is the breakdown of counts for this case while **(d)** is the
corresponding precision. Each run is plotted as a single circle, with red horizontal lines indicating the mean
number of false acceptances, green horizontal lines indicating the mean number of true acceptances, and black
horizontal lines indicating mean precision.

![Supplementary Figure 1](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/images/plots/vox_supplementary.png)
**Supplementary Figure 1. Results for known speaker set with fixed speakers in unknown set using VoxCeleb.** **(a)** shows the
breakdown of true and false acceptances when we change the number of known speakers and keep the unknown
speaker set fixed, with the exception of overlapping speakers. It also shows the Pearson’s correlation coefficient and
corresponding significance between false acceptances and number of known speakers. **(b)** shows the corresponding
precision and **(c)** shows the corresponding false acceptance rates. Each run is plotted as a single circle, with red
horizontal lines indicating the mean number of false acceptances, green horizontal lines indicating the mean
number of true acceptances, and black horizontal lines indicating the mean precision or false acceptance rate.


# Files 
## PREPROCESSING
 * Use [preprocessing_ECAPA-TDNN.py](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/code/preprocessing_ECAPA-TDNN.py) for preparing audio and getting ECAPA-TDNN embedding. Save processed files as .wav files and embeddings as .npy files.
 * Dependencies: speechbrain - make sure it is available in the environment you're running this.
 * Example command line input for running this file (only includes required arguments)
```
      python preprocessing.py -i PATH_TO_INPUT_DIR -o PATH_TO_OUTPUT_DIR -f PATH_TO_FEATURE_DIR
```
 * PATH_TO_INPUT_DIR should point to directory containing original raw audio files. If running on Mayo Speech data, a csv file containing the dataset description should also be located in this directory. The dataset description should contain file names and features such as file type (*.wav), speech task, and speaker IDs. 
 * PATH_TO_OUTPUT_DIR should point to directory where preprocessed files should be written to
 * PATH_TO_FEATURE_DIR should point to directory where embedding files should be written to
 
 * If you have a list of files to process saved as a txt file, you can use the flag '-t TO_PROCESS_TXT' so that you don't process the entire dataset. An example is [example_process.txt](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/examples/example_process.txt)

 * By default, preprocessing will resample audio to 16000 Hz, convert it to mono and remove beginning and end silences
      * To remove all silences, add the flag "--remove_all"
 * If removing all silences, it will default to removing and saving to one file rather than splitting on audio 
      * To split on silences, add the flag "--split"
 * If you have already run audio preprocessing (i.e. removed silence, normalized) but need to generate new features, use the "--skip_audio" flag to skip the audio pipeline.
 * If you do not need to generate new features, use the "--skip_features" flag to skip the feature pipeline.
 
 * Other arguments can be found using the flag "-h"
 * Some arguments of interest may be "-t" for specifying silence threshold in dBs, as well as "-m" for specifying minimum silence length in ms.

### GET EMBEDDINGS
 * If you already have files processed but no embeddings, use [get_ECAPA_TDNN_embeddings.py](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/code/get_ECAPA_TDNN_embeddings.py) to get embeddings. If you want this functionality outside of the command line, you can use the function ```generate_embeddings```, which is used in preprocessing.
 * Dependencies: speechbrain - make sure it is available in the environment you're running this.
 * Example command line input for running this file (only includes required arguments)
```
      python generate_ECAPA_TDNN_embeddings.py -i PATH_TO_INPUT_DIR -d WHICH_DATA
```
 * PATH_TO_INPUT_DIR should point to directory containing original raw audio files. 
 * WHICH_DATA should be either 'timit' or 'speech' to indicate which dataset is being used

 * If you would like to save the embeddings, add the following flags:
    * '--save' to toggle saving on
    * '-s PATH_TO_SAVE_DIR' to specify where you want to save the embeddings
    * '--mac' if you are running this file and saving on a mac/linux system

* Example embeddings for a video are saved at [vox_video_embedding](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/examples/vox_video_embedding)

### SILENCE REMOVAL
 * In order to perform silence removal in preprocessing, you need [silence.py](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/code/silence.py). It performs silence removal on a single wav_file. 
 * Example command line input for running this file (only includes required arguments)
 ```
      python silence.py -f WAV_FILE -i PATH_TO_INPUT_DIR 
 ```
 * WAV_FILE should be the bare file_name.wav file as a string
 * PATH_TO_INPUT_DIR should point to directory containing original raw audio file

 * The function ```silence_removal``` can be imported into another python code to run silence removal outside of command line

## TRAIN/TEST SPLIT
### COMBINED DATASETS 
 * Use [combined_train_test_split.py](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/code/combined_train_test_split.py) to generate a train/test split for any number of combined datasets. The result is a csv file with the file names, speaker ids, data type, and booleans indicating whether it is part of the train or test set. Output files are of the form '*_speaker_info_tr*_te*_o*.csv'. 
 * Example command line input for running this file (only includes required arguments)
```
      python combined_train_test_split.py -i PATH_TO_INPUT_DIR1 PATH_TO_INPUT_DIR2 ... -o PATH_TO_OUTPUT_DIR -d DATA_TYPE1 DATA_TYPE2 ... -t DATA_TYPE1 DATA_TYPE2 ...
```

 * PATH_TO_INPUT_DIR should point to directory containing EMBEDDINGS - files of the form '*.npy'
 * PATH_TO_OUTPUT_DIR should point to directory where resulting csv dataset descriptions should be written to
 * '-d' takes any number of arguments for which data types are being used to make the combined split. Note that they MUST be either ['vox1', 'vox2', 'vox', 'vox-test','mayo-speech'] and they need to be listed in the same order as their corresponding input directorys were listed under '-i'. 'vox' combines 'vox1' and 'vox2' for ease, and 'vox-test' specifies specifically the test set of both vox1 and vox2.
 * '-t' specifies which data types to use for the test data. It takes the same type of input as '-d'

 Note other semi-required arguments:
 * '--tasks=TASK.TXT' should point to a txt file containing which tasks from the mayo speech dataset to include in train/test/overlap. It is necessary that train and test tasks are specified, but overlap defaults to test sentences unless added in. An example txt file can be found in the [task_txt](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/examples/example_task.txt) directory. This is mandatory when mayo-speech is a specified data type. Things to consider - if you want to specify that all subtasks be included, write 'parent_task-\*' instead of 'parent_task-sub_task'. Note, if a task has no subtask (i.e. 'vowel', or 'grandfather') make sure to write it out as 'vowel-\*' or 'grandfather-\*' to make sure there is no error when running. Another functionality you can use is exclusion. If you want to exclude a task from train/test/overlap, lead with a '-' (i.e.'-s-physician'
 * '--dataset=PATH_TO_DATASET' should point to the csv file containing the dataset description for mayo-speech data. This is manadtory when mayo-speech is specified as a data type.
 * '-x NUM_TRAIN_SPEAKERS1 NUM_TRAIN_SPEAKERS2 ...' to specify number of speakers in train set. This has default values, but the number of arguments must be equal to the number of input directories ('-i') and data types ('-d') given, so it is better to manually input values to ensure everything is as you want.
 * '-y NUM_TEST_SPEAKERS1 NUM_TEST_SPEAKERS2 ...' to specify number of speakers in test set. This has default values, but the number of arguments must be equal to the number of test data types given ('-t').
 * '-n NUM_OVERLAP_SPEAKERS1 NUM_OVERLAP_SPEAKERS2 ...' to specify number of overlapping speakers in both train and test set. This has default values, but the number of arguments must be equal to the number of test data types given ('-t').

 * Some optional flags include:
    * '-f' to specify number of files per speaker in the test set (default 1 file per speaker)
    * '--dif_speakers' - if number of files is greater than 1, you can specify whether to treat each file as if they came from a different speaker rather than the same one. (i.e. embeddings aren't averaged per speaker)
    * If using the VoxCeleb dataset, use '--vox_all' if you want all files selected in test rather than just 1 file. You can also use argument '-v' to specify the number of videos used
      per speaker in train, test, and overlap. The options include ['all','half','1']. When using, specify in this order: ```-v TRAIN_V TEST_V OVERLAP_V```. Defaults to 'all 1 1'. This is different from number of files in the sense that each video has multiple files and you could select all files from one randomly selected video by using '--vox_all'

 * NOTE: the output of this will have full file paths, so if you are using different input directories, you will need to recreate the train/test split with your directories or edit the path names for existing split. 

 * NOTE 2: When using a dataset in both train and test (for overlap) + an additional dataset for non-overlapping test, you will need to include both dataset names under '-x', '-y', and '-n'. 
 
 An example for VoxCeleb:
 ```
 python3 combined_train_test_split.py -i ./voxceleb/vox_embeddings/ -o ./speaker_info/ -d vox -t vox -x 7205 -y 163 -n 5
 ```

 * NOTE 3: For Supplemental VoxCeleb tests, where we use vox test embeddings only for re-identification, you would need to give ```-i ./voxceleb/vox_embeddings/dev and ./voxceleb/vox_embeddings/test``` and also ```-d vox vox-test -t vox -x 7205 0 -y 5 158 -n 5 0``` to account for the fact that 158 speakers are from vox-test only while the 5 overlap is from the rest of vox

## RUN PLDA
 * Use [plda.py](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/code/plda.py) to train PLDA on your data.
 * Dependencies: speechbrain - make sure it is available in the environment you're running this.
 * Example command line input for running this file (only includes required arguments)
```
      python plda.py -d PATH_TO_SPLIT -o PATH_TO_OUTPUT_DIR -e PATH_TO_SAVE_CSV -n NUM_OVERLAP
``` 
 * PATH_TO_SPLIT should be the full file path pointing to the train/test split generated with [combined_train_test_split.py](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/code/combined_train_test_split.py) to make sure all the proper columns exist. Due to git size limits, example speaker infos can not be uploaded. NOTE: all paths in speaker info should be FULL FILE PATHS. If you previously created a split but are now using a different directory, you can either recreate or edit the txt files.
 * PATH_TO_OUTPUT_DIR should point to directory where results should be saved to. Some of the results from our runs have been saved in the [examples](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/examples/) directory, though many of them are too large to be saved on GitHub. The output is saved as two .json files, one with all the PLDA scores and other data, and one with only metrics. Only the [metric example](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/examples/TODO) is able to be loaded to github.
 * PATH_TO_SAVE_CSV should specify file path where you want the output csv with metrics saved. NEEDS TO BE THE SAME FOR EACH RUN IF YOU WANT DATA SAVED IN SAME PLACE. An example for our experiments can be found at [experiments.csv](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/examples/experiments.csv)
 * NUM_OVERLAP should specify the known overlap between train and test speakers (int)


 * There are other flags that can be used to change aspects of the PLDA and scoring. These can be found using the flag '-h'
 * Some flags of interest may be:
      * '-r' to change the number of times the threshold is calculated (default: 500)
      * '-s' to change the size of the subsets for threshold calculation (default: )
      * '-p' for plda rank (default: 100)

 * In terms of thresholding, there are additional options to consider
      * '-t' to set the threshold function (default = 'both', which will run both EER and minDCF. Other options are ['eer','mindcf'])
      * '-th' if you want to manually set a threshold rather than using EER or minDCF
      * If using minDCF, you can also alter the following parameters. It will run with a strict threshold as default.
            * '-cf' to specify cost of a false acceptance (default = 10.0)
            * '-cm' to specify cost of a flase rejection (default = 0.1)
            * '-pt' to specify prior probability of target (default = 0.001)
            * '-pk' to specify probability of having a known speaker for open-set identification case (=1 for verification task and =0 for closed-set case)
            * If you want to use a more relaxed configuration, consider the following settings: `-cf 1 -cm 1 -pt 0.1`

 * NOTE: this will return all the necessary information from running the PLDA to calculate addition metrics in a separate file (as we do). This file only saves out the standard precision, recall, matthew's correlation coefficient, fscore, accuracy, threshold information, and confusion matrix for both EER and minDCF. 

 * NOTE 2: the output file names have the following format: *TH-FUNC-TYPE*_*DATA-TYPE*_tr*NUM-TRAIN-SPEAKERS*_*AMOUNT-TR-RECORDINGS*_te*NUM-TEST-SPEAKERS*_*AMOUNT-TE-RECORDINGS*_o*NUM-OVERLAP-SPEAKERS*_*AMOUNT-O-RECORDINGS*_r*NUM-TH-RUNS*_s*TH-SUBSET-SIZE*_data/metrics.json
      * If both EER and minDCF were calculated, TH-FUNC-TYPE will be 'both'. If manual threshold was given, TH-FUNC-TYPE will be 'manual'. Otherwise it will just be the function name ('eer' or 'mindcf')
      * By default, AMOUNT-TR-RECORDINGS is 'all', AMOUNT-TE-RECORDINGS is '1', and AMOUNT-O-RECORDINGS is 's' for 1 sentence(utterance)

## ADDITIONAL METRICS
 * Use [additional_metrics.py](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/code/additional_metrics.py) to calculate additional metrics
 * This code will load the PLDA scores from the output of [plda.py](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/code/plda.py) and re-calculate all metrics for a rank 1 case (where only the best match per speaker is used) and a known overlap case (where the top N = known overlap cases are used). Additionally, it will calculate whether false acceptances are from an overlapping or non-overlapping speaker (i.e was the speaker in both the known and unknown set).
 * Results are saved as csv files in the original data directories. [example 1](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/examples/TODO) shows rank1 results, [example2](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/examples/TODO) shows known overlap results, and [example 3](https://github.com/Neurology-AI-Program/Speech_risk/blob/main/examples/TODO) shows non-overlapping FA results
 * Example command line input for running this file (only includes required arguments)'
```
      python additional_metrics.py -d DATA_DIR1 DATA_DIR2 ... DATA_DIRN -r PATH_TO_ROOT_DATA_DIR
``` 
* DATA_DIR# should list directories where PLDA outputs were saved - instead of typing the full directory path, you only need to type the portion of the path not specified by the root directory (e.g. if the root directory *outputs* has two directories *train* and *test* each with subdirectories *1000*, *2000*, but you only need the *1000* directory from *train* and the *2000* directory from test, you would use the command `-d train/1000 test/2000`)
* PATH_TO_ROOT_DATA_DIR should list the full file path of the root directory for the data directories from the previous command (e.g. in the previous example, the root directory was *outputs*, so you would use the command `-r ...full_path.../outputs/')

Some optional arguments of interest may be
* `-e REG_EXP_FOR_OUTPUT_FNAMES` which allows you to be more specific about which file paths you want to get counts for. The PLDA outputs two files (one for data and one for metrics), but we want to grab only the data files for this purpose. This can be complicated if outputs for more than one experiment are kept in the same folder. Since output file names list all identifying information for the run (i.e. number known speakers, number unknown speakers, number overlapping speakers, number of threshold runs, th subset size), we can make use of the information using a regular expression to pattern match and grab the files with only the data we want. By default, it will gather all outputs with 'data' in the file name using this expression `*data.json`, but you can make this more complex by changing it to something like `*_v_tr*_all_te*_1_o5_s_*r500_s100data.json`, which will get only outputs with all recordings used in the training set but with any training size `tr*_all`, 1 recording used in the test set with any test size `te*_1`, 1 recording in the overlap set with 5 speakers `o5_s`, 500 threshold calculation runs `r500`, and a subset size of 100 for threshold calculations `s100`. 
* '-t' to specify which threshold function you want to get counts for. If using results where both EER and minDCF were calculated (i.e. the file begins with 'both'), you would select either 'eer' or 'mindcf'. If using one that only calculated one of EER/minDCF (i.e. the file begins with 'eer' or 'mindcf'), use 'eer_only' or 'mindcf_only'. If you used a manual threshold (i.e. the file begins with 'manual'), use 'manual'.
* '-c' to specify which calculations you want completed. The choices are 'rank1' - to calculate metrics for rank1 matches, 'known' - to calculate metrics for known overlap case, and 'overlap' - to calculate whether FA are from overlapping or non-overlapping speakers. By default, all are calculated.
* '-n' to specify what the known overlap is. By default, it is 5.

* NOTE: in this code, results for each data directory will be saved in that director. 
      * rank 1 metrics are saved as `rank1_counts_*.csv`
      * known overlap metrics are saved as `*_known_overlap_counts_*.csv`
      * FA overlapping/non-overlapping counts are saved as `non_overlap_acceptances_*.csv`


## Code Sources
* [VoxCeleb PLDA speaker verification](https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/speaker_verification_plda.py)
* [Speech Brain fast_PLDA_scoring example](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.processing.PLDA_LDA.html#module-speechbrain.processing.PLDA_LDA)
* [ECAPA-TDNN embeddings](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
* [Silence Removal 1](https://maelfabien.github.io/project/Speech_proj/#high-level-overview)
* [Silence Removal 2](https://gist.github.com/sotelo/be57571a1d582d44f3896710b56bc60d)
