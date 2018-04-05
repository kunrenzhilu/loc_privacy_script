Required library:
    python 3.5
    tqdm
    sklearn
    numpy
    tqdm
    hd5f(optional)

Usage:
    You are supposed to run python stage_$STAGE_NUM.py [args]. 
    You should run stage_1 to stage_3 in sequence

    args: --data_dir The location of the dataset "dataset_TSMC2014_TKY.txt", which can be found https://sites.google.com/site/yangdingqi/home/foursquare-dataset
          --root_dir The location where all data will be saved to
          --num_cluster The number of clusters in the heuristic sampling. 3 is good
          --num_sample The number of sample for the 'sub' and 'cut' to be released to students
          --num_obf_sample The number of sample for the 'obf1' and 'obf2' to be released to students
    
Note:
    stage_1.py 
        Data are extracted from the raw file from *data_dir*, and generate the \full, \transition_profiles, \visit_profiles for all users, under the *root_dir* location. This works for both the \venue_level and the \discre_level
    stage_2.py
        It subsamples and perturbe from trajectories from \full, and genenerate \cut, \obf1, \obf2. The d-neighbour range is hardcoded to be [0.01, 0.05]. This works for both the \venue_level and the \discre_level
    stage_3.py
        It generate \selecte_samples that are supposed to be released to the students. It includes a heuristic solution to generate data that is not vunerable to the deterministic attack, but is only work for the discre_level, and is slow. It selected *num_sample* for the \sub and \cut datasets, and *num_obf_sample* for the \obf1 and \obf2 dataset. 

Another Note:
    Since my probabilistic attack is written in python, it's quite slow and not included in this package. 
