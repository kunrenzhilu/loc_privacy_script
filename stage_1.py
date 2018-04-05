import numpy as np
import os 
import time
import csv
import pickle
from datetime import datetime, timedelta
import collections
from collections import defaultdict, Counter
import numpy.ma as ma
from tqdm import *
from utils import *
import matplotlib
import argparse

"""
Data Format
1. User ID (anonymized)
2. Venue ID (Foursquare)
3. Venue category ID (Foursquare)
4. Venue category name (Fousquare)
5. Latitude
6. Longitude
7. Timezone offset in minutes (The offset in minutes between when this check-in occurred and the same time in UTC)
8. UTC time
"""
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/dataset_TSMC2014_TKY.txt', type=str)
parser.add_argument('--root_dir', default='./data/', type=str)
parser.add_argument('--num_cluster', default=3, type=int)
parser.add_argument('--num_sample', default=100, type=int)
parser.add_argument('--num_obf_sample', default=20, type=int)
args = parser.parse_args()

NUM_USERS = 2293
usrs = range(1, NUM_USERS+1)
scale_x = [35.510184, 35.860069];
scale_y = [139.47087765, 139.91259322]

def get_sorted_pickle_dict(data_dir):
    # read records from files and then bucketize by user id, return 3 objects.
    with open(data_dir, encoding="ISO-8859-1") as f:
        reader = csv.reader(f, delimiter='\t')
        to_pickle_dict = defaultdict(list) #This parse the time into datetime object
        sorted_pickle_dict = dict()

        for row in tqdm(reader):
            id, vid, vcid, vname, lat, long, offset, utc = row
            to_pickle_dict[int(id)].append([int(id), vid, vcid, vname, float(lat), float(long), int(offset), datetime.strptime(utc, '%a %b %d %H:%M:%S %z %Y')])
        for key in to_pickle_dict.keys():
            sorted_pickle_dict[key] = sorted(to_pickle_dict[key], key=lambda x :x[-1])
    return sorted_pickle_dict

def write_trajectory_to_file(trajs_dict, save_path, index):
    glb_dict = dict()
    usr_trajs = dict()
    for user, seq in tqdm(trajs_dict.items(), total=len(trajs_dict)):
        tmp_seq = list()
        with open(os.path.join(save_path, str(user)+'.txt'), 'w') as f:
            for p in seq:
                if not p[index] in glb_dict:
                    glb_dict[p[index]] = len(glb_dict)
                f.write(str(glb_dict[p[index]])+'\n')
                tmp_seq.append(glb_dict[p[index]])
        usr_trajs[user] = tmp_seq
    return usr_trajs, glb_dict

def get_visit_profile(usr_trajs):
    res_dict = dict()
    for usr in usr_trajs:
        p = Counter(usr_trajs[usr])
        len_p = len(usr_trajs[usr])
        for k, v in p.items():
            p[k] = v/len_p
        res_dict[usr] = dict(p)
    return res_dict

def save_visit_profiles(path, v_dict):
    for k, v in v_dict.items():
        with open(os.path.join(path, str(k)+'.txt'), 'w') as f:
            for k2 in sorted(v.keys()):
                f.write(str(k2)+', '+str(v[k2])+'\n')

def save_mc_profile(path, mat, id2ck_dict, usr_id):
    with open(os.path.join(path, str(usr_id)+'.txt'), 'w') as f:
        for i in range(len(mat)):
            for j in range(len(mat)):
                if mat[i,j] != 0:
                    f.write(str(id2ck_dict[i]) + ', ' + str(id2ck_dict[j]) + ', ' + str(mat[i,j]) + '\n') 

def create_and_save_mc_profiles(path, usr_dict):
    for usr in tqdm(usr_dict, total=len(usr_dict)):
        mat, ck2id_dict, id2ck_dict = compute_prob(usr_dict[usr])
        save_mc_profile(path, mat, id2ck_dict, usr)
        
def get_gps_trajectories(trajs_dict):
    one_seq = list()
    res_dict= dict()
    for usr, seq in tqdm(trajs_dict.items(), total=len(trajs_dict)):
        one_seq.extend(np.array(sorted_pickle_dict[usr])[:,4:6].tolist())
        res_dict[usr] = [np.array(sorted_pickle_dict[usr])[:,4:6]]
    return res_dict, one_seq

def make_token_data(trajs_dict, scale_w, scale_h, width, height, return_type=list):
    w_min, w_max = scale_w
    h_min, h_max = scale_h
    if return_type is list:
        flatten_trajs = list()
    elif return_type is dict:
        res_dict = dict()
        
    for uid, trajs in trajs_dict.items():
        for i, seq in enumerate(trajs):
            tmp_seq = list()
            for point in seq:
                tmp_seq.append(cart_2_token(
                    *get_discretized_coor(point[0], point[1], w_min, w_max, h_min, h_max, width, height)
                , width, height))
        res_dict[uid] = tmp_seq
    return res_dict

def make_normalized_coor_data(trajs_, scale_w, scale_h, return_type=list, slide_step=1):
    w_min, w_max = scale_w
    h_min, h_max = scale_h
    slide = lambda x,step: x[::step]
    if return_type is list:
        flatten_trajs = list()
    elif return_type is dict:
        res_dict = dict()
    
    if type(trajs_) is dict:
        if return_type is list:
            for uid, trajs in trajs_.items():
                for i, seq in enumerate(trajs):
                    tmp_seq = (seq[:,0:2] - [w_min, h_min]) / [float(w_max-w_min), float(h_max-h_min)]
                    flatten_trajs.append(slide(np.clip(tmp_seq,0,1),slide_step))
            return flatten_trajs
        elif return_type is dict:
            for uid, trajs in trajs_.items():
                tmp = list()
                for i, seq in enumerate(trajs):
                    tmp_seq = (seq[:,0:2] - [w_min, h_min]) / [float(w_max-w_min), float(h_max-h_min)]
                    tmp.append(slide(np.clip(tmp_seq,0,1),slide_step))
                res_dict[uid] = tmp
            return res_dict
    elif type(trajs_) is list:
        for seq in trajs_:
            tmp_seq = (seq[:,0:2] - [w_min, h_min]) / [float(w_max-w_min), float(h_max-h_min)]
            flatten_trajs.append(slide(np.clip(tmp_seq,0,1),slide_step))
        return flatten_trajs
    else:
        raise Exception('Trajs type is not defined')
    return None

def flatten_gps_dict(gps_dict):
    new_dict = dict()
    for k, v in gps_dict.items():
        new_dict[k] = v[0].astype(np.float64)
    return new_dict

if __name__=='__main__':
    print(Counter)
    data_dir = args.data_dir
    root_dir = args.root_dir

    venue_dir = os.path.join(root_dir, 'venue_level')
    venue_full_dir = os.path.join(venue_dir, 'full')
    venue_prof_dir = os.path.join(venue_dir, 'visit_profiles')
    venue_mc_dir = os.path.join(venue_dir, 'transition_profiles')                       
    discre_dir = os.path.join(root_dir, 'discre_level')
    discre_full_dir = os.path.join(discre_dir, 'full')
    discre_prof_dir = os.path.join(discre_dir, 'visit_profiles')
    discre_mc_dir = os.path.join(discre_dir, 'transition_profiles')

    dir_list = [root_dir, venue_full_dir, discre_full_dir,\
               venue_prof_dir, discre_prof_dir,\
               venue_mc_dir, discre_mc_dir]
    for path in dir_list: mkdir(path)
    
    print('Parsing data from files')
    sorted_dict_dir = os.path.join(root_dir, 'sorted_pickle_dict.pk')
    if not os.path.isfile(sorted_dict_dir):
        sorted_pickle_dict = get_sorted_pickle_dict(data_dir)
        save_pickle(sorted_dict_dir, sorted_pickle_dict)
    else: sorted_pickle_dict = read_pickle(os.path.join(root_dir, 'sorted_pickle_dict.pk'))
        
    ################ VENUE LEVEL ###############################    
    print('#'*20+'VENUE LEVEL'+'#'*20)
    print('Writing full traces to file')
    if not os.path.isfile(os.path.join(venue_full_dir, '1.txt')):
        venue_trajs, glb_dict = write_trajectory_to_file(sorted_pickle_dict, venue_full_dir, 1)
    else: 
        venue_trajs = read_and_construct_full_trace(venue_full_dir, usrs)
    
    print('Construct visit profiles')
    if not os.path.isfile(os.path.join(venue_prof_dir, '1.txt')):
        visit_profile_dict = get_visit_profile(venue_trajs)
        save_visit_profiles(venue_prof_dir, visit_profile_dict)
    else:
        visit_profile_dict = read_and_construct_visit_profiles(venue_prof_dir, usrs)
        
    print('Construct transition profiles')
    if not os.path.isfile(os.path.join(venue_mc_dir, '1.txt')):
        create_and_save_mc_profiles(venue_mc_dir, venue_trajs)
    
    ################ DISCRE LEVEL ##############################
    print('#'*20+'DISCRETE LEVEL'+'#'*20)
    print('Writing full traces to file')
    if not os.path.isfile(os.path.join(discre_full_dir, '1.txt')):
        gps_dict = make_normalized_coor_data(get_gps_trajectories( sorted_pickle_dict)[0], scale_x, scale_y, dict)
        save_dict(root_dir, flatten_gps_dict(gps_dict))
        token_trajs = make_token_data(gps_dict, [0,1], [0,1], 100, 100, dict)
        save_subsamples(discre_full_dir, token_trajs)
    else: 
        token_trajs = read_and_construct_full_trace(discre_full_dir, usrs)
    
    print('Construct visit profiles')
    if not os.path.isfile(os.path.join(discre_prof_dir, '1.txt')):
        visit_profile_dict2 = get_visit_profile(token_trajs)
        save_visit_profiles(discre_prof_dir, visit_profile_dict2)
    else:
        visit_profile_dict2 = read_and_construct_visit_profiles(discre_prof_dir, usrs)
        
    print('Construct transition profiles')
    if not os.path.isfile(os.path.join(discre_mc_dir, '1.txt')):
        create_and_save_mc_profiles(discre_mc_dir, token_trajs)
