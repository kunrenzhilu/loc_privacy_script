import numpy as np
import os 
import time
import csv
import pickle
import collections
from collections import defaultdict, Counter
from tqdm import *
from utils import *
import argparse
from stage_1 import *
from parallel_matrix import *

# subsample the trajectories
def subsample_obf(trajs_dict, d_neighbor_dict, plist=None, users=None):
    if plist is None:
        plist = np.random.uniform(0.1,0.5,len(trajs_dict))
    res_dict = dict()
    for i, (usr, seq) in enumerate(trajs_dict.items()):
        p = plist[i]
        tmpseq = list()
        coins = np.random.uniform(0, 1, len(seq))
        for j in range(len(seq)):
            if coins[j] < p:
                tmpseq.append(seq[j])
            else:
                if len(d_neighbor_dict[seq[j]]) != 0:
                    tmpseq.append(np.random.choice(d_neighbor_dict[seq[j]]))
        res_dict[usr] = tmpseq
    return res_dict, plist

def subsample_ran(trajs_dict, plist=None, usrs=None):
    if usrs is None:
        usrs = list(trajs_dict)
    if plist is None:
        plist = np.random.uniform(0.1, 0.5, len(trajs_dict))
    res_dict = dict()
    for i, (usr, seq) in enumerate(trajs_dict.items()):
        if not usr in usrs: continue
        p = plist[i]
        tmpseq = list()
        coins = np.random.uniform(0,1, len(seq))
        for j in range(len(seq)):
            if coins[j] < p:
                tmpseq.append(seq[j])
        res_dict[usr] = tmpseq
    return res_dict, plist

def subsample_cut(trajs_dict, plist=None, usrs=None):
    if usrs is None:
        usrs = list(trajs_dict)
    if plist is None:
        plist = np.random.uniform(0.1,0.5,len(trajs_dict))
    res_dict = dict()
    for i, (usr, seq) in enumerate(trajs_dict.items()):
        if not usr in usrs: continue
        p = plist[i]
        lenp = len(seq)
        s = np.random.choice(range(lenp-int(np.floor(lenp * p))-1))
        res_dict[usr] = seq[s:s+int(np.floor(lenp * p))]
    return res_dict, plist

def get_pdict(plist, usrs):
    pdict = dict()
    for i, u in enumerate(usrs):
        pdict[u] = plist[i]
    return pdict

def expand_dict(dict_, keys):
    for k in keys:
        dict_[k] = []

def save_dict_to_text(dict_, id2coor_dict, path):
    ids = list(id2coor_dict.keys())
    with open(path, 'w') as f:
        for k, v in dict_.items():
            f.write(str(k)+', ' + ', '.join(map(str, v)) + '\n')
            
def get_id2coor_dict(trajs_dict, gps_dict):
    id2coor_dict = dict()
    tmp = list()
    for usr, traj in trajs_dict.items():
        for i, p in enumerate(traj):
            id2coor_dict[p] = gps_dict[usr][i]
    return id2coor_dict

def save_different_version_obf(dirkeys, dir_dict, trajs_dict, id2coor_dict, usrs):
    import os
    def get_nb_dict(id2coor_dict, d, nProcess):
        indices = np.triu_indices(len(id2coor_dict), 1)
        ids = list(id2coor_dict)
        tmpdict = parallel_neighbors(ids, indices, id2coor_dict, d**2, nProcess)
        return tmpdict
        
    for i, d in enumerate([0.01, 0.05]):
        print('*'*20 + str(d) + '*'*20)
        plist = np.random.uniform(0.1, 0.5, len(usrs))
        print('Get D-neigbour')
        nb_dict = get_nb_dict(id2coor_dict, d, 20)
        expand_dict(nb_dict, set(id2coor_dict).difference(nb_dict))
        print('Get_subsamples')
        sub_dict_obf, plist = subsample_obf(trajs_dict, nb_dict, plist)
        print('Save things')
        save_subsamples(dir_dict[dirkeys[i]], sub_dict_obf)
        save_sample_rate(dir_dict[dirkeys[i]], plist, trajs_dict)
        save_dict_to_text(nb_dict, id2coor_dict, os.path.join(dir_dict[dirkeys[i]], 'neighbor_relation.txt'))

if __name__=='__main__':
    data_dir = args.data_dir
    root_dir = args.root_dir

    sub_dirs = ['full', 'visit_profiles', 'transition_profiles', 'sub', 'obf1', 'obf2', 'cut']
    venue_dir_dict = dict(); discre_dir_dict = dict()
    venue_dir = os.path.join(root_dir, 'venue_level')
    discre_dir = os.path.join(root_dir, 'discre_level')
    for key in sub_dirs:
        venue_dir_dict[key] = os.path.join(venue_dir, key)
        discre_dir_dict[key] = os.path.join(discre_dir, key)
    dir_list = list(venue_dir_dict.values()) + list(discre_dir_dict.values())
    for path in dir_list: mkdir(path)
    
    gps_dict = read_dict(os.path.join(root_dir, 'gps_dict.hdf5'))
    ################ VENUE VERSION ##################################
    print('Working for VENUE version')
    venue_trajs = read_and_construct_full_trace(venue_dir_dict['full'], usrs)
    
    sub_dict_cut, plist_cut = subsample_cut(venue_trajs, None)
    sub_dict_sub, plist_sub = subsample_ran(venue_trajs, None)
    save_subsamples(venue_dir_dict['sub'], sub_dict_sub)
    save_subsamples(venue_dir_dict['cut'], sub_dict_cut)
    save_sample_rate(venue_dir_dict['sub'], plist_sub, usrs)
    save_sample_rate(venue_dir_dict['cut'], plist_cut, usrs)
    id2coor_dict = get_id2coor_dict(venue_trajs, gps_dict)
    save_different_version_obf(['obf1', 'obf2'], venue_dir_dict, venue_trajs, id2coor_dict, usrs)

    ################ DISCRETE VERSION ################################
    print('Working for DISCRETE version')
    token_trajs = read_and_construct_full_trace(discre_dir_dict['full'], usrs)
    
    sub_dict_cut2, plist_cut2 = subsample_cut(token_trajs, None)
    sub_dict_sub2, plist_sub2 = subsample_ran(token_trajs, None)
    save_subsamples(discre_dir_dict['sub'], sub_dict_sub2)
    save_subsamples(discre_dir_dict['cut'], sub_dict_cut2)
    save_sample_rate(discre_dir_dict['sub'], plist_sub2, usrs)
    save_sample_rate(discre_dir_dict['cut'], plist_cut2, usrs)
    id2coor_dict2 = get_id2coor_dict(token_trajs, gps_dict)
    save_different_version_obf(['obf1', 'obf2'], discre_dir_dict, token_trajs, id2coor_dict2, usrs)

    
