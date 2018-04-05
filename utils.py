import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import h5py
import pickle
import os
from collections import defaultdict, Counter

def get_discretized_coor(x, y, w_min, w_max, h_min, h_max, width, height):
    """
    INPUT: (x,y) from GPS, which is croped by ((w_min, w_max), (h_min, h_max)), and then represent in a coordination system 
    which is capped by(width, height)
    OUTPUT: discretized Cart coor (scaled_x, scaled_y)
    """
    return ( min(np.floor(((x-w_min) / (w_max-w_min)) * width).astype(int), width-1), \
             min(np.floor(((y-h_min) / (h_max-h_min)) * height).astype(int), height-1) )

cart_2_token = lambda x, y, w, h: y*w + x #this pair of function has been tested recoverable from each other
token_2_cart = lambda token, w, h: (token%w, token//w) #this pair of function has been tested recoverable

def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass
    
def read_dict(path):
    res_dict = dict()
    with h5py.File(path, 'r') as f:
        for k in f.keys():
            res_dict[int(k)] = f.get(k).value
    return res_dict

def save_dict(path, dict_):
    with h5py.File(path, 'w') as f:
        for k, v in dict_.items():
            f.create_dataset(str(k), data=v)
            
def read_pickle(path, dtype=None):
    f = open(path, 'r') if dtype is str else open(path, 'rb') 
    thing = pickle.load(f)
    f.close()
    return thing

def save_pickle(path, thing, dtype=None):
    f = open(path, 'w') if dtype is str else open(path, 'wb') 
    pickle.dump(thing, f)
    f.close()
    
def read_and_construct_full_trace(full_path, usrs):
    def read_full_trace(fname):
        with open(fname, 'r') as f:
            tmplist = f.read().splitlines()
        return list(map(int, tmplist))

    res_dict = dict()
    for usr in usrs:
        fname = os.path.join(full_path, str(usr)+'.txt')
        res_dict[usr] = read_full_trace(fname)
    return res_dict

def read_and_construct_visit_profiles(visit_path, usrs):
    def read_visit_profile(fname):
        with open(fname, 'r') as f:
            profile = dict()
            records = f.readlines()
            for r in records:
                k,v = r.split(', ')
                profile[int(k)] = float(v.strip())
        return profile

    res_dict = dict()
    for usr in usrs:
        fname = os.path.join(visit_path, str(usr)+'.txt')
        res_dict[usr] = read_visit_profile(fname)
    return res_dict

def read_transition_profile(fname):
    profile = dict()
    tmplist = list()
    with open(fname, 'r') as f:
        records = f.readlines()
        for r in records:
            l1, l2, p = r.strip().split(', ')
            tmplist.extend([int(l1), int(l2)])
            
    with open(fname, 'r') as f:
        counter = Counter(tmplist)
        N_states = len(counter)
        ck2id_dict = dict()
        id2ck_dict = dict()
        for k in sorted(counter):
            ck2id_dict[k] = len(ck2id_dict)
            id2ck_dict[ck2id_dict[k]] = k 
        
        matrix = np.zeros(shape=(N_states, N_states))
        records = f.readlines()
        for r in records:
            l1, l2, p = r.strip().split(', ')
            matrix[ck2id_dict[int(l1)], ck2id_dict[int(l2)]] = float(p)
    return matrix ,ck2id_dict, id2ck_dict

def read_and_construct_ground_truth(path):
    res_dict = dict()
    path = os.path.join(path, 'mapping.txt') if not 'mapping.txt' in path else path
    with open(path, 'r') as f:
        records = f.readlines()
        for r in records:
            gt, nym = r.split(', ')
            res_dict[int(nym)] = int(gt)
    return res_dict

def save_subsamples(path, sub_dict):
    for usr, seq in sub_dict.items():
        with open(os.path.join(path, str(usr)+'.txt'), 'w') as f:
            for p in seq:
                f.write(str(p)+'\n')

def save_sample_rate(path, plist, usr_trajs):
    with open(os.path.join(path, 'sample_rate.txt'), 'w') as f:
        for i, usr in enumerate(usr_trajs):
            f.write(str(usr) + ', ' + str(plist[i]) + '\n')

def read_sample_rate(path, selected_usrs=None):
    pdict = dict()
    path = os.path.join(path, 'sample_rate.txt') if not 'sample_rate.txt' in path else path
    with open(path, 'r') as f:
        records = f.readlines()
        for r in records:
            usr, p = r.split(', ')
            if not selected_usrs is None:
                if int(usr) in selected_usrs:
                    pdict[int(usr)]=float(p.strip())
            else:
                pdict[int(usr)]=float(p)
    return pdict

def read_and_construct_neighbor_dict(path):
    path = os.path.join(path, 'neighbor_relation.txt') if not 'neighbor_relation.txt' in path else path
    d_neighbor_dict = dict()
    with open(path, 'r') as f:
        records = f.readlines()
        for r in records:
            line = r.replace(',','').strip().split()
            d_neighbor_dict[int(line[0])] = set(map(int, line[1:]))
    return d_neighbor_dict

def read_and_construct_usr(path):
    usrs = list()
    path = os.path.join(path, 'users.txt') if not 'users.txt' in path else path
    with open(path, 'r') as f:
        records = f.readlines()
        for r in records:
            usrs.append(int(r.strip()))
    return usrs

def reverse_shuffle_list(nyms):
    tmp = np.zeros(len(nyms))
    for i, v in enumerate(nyms):
        tmp[v] = i
    return tmp.astype(int)

def compute_prob(seq):
    counter = Counter(seq)
    N_states = len(counter)
    ck2id_dict = dict()
    id2ck_dict = dict()
    for k in sorted(counter):
        ck2id_dict[k] = len(ck2id_dict)
        id2ck_dict[ck2id_dict[k]] = k 
        
    matrix = np.zeros(shape=(N_states, N_states))
    for i in range(len(seq)-1):
        matrix[ck2id_dict[seq[i]], ck2id_dict[seq[i+1]]] += 1
    
    base = np.sum(matrix, axis=1) 
    for i in range(N_states):
        if not base[i] == 0:
            matrix[i] /= float(base[i]) 
    return matrix, ck2id_dict, id2ck_dict

accuracy = lambda l,p: np.mean(l==p)

def make_sym_mat(mat):
    i_lower = np.tril_indices(mat.shape[0], -1)
    mat[i_lower] = mat.T[i_lower]
    return mat
