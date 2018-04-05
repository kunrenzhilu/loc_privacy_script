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
from stage_2 import *
from parallel_matrix import *

def save_perturbed_samples(path, sub_vp, sub_mc, sub_dict, sub_plist, final_usrs, gt_vp_path=None, gt_mc_path=None):
    #save both the mapping, the visit profiles and the sub samples
    def mkdir(path):
        try: os.makedirs(path)
        except: pass
        return path
    size = len(sub_vp)
    sample_dir = mkdir(os.path.join(path, 'samples'))
    vprfl_dir = mkdir(os.path.join(path, 'visit_profiles'))
    if not gt_mc_path is None: 
        mcprfl_dir = mkdir(os.path.join(path, 'transition_profiles'))
    
    mapping = get_permutations(size)
    for gt, nym in mapping:
        if (gt_vp_path is None) and (not sub_vp is None):
            with open(os.path.join(vprfl_dir, str(gt))+'.txt', 'w') as f:
                for u, p in sub_vp[gt].items():
                    f.write(str(u) + ', ' + str(p)+'\n')
        elif (not gt_vp_path is None) and (sub_vp is None):
            print('Copying visit profiles {}/{}'.format(gt, size))
            os.system('cp {} {}'.format(os.path.join(gt_vp_path, str(final_usrs[gt])+'.txt'), \
                                        os.path.join(vprfl_dir, str(gt))+'.txt', 'w'))
        else: raise ValueError()
                      
        if (not gt_mc_path is None) and (not sub_mc is None):
            print('Not implemented yet')
        elif (not gt_mc_path is None) and (sub_mc is None):
            print('Copying transition profiles {}/{}'.format(gt, size))
            os.system('cp {} {}'.format(os.path.join(gt_mc_path, str(final_usrs[gt])+'.txt'), \
                                        os.path.join(mcprfl_dir, str(gt))+'.txt', 'w'))
        else: print('Not saving mc')
        with open(os.path.join(sample_dir, str(nym))+'.txt', 'w') as f:
            for p in sub_dict[gt]:#write the sub trajs in different name
                f.write(str(p)+'\n')
    with open(os.path.join(path,'mapping.txt'), 'w') as f:
        for gt, nym in mapping:
            f.write(str(gt)+', '+str(nym)+'\n')
    with open(os.path.join(path, 'sample_rate.txt'), 'w') as f:
        for gt, _ in mapping:
            f.write(str(gt) + ', ' + str(sub_plist[gt])+ '\n')
    with open(os.path.join(path, 'users.txt'), 'w') as f:
        for gt, _ in mapping:
            f.write(str(final_usrs[gt]) + '\n')
    print('Saved samples dir: {}, visit prfls dir: {}'.format(sample_dir, vprfl_dir))
    
def mv_and_save_obf_samples(path, final_usrs, gt_vp_path, gt_mc_path, gt_obf_path):
#     from subprocess import call
    def mkdir(path):
        try: os.makedirs(path)
        except: pass
        return path
    
    call = os.system
    size = len(final_usrs)
    mapping = get_permutations(size)
    vprfl_dir = mkdir(os.path.join(path, 'visit_profiles'))
    mcprfl_dir = mkdir(os.path.join(path, 'transition_profiles'))
    sample_dir = mkdir(os.path.join(path, 'samples'))
    assert os.path.isdir(path)
    assert os.path.isfile(os.path.join(gt_vp_path,'1.txt')), 'vp file does not exist'
    assert os.path.isfile(os.path.join(gt_mc_path,'1.txt')), 'mc file does not exist'
    assert os.path.isfile(os.path.join(gt_obf_path, '1.txt')), 'obf file does not exist'
    
    for gt, nym in mapping:
        usr= final_usrs[gt]
        call('cp {} {}'.format(os.path.join(gt_vp_path, str(usr)+'.txt'), \
                     os.path.join(vprfl_dir, str(gt)+'.txt')))
        call('cp {} {}'.format(os.path.join(gt_mc_path, str(usr)+'.txt'), \
                     os.path.join(mcprfl_dir, str(gt)+'.txt')))
        call('cp {} {}'.format(os.path.join(gt_obf_path, str(usr)+'.txt'), \
                     os.path.join(sample_dir, str(nym)+'.txt')))#perturbed
    with open(os.path.join(path,'mapping.txt'), 'w') as f:
        for gt, nym in mapping:
            f.write(str(gt)+', '+str(nym)+'\n')
    sample_rate = read_sample_rate(gt_obf_path, final_usrs)
    with open(os.path.join(path, 'sample_rate.txt'), 'w') as f:
        for gt, _ in mapping:
            f.write(str(gt) + ', ' + str(sample_rate[final_usrs[gt]])+ '\n')
    with open(os.path.join(path, 'users.txt'), 'w') as f:
        for gt, _ in mapping:
            f.write(str(final_usrs[gt]) + '\n')
    call('cp {} {}'.format(os.path.join(gt_obf_path, 'neighbor_relation.txt'),\
                           os.path.join(path, 'neighbor_relation.txt')))
    
    print('save_path: {}, obf_path: {}'.format(path, gt_obf_path))

def get_sub_dict_and_reorder(sub_ind, origin):
    if type(origin) is dict:
        return {i:origin[k] for i, k in enumerate(sub_ind)}
    elif type(origin) is list:
        return [origin[k] for i, k in enumerate(sub_ind)]
    else: raise ValueError()

def get_permutations(size):
    per = np.random.permutation(size)
    return [(i, per[i]) for i in range(size)]

"""Trick, a.k.a. the deterministic attack"""
def trick(cur_prfl, sub_prfl):
        return set(sub_prfl).issubset(set(cur_prfl))

def get_trick_mat(usrs, selected_sub, visit_profile_dict):
    lenu = len(usrs)
    mat = np.zeros((lenu, lenu))
    for i in trange(lenu):
        ui = usrs[i]
        for j in range(i+1, lenu):
            uj = usrs[j]
            mat[i,j] = trick(visit_profile_dict[ui],get_visit_profile({uj:selected_sub[uj]})[uj])
    return mat

def get_visit_profile_vector(vp, nnode):
    res_dict = dict()
    for usr, p in vp.items():
        tmp_vec = np.zeros(nnode)
        for k, v in p.items():
            tmp_vec[k] = v
        res_dict[usr] = tmp_vec
    return res_dict

def get_graph(vpv, usrs):
    lenu = len(usrs)
    mat = np.zeros((lenu, lenu))
    lt = list()
    for i in trange(lenu):
        for j in range(i+1, lenu):
            cos = np.sum(vpv[usrs[i]]*vpv[usrs[j]])
            mat[i,j] = cos
            lt.append([i,j,cos])
    return mat, lt

def get_clustered_usrs_from_affinity_mat(mat, k, usrs):
    return get_selected_usrs_from_predicts(get_cluster(mat, k), k, usrs)

def get_cluster(mat, k, is_sym=False):
    from sklearn import cluster
    if not is_sym: mat = make_sym_mat(mat)
    spectral = cluster.SpectralClustering(
        n_clusters=k, eigen_solver='arpack',
        affinity="precomputed")
    tick = time.time()
    spectral.fit(mat)
    print(time.time()-tick)
    return spectral.labels_

def get_selected_usrs_from_predicts(labels, k, usrs):
    res_dict = defaultdict(list)
    for i, usr in enumerate(usrs):
        res_dict[labels[i]].append(usr)
    return res_dict

def get_selected_subsamples(sample_func, clusters, trajs_dict, visit_profile, Nsample, false_rate=80):
    """A heuristic function to select the profiles such that they are immute from the deterministic attack, it's not guarantee to converge, so it might takes pretty long time. """
    print('The desired false rate is %f'%(false_rate/Nsample))
    crter = 0
    done_first_round = False
    nclusters = len(clusters)
    
    print('Start the first selection until the number of potential profiles is more than Nsample')
    while crter < Nsample:
        i = np.random.choice(range(nclusters))
        if len(clusters[i]) > Nsample*5 or len(clusters[i]) < Nsample: continue
        # try sampling
        selected_spl, plist_spl = sample_func(trajs_dict, plist=None, usrs=clusters[i])
        # do the deterministic attack
        a2 = get_trick_mat(clusters[i] , selected_spl, visit_profile)
        nonzero_list = [np.sum(np.count_nonzero(ai))>=1 for ai in make_sym_mat(a2)] 
        crter = np.sum(nonzero_list)
    
    print('Finish the first round selection, %d candidates are selected from cluster %d'%(crter, i))
    round_one_usrs = np.array(clusters[i])[nonzero_list]
    
    crter2 = 0; len_rone = len(round_one_usrs)
    print('Start the second selection until false rate %f'%(false_rate/Nsample))
    while crter2 < false_rate:
        final_selected_usrs = round_one_usrs[np.random.choice(len_rone, Nsample, replace=False)]
        tmp = get_trick_mat(final_selected_usrs, selected_spl, visit_profile)
        crter2 = np.sum([np.sum(np.count_nonzero(ai))>=1 for ai in make_sym_mat(tmp)])
    print('Final false rate for deterministic attack%f'%(crter2/Nsample))
    return selected_spl, final_selected_usrs, plist_spl

if __name__ == '__main__':

    Ncluster = args.num_cluster #default to be 3
    Nsample  = args.num_sample #default to be 100
    Nobf_sample = args.num_obf_sample #default to be 20
    data_dir = args.data_dir
    root_dir = args.root_dir
    select_dir = os.path.join(root_dir, 'selected_data')
    
    sub_dirs = ['full', 'visit_profiles', 'transition_profiles', 'sub', 'obf1', 'obf2', 'cut']
    venue_dir_dict = dict(); discre_dir_dict = dict()
    venue_dir = os.path.join(root_dir, 'venue_level')
    discre_dir = os.path.join(root_dir, 'discre_level')
    for key in sub_dirs:
        venue_dir_dict[key] = os.path.join(venue_dir, key)
        discre_dir_dict[key] = os.path.join(discre_dir, key)
    select_dir_dict = {name: os.path.join(select_dir, name) for name in ['sub', 'cut', 'obf1', 'obf2']}
    for path in select_dir_dict.values(): mkdir(path)
   
    """
    NOTED: ONLY SUPPORT FOR THE DISCRETE VERSION
    """
    token_trajs = read_and_construct_full_trace(discre_dir_dict['full'], usrs)
    visit_profile_dict2 = read_and_construct_visit_profiles(discre_dir_dict['visit_profiles'], usrs)
    vpv2 = get_visit_profile_vector(visit_profile_dict2, 10000) #totally 10000 locations in this dataset.
    print('Getting affinity matrix')
    aff_mat2, _ = get_graph(vpv2, usrs) #get the affinity matrixa
    
    """Using the spectrum clustering, cluster the graph into 3 categories, and the n pick the two clusters with the lesat number of vertices. Thus, points within these two cluster are reckoned to be the profiles that are close to each other, and thus has higher probabilities to immute from the deterministic attack. This is a very heuristic method, and the why this is a result from try and error"""
    print('Clustering')
    clusters = get_clustered_usrs_from_affinity_mat(aff_mat2, Ncluster, usrs)
    for i, name in enumerate(['sub', 'cut']):
        print('Working for %s'%name)
        selected_spl, final_selected_usrs, plist = get_selected_subsamples(subsample_ran, clusters, token_trajs, visit_profile_dict2, Nsample)
        save_perturbed_samples(path=select_dir_dict[name], \
            sub_vp=get_sub_dict_and_reorder(final_selected_usrs, visit_profile_dict2), \
            sub_mc=None,
            sub_dict=get_sub_dict_and_reorder(final_selected_usrs, selected_spl ), \
            sub_plist=get_sub_dict_and_reorder(final_selected_usrs, plist.tolist()),\
            final_usrs=final_selected_usrs,\
            gt_vp_path=None,\
            gt_mc_path=discre_dir_dict['transition_profiles'])

    print('#'*100)
    print('Working for obfuscated samples')
    for i in range(1,3):
        name = 'obf%d'%i
        mv_and_save_obf_samples(path=select_dir_dict[name],\
            final_usrs=np.random.choice(usrs, Nobf_sample, replace=False),\
            gt_vp_path=discre_dir_dict['visit_profiles'],\
            gt_mc_path=discre_dir_dict['transition_profiles'],\
            gt_obf_path=discre_dir_dict[name])


    
	






    


