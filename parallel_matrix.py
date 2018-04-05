import numpy as np
import multiprocessing as mp
import time
from collections import defaultdict

def flatten_matrix(mat, mode, offset=None):
    w, h = mat.shape
    if mode is 'triu':
        assert not offset is None, 'offset is None'
        indices = np.triu_indices(w, offset)
        return mat[indices], indices
    elif mode is 'full':
        indices = np.indices((w, h))
        flatten_ind = tuple((indices[0].reshape[-1], indices[1].reshape(-1)))
        return np.reshape(mat, -1), flatten_ind
    else: 
        raise ValueError('invalid mode')
        
def pack_matrix(value_array, indices, n, mode):
    assert len(indices[0]) == len(value_array), 'invalid indices or array'
    if mode is 'triu':
        assert n > np.sqrt(len(indices))
        mat = np.zeros((n, n))
        mat[indices] = value_array
        return mat
    elif mode is 'full':
        assert n == np.sqrt(len(indices))
        mat = np.zeros((n,n))
        mat[indices] = value_array
        return mat
    else: 
        raise ValueError('invalid mode')

def parallel_triu(mat, n, func, nProcess, offset=None, dtype=float):
    tick = time.time()
    print('Start multiprocessing')
    
    pool = mp.Pool(processes=nProcess)
    array, ind = flatten_matrix(mat, 'triu', offset)
    chunks = np.array_split(array, nProcess)
    results = pool.map(func, chunks)
    results = np.concatenate(results)
    pool.close()
    pool.join()
    
    print('Time used: ', time.time()-tick)
    return pack_matrix(results, ind, n, 'triu').astype(dtype)

def map_creat_neighbor_dict(args):
    indices, id2coor_dict, ids, d = args 
    d_neighbor_dict = defaultdict(list)
    for (i, j) in indices:
        if is_neighbor(*id2coor_dict[ids[i]], *id2coor_dict[ids[j]], d):
                d_neighbor_dict[ids[i]].append(ids[j])
                d_neighbor_dict[ids[j]].append(ids[i])
    return d_neighbor_dict

def reduce_neighbor_dicts(dicts_list, ids):
    d_neighbor_dict = dict()
    for i in ids:
        tmp = [dict_[i] for dict_ in dicts_list if i in dict_]
        neighbors = np.concatenate(tmp).tolist() if len(tmp)>0 else list()      
        d_neighbor_dict[i] = list(set(neighbors))
    return d_neighbor_dict

is_neighbor = lambda x1, y1, x2, y2, d: ((x1-x2)**2 + (y1-y2)**2)<d

def parallel_neighbors(ids, indices, id2coor_dict, d, nProcess):
    tick = time.time()
    print('Start multiprocessing')
    
    pool = mp.Pool(processes=nProcess)
    chunks = np.array_split(np.stack(indices, axis=1), nProcess)
    results = pool.map(map_creat_neighbor_dict, zip(chunks, [id2coor_dict]*nProcess, [ids]*nProcess, [d]*nProcess))
    d_neighbor_dict = reduce_neighbor_dicts(results, ids)
    pool.close()
    pool.join()
    print('Time used: ', time.time()-tick)
    return d_neighbor_dict