import numpy as np
import dask.array as da

def zeros(shape, backend='numpy', dtype=float):
    ret = np.zeros(shape, dtype = dtype)
    if backend == 'numpy' or backend == 'n':
        return ret
    if backend == 'dask' or backend == 'da':
        return da.from_array(ret)
    
def zeros_like(arr):
    if isinstance(arr, np.ndarray):
        return np.zeros_like(arr)
    if isinstance(arr, da.core.Array):
        return da.zeros_like(arr)
    
def ones(shape, backend='numpy', dtype=float):
    ret = np.ones(shape, dtype = dtype)
    if backend == 'numpy' or backend == 'n':
        return ret
    if backend == 'dask' or backend == 'da':
        return da.from_array(ret)
    
