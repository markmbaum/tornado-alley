import xarray as xr
import pandas as pd
import numpy as np
import tensorflow as tf
from os import cpu_count
from os.path import join, isfile
from multiprocessing import Pool

# %% --------------------------------------------------------------------------

#directory with raw inputs
RAWDIR = join('..', 'data', 'raw')

#directory for output converted
OUTDIR = join('..', 'data', 'pro')

#3D variables in the datasets (pressure level)
PVAR = [
    'hgt',
    'uwnd',
    'vwnd',
    'air',
    'omega',
    'shum'
]

YEARS = range(2000, 2022)

MONTHS = range(1, 13)

storm2label = {
    'Non-storm': 0,
    'Tornado': 1,
    'Winter Storm': 2
}

# %% --------------------------------------------------------------------------

filepath = lambda fn: join(RAWDIR, fn)

def convert_dataset(year, month):
    #load storm inputs and attributes
    sin = xr.open_dataset(filepath(f'{year}_{month}_storm_inputs.nc'))
    sat = pd.read_feather(filepath(f'{year}_{month}_storm_attributes.feather'))
    nin = xr.open_dataset(filepath(f'{year}_{month}_non-storm_inputs.nc'))
    nat = pd.read_feather(filepath(f'{year}_{month}_non-storm_attributes.feather'))
    #slice out non-storm instances too close to storms
    b = ((nat.time_close > 60) | (nat.lat_close < 2) | (nat.lon_close < 2))
    idx = np.flatnonzero(b)
    nat = nat.iloc[idx]
    nin = nin.sel({'index': idx})
    #take only tornados and winter storms
    b = ((sat['type'] == 'Tornado') | (sat['type'] == 'Winter Storm')).values
    idx = np.flatnonzero(b)
    sat = sat.iloc[idx]
    sin = sin.sel({'index': idx})
    #check for empty months
    if len(sat) == 0:
        return None, None, None
    #if needed (probably is), remove some non-storms randomly for even number of storms/non-storms
    if len(nat) > len(sat):
        idx = np.arange(len(nat))
        np.random.shuffle(idx)
        idx = idx[:len(sat)]
        nat = nat.iloc[idx]
        nin = nin.sel({'index': idx})
    #combine the inputs and attributes
    inputs = xr.concat([nin, sin], 'index')
    labels = pd.concat([nat, sat])
    #take only the storm types for labels, but keep the strings around
    storm_types = list(labels['type'].values)
    #one-hot encoding for labels
    labels = tf.one_hot(labels['type'].map(storm2label), len(storm2label.keys())).numpy()
    #construct a single array of 3D input fields
    inputs = np.stack([inputs[v].as_numpy() for v in PVAR], axis=-1)

    return inputs, labels, storm_types

def convert_save_dataset(year, month):

    inputs, labels, storm_types = convert_dataset(year, month)
    if inputs is not None:
        np.save(join(OUTDIR, f'{year}_{month}_inputs'), inputs)
        np.save(join(OUTDIR, f'{year}_{month}_labels'), labels)
        with open(join(OUTDIR, f'{year}_{month}_storm_types.txt'), 'w') as ofile:
            ofile.write('\n'.join(storm_types))
        print(f'{year}-{month} complete')
    else:
        print(f'{year}-{month} empty')
    
    return None

# %% --------------------------------------------------------------------------

if __name__ == '__main__':

    nproc = cpu_count()
    pool = Pool(nproc)
    print(f'pool started with {nproc} processes')
    tasks = []

    for year in YEARS:
        for month in MONTHS:
            fns = filepath(f'{year}_{month}_storm_inputs.nc')
            fnn = filepath(f'{year}_{month}_non-storm_inputs.nc')
            if isfile(fns) and isfile(fnn):
                tasks.append(
                    pool.apply_async(
                        convert_save_dataset,
                        (year, month)
                    )
                )
            else:
                print(f'{year}-{month} absent')
    
    #fetch each month's task
    [task.get() for task in tasks]

    pool.close()
