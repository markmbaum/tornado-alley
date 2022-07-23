import xarray as xr
import pandas as pd
import numpy as np
import tensorflow as tf
from os.path import join, isfile
from multiprocessing import Pool

# %% --------------------------------------------------------------------------

#directory with raw inputs
RAWDIR = join('..', 'data', 'raw-dataset')

#directory for output converted
OUTDIR = join('..', 'data', 'converted-dataset')

#3D variables in the datasets (pressure level)
PVAR = [
    'hgt',
    'uwnd',
    'vwnd',
    'air',
    'omega',
    'shum'
]

#mapping storm strings to numeric indices
storm2label = {
    'Non-storm': 0,
    'Thunderstorm Wind': 1,
    'Hail': 2,
    'Flash Flood': 3,
    'Winter Storm': 4,
    'High Wind': 5,
    'Heavy Snow': 6,
    'Tornado': 7
}

YEARS = range(2000, 2022)

MONTHS = range(1, 13)

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
    #also, if needed, remove some non-storms randomly for even numbers of each
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
    labels = tf.one_hot(labels['type'].map(storm2label), 8).numpy()
    #construct a single array of 3D input fields
    inputs = np.stack([inputs[v].as_numpy() for v in PVAR], axis=-1)

    return inputs, labels, storm_types

def convert_save_dataset(year, month):

    inputs, labels, storm_types = convert_dataset(year, month)
    np.save(join(OUTDIR, f'{year}_{month}_inputs'), inputs)
    np.save(join(OUTDIR, f'{year}_{month}_labels'), labels)
    with open(join(OUTDIR, 'storm_types.txt'), 'w') as ofile:
        ofile.write('\n'.join(storm_types))
    print(f'{year}-{month} complete')
    
    return None

# %% --------------------------------------------------------------------------

if __name__ == '__main__':

    pool = Pool()
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
