# %%

from os.path import join
from copy import deepcopy
from numpy import *
from numpy.random import rand
from pandas import read_feather, Timestamp, Period, DataFrame
from xarray import open_dataset, Dataset, concat

#------------------------------------------------------------------------------
# %% INPUTS

#directory where the tornado alley storm data is located
STORMDIR = join('..', 'data', 'pro')

#directory with 'monolevel' and 'pressure' subdirectories
NCDIR = join('..', 'data', 'test')

#directory for output netcdf files
OUTDIR = join('..', 'data', 'pro', 'inputs')

#variables with three spatial dimensions (on pressure levels)
PVAR = [
    'air',
    'hgt'
]

#variables with two spatial dimensions (flat)
MVAR = [
    'cape',
    'cdcon'
]

#number of latitude cells (full rows) above and below target cell to include
LATOFFSET = 8

#number of longitude cells (columns) left and right of target cell to include
LONOFFSET = (24, 8)

#bounding latitude longitude box of the storm events
MINLAT, MAXLAT = 33, 46
MINLON, MAXLON = -104.2, -89.2

#years and months to include
YEARS = range(2008, 2009)
MONTHS = range(6,8)

#number of non-storm inputs to create for every storm in a given month
NNONSTORM = 5

#------------------------------------------------------------------------------
# %% FUNCTIONS

#construct the path to a monolevel data file from the name and yaer
def monolevel_path(name, year):
    return join(NCDIR, 'monolevel', f'{name}.{year}.nc')

#open a monolevel data file (lazily)
def open_monolevel(name, year):
    return open_dataset(monolevel_path(name, year))

#construct the path to a pressure level data file from the name, month, year
def pressure_path(name, year, month):
    return join(NCDIR, 'pressure', f'{name}.{year}{month:02}.nc')

#open a pressure level data file (lazily)
def open_pressure(name, year, month):
    return open_dataset(pressure_path(name, year, month))

#--------------------------------------

def attrs2df(ds):
    d = dict()
    for k in ds[0].attrs.keys():
        d[k] = [x.attrs[k] for x in ds]
    df = DataFrame(d)
    return df

#write an input Dataset to file
def write_inputs(ds, name, year, month):
    #pack all the attributes into a dataframe
    df = attrs2df(ds)
    #concatenate all the blocks into a single Dataset along a new index
    X = concat(ds, 'index')
    #file name base
    fnb = join(OUTDIR, f'{year}_{month}_{name}_')
    #write attribute table to compressed file
    df.to_feather(fnb + 'attributes.feather')
    #write the input block to netcdf after stripping off attributes
    X.attrs = {}
    X.to_netcdf(fnb + 'inputs.nc')

    return None

#--------------------------------------

#time zero for a given month in a given year
def month_begin(year, month):
    return Timestamp(year=year, month=month, day=1)

def days_in_month(year, month):
    return Period(freq='D', year=year, month=month).days_in_month

#the final reanalysis time for a month, on the 21st hour of the last day
def month_end(year, month):
    return Timestamp(
        year=year,
        month=month,
        day=days_in_month(year, month),
        hour=21
    )

#produces a random (time, lat, lon) coordinate in the reanalysis domain
def random_coordinate(year, month):
    mb = month_begin(year, month)
    me = month_end(year, month)
    f = rand()
    t = mb + f*(me - mb).floor('s') #round to the nearest second
    lat = (MAXLAT - MINLAT)*rand() + MINLAT
    lon = (MAXLON - MINLON)*rand() + MINLON
    return t, lat, lon

def random_coordinates(n, year, month):
    coords = [random_coordinate(year, month) for _ in range(n)]
    t, lat, lon = zip(*coords)
    return array(t), array(lat), array(lon)

def nearest_cell(lat, lon, Lat, Lon):
    #L1 distance because it's fast
    d = abs(lat - Lat) + abs(lon - Lon)
    #use numpy tricks to get the index of closest cell
    ilat, ilon = unravel_index(d.argmin(), d.shape)
    return ilat, ilon

def blank_input(attrs=dict()):
    #start the variable dictionary
    data_vars = dict()
    #monolevel variables
    shp = (2*LATOFFSET, sum(LONOFFSET))
    for name in MVAR:
        data_vars[name] = (['y', 'x'], zeros(shp, dtype=float32))
    #pressure level variables
    shp = (29, 2*LATOFFSET, sum(LONOFFSET))
    for name in PVAR:
        data_vars[name] = (['levels', 'y', 'x'], zeros(shp, dtype=float32))
    #create the xarray dataset instance
    ds = Dataset(
        data_vars=data_vars,
        coords=dict(
            levels=arange(29, dtype=int16),
            y=arange(2*LATOFFSET, dtype=int16),
            x=arange(sum(LONOFFSET), dtype=int16)
        ),
        attrs=dict(attrs)
    )
    return ds

def find_cell(q, V):
    n = len(V)
    #handle boundaries
    assert q >= V[0]
    assert q <= V[-1]
    #bisection search for the containing cell
    L = 0
    H = n
    while H - L > 1:
        M = (H + L) // 2
        if V[M] > q:
            H = M
        else:
            L = M

    return L

def interpolate_monolevel(time, lat, lon, Lat, Lon, M, k, ds):
    #locate indices of nearest grid cell
    ilat, ilon = nearest_cell(lat, lon, Lat, Lon)
    #find the nearest cell in time
    #itime = searchsorted(M[k].coords['time'].values, time)
    itime = find_cell(time, M[k].coords['time'].values)
    #slice the variable
    X = M[k][k][
        itime : itime + 2,
        ilat - LATOFFSET : ilat + LATOFFSET,
        ilon - LONOFFSET[0] : ilon + LONOFFSET[1]
    ]
    #interpolate along the time axis
    t1, t2 = X.coords['time'].values
    assert t1 <= time <= t2
    f = (time - t1).total_seconds()/(3*3600)
    X = (1 - f)*X[0,:,:] + f*X[1,:,:]
    ds[k][:] = X.values
    return None

def interpolate_pressure(time, lat, lon, Lat, Lon, P, k, ds):
    #locate indices of nearest grid cell
    ilat, ilon = nearest_cell(lat, lon, Lat, Lon)
    #find the nearest cell in time
    #itime = searchsorted(M[k].coords['time'].values, time)
    itime = find_cell(time, P[k].coords['time'].values)
    #slice the variable
    X = P[k][k][
        itime : itime + 2,
        :,
        ilat - LATOFFSET : ilat + LATOFFSET,
        ilon - LONOFFSET[0] : ilon + LONOFFSET[1]
    ]
    #interpolate along the time axis
    t1, t2 = X.coords['time'].values
    assert t1 <= time <= t2
    f = (time - t1).total_seconds()/(3*3600)
    X = (1 - f)*X[0,:,:,:] + f*X[1,:,:,:]
    ds[k][:] = X.values
    return None
    
#------------------------------------------------------------------------------
# %% MAIN

#load the table of storms
df = read_feather(join(STORMDIR, 'tornado_alley.feather'))
df.sort_values('time', inplace=True)
df.index = range(len(df))

#swap some column names for convenience
cols = list(df.columns)
for a,b in (('begin_lat', 'lat'), ('begin_lon', 'lon'), ('event_type', 'type')):
    cols[cols.index(a)] = b
df.columns = cols

#strip times of their timezone for compatibility
df['time'] = df['time'].map(lambda x: x.tz_localize(None))

#add year and month columns for convenience
df['year'] = df.time.map(lambda x: x.year)
df['month'] = df.time.map(lambda x: x.month)

#convert the 'location_filled' flag to integers for writing netcdf files
df['location_filled'] = df.location_filled.map(uint8)

#--------------------------------------
# %%

for year in YEARS:

    #load monolevel datasets, which cover whole years, lazily
    M = {name: open_monolevel(name, year) for name in MVAR}

    for month in MONTHS:

        #Load pressure level datasets, which cover single months, lazily
        P = {name: open_pressure(name, year, month) for name in PVAR}

        #slice out storms during this year-month period
        sl = df[(df.year == year) & (df.month == month)].copy()
        sl.sort_values('time', inplace=True)
        L = len(sl)
        sl.index = range(L)

        #create blank inputs for each storm
        storms = [blank_input(row) for (_, row) in sl.iterrows()]

        #create random coordinates for normal (non-storm) conditions
        time, lat, lon = random_coordinates(NNONSTORM*L, year, month)

        #create blank for non-storm inputs, annotated with distance to closest storm (including in time)
        idx = searchsorted(sl.time, time)
        idx[idx >= L] = L - 1
        norms = []
        for i in range(len(idx)):
            j = idx[i]
            dt = abs((time[i] - sl.time[j]).total_seconds())/3600
            dlat = abs(lat[i] - sl.lat[j])
            dlon = abs(lon[i] - sl.lon[j])
            ds = blank_input(dict(
                time=time[i],
                lat=lat[i],
                lon=lon[i],
                type='Non-storm',
                time_close=dt, #time of closest storm [hours]
                lat_close=dlat, #latitude of closest storm [degrees]
                lon_close=dlon, #longitude of closest storm [degrees]
                type_close=sl.type[j] #type of nearest storm
            ))
            norms.append(ds)

        #retrieve the lat-lon grid
        X = next(iter(M.values()))
        Lat, Lon = X.coords['lat'].values, X.coords['lon'].values

        #fill in the blank input blocks
        for k in M:
            #storms
            for i in range(L):
                interpolate_monolevel(sl.time[i], sl.lat[i], sl.lon[i], Lat, Lon, M, k, storms[i])
            for i in range(L*NNONSTORM):
                interpolate_monolevel(time[i], lat[i], lon[i], Lat, Lon, M, k, norms[i])
        for k in P:
            #storms
            for i in range(L):
                interpolate_pressure(sl.time[i], sl.lat[i], sl.lon[i], Lat, Lon, P, k, storms[i])
            for i in range(L*NNONSTORM):
                interpolate_pressure(time[i], lat[i], lon[i], Lat, Lon, P, k, norms[i])
        
        #close all the pressure level datasets
        for k in P:
            P[k].close()

        #write the newly created inputs to file as whole-month blocks
        write_inputs(storms, 'storm', year, month)
        write_inputs(norms, 'non-storm', year, month)
    
    #close all the monolevel datasets
    for k in M:
        M[k].close()
# %%
