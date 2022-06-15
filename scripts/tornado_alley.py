# %%

from os.path import join
from pandas import read_feather, Timestamp, Timedelta
from numpy import *

# ----------------------------------'-------------------------------------------
# %%

#input file
fnin = join('..', 'data', 'pro', 'storm_events.feather')

#output file
fnout = join('..', 'data', 'pro', 'tornado_alley.feather')

#states to include
states = [
    'south dakota',
    'nebraska',
    'oklahoma',
    'missouri',
    'iowa',
    'arkansas',
    'kansas'
]

#severe weather event types to include
event_types = [
    'Thunderstorm Wind',
    'Hail',
    'Flash Flood',
    'Winter Storm',
    'High Wind',
    'Heavy Snow',
    'Tornado'
]

#------------------------------------------------------------------------------
# %%

def to_timestamp(yearmonth, day, time):
    year = int(str(yearmonth)[:4])
    month = int(str(yearmonth)[-2:])
    hm = str(time)
    minute = int(hm[-2:])
    if len(hm) > 2:
        hour = int(hm[:-2])
    else:
        hour = 0
    ts = Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
    return(ts)

#map to number of hours behind UTC
tz2utc = {
    'CDT': 5,
    'CSC': 6,
    'CST': 6,
    'CST-6': 6,
    'CSt': 6,
    'EST': 5,
    'MDT': 6,
    'MST': 7,
    'MST-7': 7
}

#------------------------------------------------------------------------------
# %%

#read initial data
df = read_feather(fnin)

#make sure the selectors are present
for state in states:
    assert state in df.state.values
for event_type in event_types:
    assert event_type in df.event_type.values

#select states and types of severe weather events
df = df.loc[df.state.isin(states)].copy()
df = df.loc[df.event_type.isin(event_types)]
df.dropna(axis=0, subset=['begin_lat', 'begin_lon'], inplace=True)

#there are a couple of obviously incorrect locations from original data
df = df[(df.begin_lon < -85) & (df.begin_lat < 60)]
df.index = range(len(df))
print(df)

# %%

#create datetimes
tsb = []
tse = []
for i in df.index:
    r = df.loc[i]
    tsb.append(
        to_timestamp(
            r.begin_yearmonth,
            r.begin_day,
            r.begin_time
        )
    )
    tse.append(
        to_timestamp(
            r.end_yearmonth,
            r.end_day,
            r.end_time
        )
    )
df['begin_time'] = tsb
df['end_time'] = tse

# %%

#construct a new column for times in UTC
utc = []
for (ts,tz) in zip(df.begin_time, df.cz_timezone):
    utc.append(
        Timestamp(ts - Timedelta(tz2utc[tz], unit='hr'), tz='UTC')
    )
df['time'] = utc
# %%

#remove unwanted information
df.drop(columns=[
    'begin_yearmonth',
    'begin_day',
    'end_yearmonth',
    'end_day',
    'begin_range',
    'begin_azimuth',
    'state_fips',
    'year',
    'cz_type',
    'cz_name',
    'begin_location'
], inplace=True)

#remove nulls
df.dropna(inplace=True)

# %%

#all done
df.index = range(len(df))
df.reindex(columns=sorted(df.columns)).sort_values('time').to_feather(fnout)
print('file written:', fnout)

# %%
