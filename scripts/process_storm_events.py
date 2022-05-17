from os import listdir
from os.path import join
from pandas import read_csv, concat
from geopandas import read_file
from numpy import isnan

#------------------------------------------------------------------------------

#input data directory
dirin = join('..', 'data',  'raw', 'storm-events', 'extracted')

#columns to keep
cols = [
    'begin_yearmonth',
    'begin_day',
    'begin_time',
    'end_yearmonth',
    'end_day',
    'end_time',
    'state',
    'state_fips',
    'year',
    'event_type',
    'cz_type',
    'cz_name',
    'cz_timezone',
    'injuries_direct',
    'injuries_indirect',
    'deaths_direct',
    'deaths_indirect',
    'damage_property',
    'damage_crops',
    'magnitude',
    'begin_range',
    'begin_azimuth',
    'begin_location',
    'begin_lat',
    'begin_lon'
]

#output file path
fnout = join('..', 'data',  'pro', 'storm_events.csv')

#county shapefile
fncounty = join('..', 'data', 'raw', 'us-counties', 'cb_2018_us_county_20m.shp')

#state fips codes
fnfips = join('..', 'data', 'raw', 'state-fips', 'state_fips.csv')

#------------------------------------------------------------------------------

#load all the individual years of data
storm = []
for fn in listdir(dirin):
    df = read_csv(join(dirin, fn), low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df = df[cols].copy()
    df['state'] = [x.lower() for x in df.state.values]
    storm.append(df)
#combine into a single table
storm = concat(storm, ignore_index=True)

#read in counties and state codes
county = read_file(fncounty)
fips = read_csv(fnfips)

#precompute all county centroid coordinates
county['centroid_lon'] = county.geometry.map(lambda g: g.centroid.x)
county['centroid_lat'] = county.geometry.map(lambda g: g.centroid.y)

#mapping between state code and state name
state2fips = dict(zip(fips.name.values, fips.fips.values))

#some extra massaging
storm.cz_name = [n.lower() for n in storm.cz_name]
county.columns = [c.lower() for c in county.columns]
county.name = [n.lower() for n in county.name]
county.statefp = county.statefp.astype(int)

print("fraction missing location ~", sum(isnan(storm.begin_lon))/len(storm))

#approximate latitude and longitude for each storm event, where missing
for i in storm.index:
    #check if geo coords are missing
    if isnan(storm.at[i,'begin_lon']) and isnan(storm.at[i,'begin_lat']):
        #see if the county and state have a match
        state = storm.at[i,'state']
        if state in state2fips:
            f = state2fips[storm.at[i,'state']]
            c = storm.at[i,'cz_name']
            sl = county[(county.statefp == f) & (county.name == c)]
            #a single match is expected
            if len(sl) == 1:
                storm.at[i,'begin_lon'] = sl.centroid_lon
                storm.at[i,'begin_lat'] = sl.centroid_lat

print("fraction missing location ~", sum(isnan(storm.begin_lon))/len(storm))

storm.to_csv(fnout, index=False)
print("file written:", fnout)