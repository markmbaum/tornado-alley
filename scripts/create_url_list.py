from os.path import join

#------------------------------------------------------------------------------

fnout = join('..', 'urllist.txt')

urlbase = 'https://downloads.psl.noaa.gov/Datasets/NARR/'

#single level variables
monolevels = ['cape', 'cdcon', 'dswrf']

#multiple level variables
pressure = ['hgt', 'uwnd', 'vwnd', 'omega', 'air', 'shum']

#------------------------------------------------------------------------------

with open(fnout, 'w') as f:
    f.write('TsvHttpData-1.0\n')
    for year in range(1994, 2022):
        #monolevel variables, split by year
        #see https://psl.noaa.gov/data/gridded/data.narr.monolevel.html
        for x in monolevels:
            f.write(urlbase + 'monolevel/{0}.{1}.nc\n'.format(x, year))
        #variables on the pressure grid, which are split into months
        #see https://psl.noaa.gov/data/gridded/data.narr.pressure.html
        for month in range(1, 13):            
            for x in pressure:
                f.write(urlbase + 'pressure/{0}.{1}{2:02}.nc\n'.format(x, year, month))