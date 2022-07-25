# Tornado Alley

This repo contains a short and somewhat speculative project that was mostly intended to help me learn deep learnign practicalities and demonstrate basic data handling competence, in particular using some of the google cloud platform tools.

The idea was to merge high resolution reanalysis products with the record of severe weather in tornado alley, seeing if 3D properties conducive to storms can be learned. Reanalysis products are high-resolution datasets representing a blend of observational data and physical models. Both of these datasets are from NOAA and some more detailed information is in the [SOURCES.txt](SOURCES.txt) file.

Here are the basic steps:

1. First, not all of the storm observations have precise latitude and longitude information. They do generally have state and county labels, however. For storms with missing locations, I used a map of all the counties in the US to approximate the storm's location using the containing county's centroid. Storm timing is also in various time zones, so these have to be converted to UTC.

2. The huge 3-hourly NARR reanalysis files (netcdf) for past 20 years were downloaded straight to Google Cloud Storage with a urllist and transfer job.

3. For each storm in tornado alley, a 3D block of atmospheric state is cropped out of the reanalysis files. To do this, I find the reanalysis cell closest to the storm and slice out a block 32 cells wide, 16 cells long, and including all 29 atmospheric levels. Because the reanalysis is on a 3-hourly temporal grid, I interpolate in time for each storm. These block represent the modeling inputs. All of this happened on Google's virtual machines.

4. For each month, I generate roughly twice as many inputs from normal atmospheric conditions randomly. Picking random locations in tornado alley and random times within the month, the same cropping procedure takes place.

After these steps, we have tens of thousands of cropped atmospheric blocks associated with various storms or with normal conditions. As a first step, [I attempted to build a classifier](tornado_alley.ipynb) that could discriminate between these different storms/states using 3D convolutional neural networks. I'm not terribly surprised, but it failed. It appears that the inputs are not distinct enough across the different labels. The primary reason for this is probably the temporal resolution.