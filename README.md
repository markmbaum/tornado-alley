# Tornado Alley

This repo contains a short and somewhat speculative project. It was conceived to continue learning about deep neural networks, larger-scale weather data, and some of the Google Cloud Platform (GCP) tools accessible with free credits. It's also a good demonstration of competence with spatial data and the relevant Python tools.

The idea was to merge high resolution reanalysis products with the record of severe weather in Tornado Alley (USA) and learn 3D atmospheric features likely to produce different types of storms. Reanalysis products are high-resolution datasets representing a blend of observational data and physical models. Both of these datasets are from NOAA and more detailed information is in the [SOURCES.txt](SOURCES.txt) file.

Here are the basic steps involved in this project:

1. *Cleaning up storm locations and times*. Not all of the storm observations have precise latitude and longitude information. They do generally have state and county labels, however. For storms with missing locations, I used a map of all the counties in the US to approximate the storm's location using the containing county's centroid. Storm timing is also in various time zones, so these have to be converted to UTC.

2. *Downloading reanalysis*. The huge 3-hourly NARR reanalysis files (netcdf) for past 20 years were downloaded straight to Google Cloud Storage with a urllist and transfer job.

3. *Generating inputs*. For each storm in tornado alley, a 3D block of atmospheric state is cropped out of the reanalysis files at the time and location of the storm. To do this, I find the reanalysis cell closest to the storm and slice out a block 32 cells wide, 16 cells long, and including all 29 atmospheric levels. Because the reanalysis is on a 3-hourly temporal grid, I interpolate in time for each storm. Each block represents a 4 dimensional model input with three spatial dimensions and 6 channels for the selected atmospheric variables. All of this happened on Google's virtual machines. For each month, I also randomly generate roughly twice as many inputs representing normal (non-storm) atmospheric conditions.

4. *Convolutional network*. After the above steps, there are tens of thousands of cropped atmospheric blocks associated with various storms and normal conditions. As a first step, [I attempted to build a classifier](tornado_alley.ipynb) that could discriminate between these different storms/states using 3D convolutional neural networks. After a bunch of tinkering with different models, large and small, all attempts simply failed. No attempts were able to classify even a small subset of the entire dataset. It appears that the inputs are not distinct enough across the different labels and the low temporal resolution is probably an issue.
