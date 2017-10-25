# Weather

A set of routines useful for getting and plotting Environment Canada bulk weather data. EC keeps historical data (updated daily) for a large number of weather stations across Canada. This bulk data can be obtained manually using terminal commands (wget) and saved as text files for later use. 

Up-to-date instruction for downloading data from Environment Canada can be found at:
ftp://client_climate@ftp.tor.ec.gc.ca/Pub/Get_More_Data_Plus_de_donnees/ 

Folder: Get_More_Data_Plus_de_donnees > Readme.txt

This is cumbersome, so I created these routines to easily download the data for a particular station, as well as plot the resulting data.

The routines do the following functions:
* download data from Environment Canada
* save data in a consolidated format
* load consolidated data
* update data with data from Environment Canada
* grouping routines to group data by month or year
* smoothing routines to estimate a trendline for the data
* various plotting routines to display the data in useful forms
* miscellatious routines

## Current Work
Cleaning up existing charts to look better.

## Upcoming Changes
* Several more graph types towards the end of the year.
* Update cities.csv to allow multiple stations per city when one station doesn't contain entire data history (eg Calgary).
* ~~I will eventually move the smoothing functions into their own file to make them useful elsewhere.~~ Done
* ~~The initializing weather stations and cities will be put into a text file loaded at the beginning to add more flexibility. Change a file, rather than changing the source code. I may do the same with the basepath variable, or determine from code location.~~
* ~~Considering making a subclass of DataFrames to hold city descriptive info together with the data. May be just simpler to add attributes to data when loading.~~

**Update**: (25Oct17) Much work completed. DataFrames subclassed to WxDF (probably won't subclass Series, since I'm usually doing all work on the original dataframe). This was a challenge since I wasn't creating a blank object on creation and I couldn't find examples that helped. Started a cities.csv table to simplify using with multiple cities. Moved annotation code to a separate Annotate module. Added a chart that shows historical stats for a particular month.

**Update**: (27Sep17) Added padding option to all smoothing functions. This significantly improves performance at the start and end of the data. Padding consists of reflecting the data around the first and last points. This has the side-effect of causing the trend lines to end up being horizontal (zero slope) at the start and end. However, on the test data it doesn't look too bad. Lowess requires significantly more points to get similar performance as the other methods (50 vs 31 point window). Lowess linear and SSA one component are very similar and the least wavy. Weighted Moving Average, Lowess binomial, and SSA two component are very similar, and follow the data more closely.

**Update**: (24Sep17) Singular Spectral Analysis (SSA) added. This was also less useful than expected. The problem is, as usual, dealing with smoothing at ends of data. SSA was worse than other methods, since it seemed to act like data is padded with 0s, which makes the trend line drop to 0 at beginning and end. Theoretically, I can use SSA to predict future points, but for the temperature example I used this is dominated by the trend (1st) reconsistution, which is what drops to 0. I also added a StackedPlot function to make it easier to see all the returned reconsituted components.

     df = LoadDF()
     yf = GetYear(df, cols=[8])
     rc = SSA(yf, 32, allRC=True)
     StackPlot(rc, cols=4)

**Update**: Polynomial regression has been added to lowess, but it is less useful than I expected. Higher orders allow more peaks and valleys, so there is actually *less* smoothing of the data. This could be useful if you have an idea of the shape of the data (order = # of peaks and values + 1) and use a large window. I tried removing the weighting function, but this added lots of discontinuities.




