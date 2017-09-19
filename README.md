# Weather

A set of routines useful for getting and plotting Environment Canada bulk weather data. EC keeps historical data (updated daily) for a large number of weather stations across Canada. This bulk data can be obtained manually using terminal commands (wget) and saved as text files for later use. 

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
