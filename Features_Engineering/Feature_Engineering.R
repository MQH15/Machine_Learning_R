##########################################################################
# Master Data Science: Feature Engineering
##########################################################################

#install.packages(c("nycflights13","DataExplorer"))
library(nycflights13)
library(DataExplorer)

data_list <- list(airlines, airports, flights, planes, weather)
str(data_list)
#structure of the data
plot_str(data_list)
#radial form
plot_str(data_list, type = "r")

merge_airlines <- merge(flights, airlines, by = "carrier", all.x = TRUE)
merge_planes <- merge(merge_airlines, planes, by = "tailnum", all.x = TRUE, suffixes = c("_flights", "_planes"))
merge_airports_origin <- merge(merge_planes, airports, by.x = "origin", by.y = "faa", all.x = TRUE, suffixes = c("_carrier", "_origin"))
final_data <- merge(merge_airports_origin, airports, by.x = "dest", by.y = "faa", all.x = TRUE, suffixes = c("_origin", "_dest"))
plot_str(final_data)


# to "introduce" a new data set:
introduce(final_data)
plot_intro(final_data)

# missing values:
plot_missing(final_data)
final_data <- drop_columns(final_data, "speed")

plot_bar(final_data)

# clean up duplications

final_data[which(final_data$manufacturer == "AIRBUS INDUSTRIE"),]$manufacturer <- "AIRBUS"
final_data[which(final_data$manufacturer == "CANADAIR LTD"),]$manufacturer <- "CANADAIR"
final_data[which(final_data$manufacturer %in% c("MCDONNELL DOUGLAS AIRCRAFT CO", "MCDONNELL DOUGLAS CORPORATION")),]$manufacturer <- "MCDONNELL DOUGLAS"

plot_bar(final_data$manufacturer)

#Feature dst_origin and tzone_origin contains only 1 value, so we should drop them:

final_data <- drop_columns(final_data, c("dst_origin", "tzone_origin"))
plot_bar(final_data, with = "arr_delay")

#To visualize distributions for all continuous features:

plot_histogram(final_data)


#Set flight to categorical, since that is the flight number with no mathematical meaning:

final_data$flight <- as.factor(final_data$flight)

#Remove year_flights and tz_origin since there is only one value:

final_data <- drop_columns(final_data, c("year_flights", "tz_origin"))

plot_correlation(na.omit(final_data), maxcat = 5L)
#For discrete features, the function first dummifies all categories, 
#then calculates the correlation matrix (see cor) and plots it.

pca_df <- na.omit(final_data[, c("origin", "dep_delay", "arr_delay", "air_time", "year_planes", "seats")])

plot_prcomp(pca_df, variance_cap = 1)  

## Reduce data size for demo purpose
arr_delay_df <- final_data[, c("arr_delay", "month", "day", "hour", "minute", "dep_delay", "distance", "year_planes", "seats")]

## Call boxplot function
plot_boxplot(arr_delay_df, by = "arr_delay")

## scatterplots
arr_delay_df2 <- final_data[, c("arr_delay", "dep_time", "dep_delay", "arr_time", "air_time", "distance", "year_planes", "seats")]

plot_scatterplot(arr_delay_df2, by = "arr_delay", sampled_rows = 1000L)  


############################################################################################
# Data Engineering: MIssing values
############################################################################################
# Missing values may have meanings for a feature. 
# Other than imputation methods, we may also set them to some logical values. 
# For example, for discrete features, we may want to group missing values to a new category. 
# For continuous features, we may want to set missing values to a known number 
# based on existing knowledge.
# 
# In DataExplorer, this can be done by set_missing. 
# The function automatically matches the argument for either discrete or continuous features, 
# i.e., if you specify a number, all missing continuous values will be set to that number. 
# If you specify a string, all missing discrete values will be set to that string. 
# If you supply both, both types will be set.

final_df <- set_missing(final_data, list(0L, "unknown"))
plot_missing(final_df)  

############################################################################################
# Data Engineering: Group sparse categories
############################################################################################
group_category(data = final_data, feature = "manufacturer", threshold = 0.2)
#the bottom x% categories to be grouped, e.g., if set to 20%, 
#categories with cumulative frequency of the bottom 20% will be grouped

final_df <- group_category(data = final_data, feature = "manufacturer", 
                           threshold = 0.2, update = TRUE)
plot_bar(final_df$manufacturer)

group_category(data = final_data, feature = "name_carrier", 
               threshold = 0.2, measure = "distance")
final_df <- group_category(data = final_data, feature = "name_carrier", 
                           threshold = 0.2, measure = "distance", update = TRUE)
plot_bar(final_df$name_carrier)


############################################################################################
# Data Engineering: dummy encoding (one.hot)
############################################################################################
#Data dummification is also known as one hot encoding or feature binarization. 
#It turns each category to a distinct column with binary (numeric) values.

plot_str(
  list(
    "original" = final_data,
    "dummified" = dummify(final_data, maxcat = 5L)
  )
)

############################################################################################
# Data Engineering: drop features
############################################################################################
identical(
  drop_columns(final_data, c("dst_dest", "tzone_dest")),
  drop_columns(final_data, c(36, 37))
)

## [1] TRUE
