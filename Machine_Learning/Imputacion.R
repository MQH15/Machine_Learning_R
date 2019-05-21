install.packages("mice")
library(mice)

#########################################################################
#         example 1: CVD
#########################################################################

dat <- read.csv(url("https://goo.gl/4DYzru"), header=TRUE, sep=",")
head(dat)
sapply(dat, function(x) sum(is.na(x)))
original <- dat

#randomly add some missing values
set.seed(10)
dat[sample(1:nrow(dat), 20), "Cholesterol"] <- NA
dat[sample(1:nrow(dat), 20), "Smoking"] <- NA
dat[sample(1:nrow(dat), 20), "Education"] <- NA
dat[sample(1:nrow(dat), 5), "Age"] <- NA
dat[sample(1:nrow(dat), 5), "BMI"] <- NA

#check where they are
sapply(dat, function(x) sum(is.na(x)))

#check variable type
library(dplyr) 
dat <- dat %>%
  mutate(
    Smoking = as.factor(Smoking),
    Education = as.factor(Education),
    Cholesterol = as.numeric(Cholesterol)
  )

str(dat)

######################### imputation ######################################
library(mice)
init = mice(dat, maxit=0)
meth = init$method
predM = init$predictorMatrix
predM[,"BMI"]=0
# we want to skip a variable for prediction:
meth[c("Age")]=""

# we use different methods to predict each value since the variables are different
meth[c("Cholesterol")]="norm" 
meth[c("Smoking")]="logreg" 
meth[c("Education")]="polyreg"

# now run multiple imputation

set.seed(103)
imputed = mice(dat, method=meth, predictorMatrix=predM, m=5)
#create a data set
imputed <- mice::complete(imputed)

#Lets check how well we did it since we know the actual numbers
# Cholesterol
actual <- original$Cholesterol[is.na(dat$Cholesterol)]
predicted <- imputed$Cholesterol[is.na(dat$Cholesterol)]
# Smoking
actual <- original$Smoking[is.na(dat$Smoking)] 
predicted <- imputed$Smoking[is.na(dat$Smoking)] 
table(actual)
table(predicted)

table(actual,predicted)

#########################################################################
#         example 2: Air quality
#########################################################################
#remove some data
data <- airquality
data[4:10,3] <- rep(NA,7)
data[1:5,4] <- NA

summary(data)
# check the distribution of the missings
data <- data[-c(5,6)]
summary(data)

#remember we could do this much more elegant from the DataExplorer Package
plot_missing(data)

# using a function from mice library:
md.pattern(data)

#or another library:
library(VIM)
aggr_plot <- aggr(data, col=c('navyblue','red'), 
                  numbers=TRUE, sortVars=TRUE, 
                  labels=names(data), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))

#The plot helps us understanding that almost 70% of the samples are not missing any information,
#22% are missing the Ozone value, and the remaining ones show other missing patterns. 

marginplot(data[c(1,2)])

#The red box plot on the left shows the distribution of Solar.R with Ozone missing while the 
#blue box plot shows the distribution of the remaining datapoints. Likewhise for the Ozone 
#box plots at the bottom of the graph.
#If our assumption of MCAR data is correct, then we expect the red and blue box plots to be 
#very similar: i.e. the missing values in one and the other variable are independent from each other

# two types of missing data:
#MCAR: missing completely at random. This is the desirable scenario in case of missing data.

#MNAR: missing not at random. Missing not at random data is a more serious issue and in this 
#case it might be wise to check the data gathering process further and try to understand why 
#the information is missing. For instance, if most of the people in a survey did not answer 
#a certain question, why did they do that? Was the question unclear?

#Even for MCAR if more than 5% of the data is missing, leave the feature out
pMiss <- function(x){sum(is.na(x))/length(x)*100}
# % missing per variable
apply(data,2,pMiss)
#%missing per observation
apply(data,1,pMiss)

data<-mutate(data,missing.obs=apply(data,1,pMiss))

# Observations that are missing 2 or more features (>50%), should be dropped if possible.
data <- filter(data,missing.obs<50)
data<-select(data,-missing.obs)

##### Data imputation using mice
data <- airquality
data[4:10,3] <- rep(NA,7)
data[1:5,4] <- NA

tempData <- mice(data,m=5,maxit=50,meth='pmm',seed=500)
summary(tempData)

tempData$imp$Ozone
tempData$meth
completedData <- mice::complete(tempData,1)
#The missing values have been replaced with the imputed values in the first of the five datasets. 
#If you wish to use another one, just change the second parameter in the complete() function.

xyplot(tempData,Ozone ~ Wind+Temp+Solar.R,pch=18,cex=1)
densityplot(tempData)
stripplot(tempData, pch = 20, cex = 1.2)

########### pooling
# ideally we want to use not a single model but one pooling the results of the five estimations

modelFit1 <- with(tempData,lm(Temp~ Ozone+Solar.R+Wind))
summary(pool(modelFit1))

tempData2 <- mice(data,m=50,seed=245435)
modelFit2 <- with(tempData2,lm(Temp~ Ozone+Solar.R+Wind))
summary(pool(modelFit2))
