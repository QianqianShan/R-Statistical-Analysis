setwd("~/Dropbox/R-Statistical-Analysis/The-R-Book/therbookdata")
getwd()
##Chapter 1 
###Get started

citation()
contributors()

##get help 
help.search(ggplot2)
find("lowess")
apropos("lm")

#see a worked example
example(lm)

##see demenstrations of r functions
demo(persp)
demo(graphics)
demo(Hershey)
demo(plotmath)


## Install packages

#lattice graphics for panel plot or trellis (gezi) graphs
install.packages("lattice")

# Modern applied statistics using s plus
install.packages("MASS")

##generalized additive model
install.packages("mgcv")

# mixed effects model both linear and non-linear
install.packages("nlme")

# feed forward neural networks and multinomial log linear models
install.packages("nnet")

# functions for kriging and point pattern analysis
install.packages("spatial")

# survival analysis including penalized likelihood
install.packages("survival")

## get contents in the packages###
###################################

library(help=spatial)

install.packages("akima")
install.packages("boot")
install.packages("car")
install.packages("lme4")
install.packages("meta")
install.packages("mgcv")
install.packages("deSolve")
install.packages("R2jags")
install.packages("RColorBrewer")
install.packages("RODBC")
install.packages("rpart")
install.packages("spatstat")
install.packages("spdep")
install.packages("tree")


##change the look of the R screen

objects()
search()

#remove (rm) any variables names you have created 
rm(x)

detach()
## The detach command does not make the dataframe called worms disappear;
# it just means that the variables within worms, such as Slope and Area, are no longer accessible directly by name. 
#To get rid of everything, including all the dataframes, type

rm(list=ls())


##Chapter 2: essentials of the R language

2+3;4+5;

#complex number
z<-3.5+8i
Re(z)
Im(z)
#Modulus of complex number
Mod(z)

#argument 
Arg(z)

Arg(z)*180/pi

#conjugate
Conj(z)

is.complex(z)

as.complex(z)

#Rounding 

floor(5.7)
ceiling(5.7)

signif(3.45678,3)

#log to the base e
log(10)

#log to the base 10
log10(10)

# log with other bases

log(9,3) # log to base 3 of 9

# other mathmatical functions

choose(10,3) #choose (n,x), binomial coefficients

lgamma(10) #natural log of gamma(x)

trunc(3.4) # closest integer to 0

acos(1/2)*180/pi

?acosh

119%/%9 #quotient (shangshu)

119%%9 #remainder,modulo

## set up integer array

x<-c(5,3,7,8)
class(x)
x<-as.integer(x)
is.integer(x)


##Logical operations
T==F

##usage of all.equal
#unlike "identical", all.equal allows for insignificant differences
x<-.3-.2
y<-.1
all.equal(x,y)
#summarize differences using all.equal

#combination of T&F

x<-c(NA,TRUE,FALSE)
names(x)<-as.character(x)
names(x)

outer(x,x,"&") #outer product of arrays
outer(x,x,"|")

sequence(c(4,3,4,4,4,5)) #first generate seq(1,4,1),then seq(1,3,1)...
#generate a vector make up of sequence of unequal lengths

#genearate repeats,differentiate the following commands

rep(1:4,2)

rep(1:4,each=2)

rep(1:4,each=2,times=3)

rep(1:4,1:4)

rep(1:4,c(4,1,4,2))

rep(c("cat","goldfish","dog","rat"),c(3,2,5,1))

##generate factor levels by "gl(generate levels)"



















#######################################
## Chapter 22: Bayesian statistics
#######################################
install.packages("rjags")
install.packages("R2jags")
install.packages("coda")


# Install jags from sourceforge if there were an error saying that .so cannot be loaded. 
########MCMC for simple linear regresssion
data2<-read.table("regression.txt",header=T)
attach(data2)
head(data2)

library(R2jags)


# tell jags the name of variables
growth
N=9
data.jags<-list("growth","tannin","N")
data.jags
model<-jags(data=data.jags,parameters.to.save=c("a","b","tau"),
            n.iter=100000,model.file="~/Dropbox/R-Statistical-Analysis/The-R-Book/therbookdata/regression.bugs.txt",
            n.chains=3)

model
plot(model)


## use functions from coda to get attractive graphical output

model.mcmc<-as.mcmc(model)
head(model.mcmc)

library(lattice)
densityplot(model.mcmc)



#########MCMC for a mode with temporal pseudoreplication
## Repeated measures

data<-read.table("fertilizer.txt",header=T)
head(data)
attach(data)
dim(data)
data

y<-root
N=12
T=5
dim(y)<-c(5,12)
y<-t(y)
y
x<-week[1:5]
x
data.jags<-list("y","x","N","T")

model<-jags(data=data.jags,parameters.to.save=c("alpha","beta","tau.c","alpha.c","alpha.tau","beta.c","beta.tau"),
            n.iter=100000,model.file="bayes.lme.txt",
            n.chains=3)
model



############
# MCMC for a model with binomial errors
data<-read.table("~/Dropbox/R learning/therbook/germination.txt",header=T)
data
attach(data)
dim(data)

N<-21
n<-sample
r<-count
x1<-Orobanche
x2<-extract

data.jags<-list("r","n","x1","x2","N")

model<-jags(data=data.jags,parameters.to.save=c("alpha0","alpha1","alpha2","alpha12","tau"),
            n.iter=100000,model.file="~/Dropbox/R learning/therbook/bayes.glm.txt",
            n.chains=3)
model






###################################
## Chapter 23: Tree models 
###################################
## grep :search for matches to argument pattern within each element of a 
#character vector: they differ in the format of and amount of detail in the results.
# grep(pattern, x, ignore.case = FALSE, perl = FALSE, value = FALSE,
#fixed = FALSE, useBytes = FALSE, invert = FALSE)

# deparse : Turn unevaluated expressions into character strings.


# substitute: substitute returns the parse tree for the (unevaluated) expression expr, substituting any variables bound in env
# substitute(expr, env)

# computationaly intensive methods that are used in situations where there are 
#many explanatory variables

## virtues of tree models
# simple, excellent for initial data inspection,give clear picture of the 
#structure of the data,provide highly intuitive insight into kind of 
#interactions between variables

install.packages("tree")
library(tree)
pollute<-read.table("Pollute.txt",header=T)
head(pollute)
attach(pollute)
names(pollute)
model<-tree(pollute)
plot(model)
text(model)

# split maximally distinguishes the response variable in the left and right branches
# variable explaining the greatest amount of the deviance in y is selected

low<-(Industry<748)
tapply(Pollution,low,mean)
plot(Industry,Pollution, pch=16)
#pch: plotting character: symbol to use
abline(v=748,lty=2)
lines(c(0,748),c(24.92,24.92))
lines(c(748,max(Industry)),c(67,67))

# regression tree
# fit multiple regression, forward selection of variables
model<-tree(Pollution~.,pollute)
print(model)


# use rpart "recursively partitioning" to fit tree models
attach(pollute)

par(mfrow=c(1,2))
#use either mfrow or layout for multiple panels in one plot

library(rpart)
model<-rpart(Pollution~.,data=pollute)
plot(model,margin=0.2)
text(model)
library(tree)
model<-tree(pollute)
plot(model)
text(model)
dev.off()


## Tree models as regressions: compare with linear regression

car.test<-read.table("car.test.frame.txt",header=T)
attach(car.test)
names(car.test)

#bg: fill color; col: border color
model<-tree(car.test)
plot(model)
text(model)

library(rpart.plot)
model<-rpart(Mileage~Weight,car.test,cp=.02)

## Use margin to avoid text cut off
plot(model,uniform=T,margin=0.2)
text(model)
print(model)

plot(Weight,Mileage,pch=21,col="brown",bg="green")
a<-mean(Mileage[Weight<2567.5])
b<-mean(Mileage[Weight>=2567.5])
lines(c(1500,2567.5,2567.5,4000),c(a,a,b,b))

# model simplication 

## prue.tree: determines a nested sequence of the sub trees of the supplied tree by recursively 
# snipping off the least important splits

model<-tree(Pollution~.,pollute)
print(model)

#
prune.tree(model)

# the first value of k defaults to be -Inf

# plot it

plot(prune.tree(model))

#specify the number of nodes to which you want the tree to be pruned 
model2<-prune.tree(model,best=4)
plot(model2)
text(model2)


# classification trees for categorical explanatory variables

epil<-read.table("epilobium.txt",header=T)
epil
str(epil)
attach(epil)
names(epil)


# produce the taxonomic keys

model<-tree(species~.,epil,mindev=1e-6,minsize=2)
plot(model)
text(model,cex=0.7)

## the main principal is : find the characters that explain the most of the variation
## and use these to split the cases into roughly equal sized groups at each dichotomy

print(model)
prune.tree(model) # not working ?


##### classification tree for replicated data

tax<-read.table("taxonomy.txt",header=T)
tax
names(tax)
attach(tax)

model<-tree(Taxon~.,tax)
plot(model)
text(model)
summary(model)

# have 4 nodes, 3 dichotomies

print(model)

## plot for classification tree: partition.tree, but only limited to have two 
##explanatory variables

model2<-tree(Taxon~Sepal+Leaf,tax)
partition.tree(model2)
points(Sepal,Leaf) # lay out the points

### label the points
# ifelse(test, yes, no)
# use double equal here

label<-ifelse(Taxon=="I","a",ifelse(Taxon=="II","b",ifelse(Taxon=="III","c","d")))
text(Sepal,Leaf,label,col=1+as.numeric(factor(label)))
# 



###### test for existence of humps
# tree models can be used in assessing whether or not there is a hump between y and x
# can distinguish humps and asymototes

hump<-function(x,y){
  library(tree)
  model<-tree(y~x)
  xs<-grep("[0-9]",model[[1]][[5]])
  xv<-as.numeric(substring(model[[1]][[5]][xs],2,10))
  xv<-xv[1:(length(xv)/2)]
  xv<-c(min(x),sort(xv),max(x))
  yv<-model[[1]][[4]][model[[1]][[1]]=="<leaf>"]
  plot(x,y,col="red",pch=16,
       xlab=deparse(substitute(x)),
       ylab=deparse(substitute(y)))
  i<-1
  j<-2
  k<-1
  b<-2*length(yv)+1
  for (a in 1:b){
    lines(c(xv[i],xv[j]),c(yv[k],yv[i]))
    if (a %% 2==0){
      j<-j+1
      k<-k+1
    }
    else{
      i<-i+1
    }
  }
}

library(lattice)
attach(ethanol)
names(ethanol)

x=E[E<1.007]
y=NOx[E<1.007]

par(mfrow=(c(1,2)))
hump(ethanol$E[ethanol$E<1.007],NOx[ethanol$E<1.007])
hump(ethanol$E[ethanol$E<1.006],NOx[ethanol$E<1.006])

dev.off()
hump(x,y)
NOx
NOx[E<1.007]

# remove any existing variable
rm(list=ls())
detach()











###########################################
Chapter 24: Time series analysis 
###########################################

#### key concepts: trend, serial dependence, stationarity


# one example blowflies
blow<-read.table("blowfly.txt",header=T)
names(blow)
attach(blow)
flies<-ts(flies)
plot(flies,type="l",col=4)
points(flies,col=3)
length(flies)


####  plot for lags 1 to 4 in a function
par(mfrow=c(2,2))
sapply(1:4,function(x) plot(flies[-c(361:(361-x+1))],flies[-c(1:x)]))

head(flies[-c(1:2)])
head(flies)

tail(flies[-c(361:(361-2+1))])
tail(flies)
dev.off()

# check if there is a linear trend
# x values for the linear fit
summary(lin_fit)


# detrend the data

detrend<-flies-predict(lin_fit)

par(mfrow=c(2,2))
ts.plot(detrend)
acf(detrend)
pacf(detrend)
acf(detrend,type="p")
dev.off()

# seasonal data
rm(list=ls())
weather<-read.table("SilwoodWeather.txt",header=T)
attach(weather)
names(weather)
head(weather)
index<-1:length(upper)
plot(index/365,upper)


# model for a seasonal cycle

model<-lm(upper~sin(index/365*2*pi)+cos(index/365*2*pi))
plot(index/365,upper,col=2,lwd=2,pch=".")  #pch: points characater
# bg: background fill color
#cex: character expansion
lines(index/365,predict(model),col=3)
#points(index/365,predict(model),col=3)

summary(model)

plot(model$resid,pch=".")
abline(h=0,lty="dotted")


## look for serial correlation in residuals

# windows(7,4)

par(mfrow=c(1,2))
acf(model$resid,main="")
acf(model$resid,type="p",main="")


## check monthly means pattern

head(weather)

temp<-ts(as.vector(tapply(upper,list(month,yr),mean)))

# as.vector to sum all the monthly means to a vector
# tapply 
temp
str(temp)

acf(temp,main="")
pacf(temp,main="")


####### Decompositions of the time series

# convert the normal data into time series
head(weather)
high<-ts(upper,start=c(1987,1),frequency=365) # start: the time of the first obs
plot(high)
head(high)

# use stl: to decompose a ts into seasonal trend and irregular components using loess
# loess: local polynomial regression fit

up<-stl(high,"periodic")

plot(up)
names(up)


#### test for a trend in the time series

ys<-factor(1+(yr>2002))
ys
length(ys)
tapply(upper,ys,mean)
# the mean temp is higher for the last 9 years than before

# Case 1
# if there are too many pseudo relication (temporal), linear fit will not work
#try MIXED MODEL here, use lmer

library(lme4)
model2<-lmer(upper~index+sin(index/365*2*pi)+cos(index/365*2*pi)+(1|factor(yr)),REML=F)
model3<-lmer(upper~sin(index/365*2*pi)+cos(index/365*2*pi)+(1|factor(yr)),REML=F)
anova(model2,model3)

# can tell that no significant effect of trend from above



# Case 2
# Ignore the pseudo replication by average over years
# fit the linear regression

means<-as.vector(tapply(upper,yr,mean))
means

model<-lm(means~I(1:length(means)))
summary(model)
# shows a significant trend





#### spectral analysis
# fundamental tool: periodogram

number<-read.table("lynx.txt",header=T)
attach(number)
names(number)
head(number)
plot.ts(Lynx)
ts.plot(Lynx)

spectrum(Lynx,main="",col="blue")

str(number)

number<-ts(number)
ts.plot(number)

par(mfrow=c(1,2))
spectrum(Lynx,main="",col="blue")
ts.plot(number)

#### multiple time series
twoseries<-read.table("twoseries.txt",header=T)
attach(twoseries)
names(twoseries)
head(twoseries)
plot.ts(cbind(x,y),main="")

# analyze acf seperately and check cross correlation

par(mfrow=c(2,2))
acf(cbind(x,y),type="p",col="red")

# 1.the evidence of periodicity is stronger in x than in y: 
# the pacf is signif and negative at lag 2 for x while not for y

# 2. the cross correlation between x and y is signif at lag 1 and 2: positive 
# change in x is related with negative change in y





#### simulate time series for the purpose of demonstration

# iid case
y<-rnorm(250,0,1)
par(mfrow=c(1,2))
plot.ts(y)
acf(y,main="")

#  AR MODEL
genAR<-function(n,parm,mu=0,B=500){
  ## n is the sample size
  ## parm is a vector of p AR parameters phi1,...,phip (order matters)
  ## mu is the mean of process, default is 0
  ## B is burn-in length, B=500 is default, B>p must hold
  p<-length(parm)
  #### set your distribution here
  ####
  M<-B+n
  e<-rnorm(M)
  # e<-rchisq(M,1)-1
  # e<- rbinom(M,1,.5)-.5
  y<-e
  for(i in (p+1):M){
    s<-0
    for(j in 1:p){
      s<-s+ parm[j]*y[i-j]
    }
    y[i]<-e[i]+s
  }
  y<-y[(B+1):M]+mu
  return(y)
}


genAR(300,0.5)


## Generate MA model
genMA<-function(n,parm,mu=0){
  ## n is the sample size
  ## parm is a vector of q MA parameters theta1,...,thetaq (order matters)
  ## mu is the mean of process, default is 0
  q<-length(parm)
  #### set your distribution here
  M<-q+n
  #e<-rnorm(M)
  # e<-rchisq(M,1)-1
  e<- rbinom(M,1,.5)-.5
  y<-e
  for(i in (q+1):M){
    s<-0
    for(j in 1:q){
      s<-s+ parm[j]*e[i-j]
    }
    y[i]<-y[i]+s
  }
  y<-y[(q+1):M]+mu
  return(y)
}




##########Time series models
Lynx
acf(Lynx,main="")
pacf(Lynx)

arma<-arima(Lynx,order=c(2,0,0))
AIC(arma)
# Ar(4) model best from the pacf (negative , cut off at lag 4)













##############################################################
#### Multivariate Statistics
##############################################################

##### Principal components analysis
# : a procedure that uses orthogonal transformation to convert a set of obs 
# of possibly correlated variable to a set of values of linearly uncorrelated

# advantage: easy to calculate
# cons: interpreting what the components means in scientific term is hard

pgdata<-read.table("pgfull.txt",header=T)
head(pgdata)
summary(pgdata)
dim(pgdata)

pgd<-pgdata[,1:54]
model<-prcomp(pgd,scale=T) # procomp performs a PCA on the given data matrix
# and returns as an object of class procomp
# alternative is : princomp
# scale indicates whether the variables should be scaled to have unit variance before 
#the analysis

model
summary(model)
plot(model,main="",col=3) #scree plot:
##A scree plot displays the eigenvalues associated with a component or factor 
#in descending order versus the number of the component or factor. 
#You can use scree plots in principal components analysis and factor analysis 
#to visually assess which components or factors explain most of the variability
#in the data.


# Or use this statement
screeplot(model)

# draw biplot 
biplot(model)
head(model)
names(model)
model$x  # y=x*variables, the one with largest variance is pc1
dim(predict(model))
predict(model)[,1]


################ Factor analysis
# is a statistical method used to describe variability among observed,
# correlated variables in terms of a potentially lower number of 
# unobserved variables called factors.

# provide useable numerical values for quantities taht are not directly measurable
pgd
help(factanal) #Perform maximum-likelihood factor analysis
#on a covariance matrix or data matrix.
factanal(pgd,factors=3)
model<-factanal(pgd,factors=8)   #length of the loadings equal the number of variables;
# not number of rows

loadings(model)
model$loadings

## NOte: factanal : be conventionally described as explanatory factor analysis
#   sem: package, confirmatory factor analysis (structual equation models)






############ Cluster analysis

# kmeans, hclust,table...

# group (cluster) data so that objects in the same group are moer similar 
# than the objects in the other groups

### Three ways carrying out: 
# Way 1: partitioning into a number of clusters specified by the user 
# with functions such as "kmeans"
# Way 2: hierarchical, staring with each individual as a separate entity and ending up
# with a single aggregation using functions like "hclust"
# Way 3: divise, starting with a single aggregate of all the individuals and splitting 
# up clusters until all the individuals are in different groups

kmd<-read.table("kmeansdata.txt",header=T)
kmd # contains x, y, group
attach(kmd)
names(kmd)

plot(x,y,col=group,pch=16)
model.1<-kmeans(data.frame(x,y),6)
plot(x,y,col=model.1[[1]])

model.2<-kmeans(data.frame(x,y),4)
plot(x,y,col=model.2[[1]])


help(kmeans) # perform k means clustering on a data matrix

# see the rate of misclassification by tabulating the real groups against
# the groups determined by kmeans

model.1

table(model.1[[1]],group) # first column shows the group by kmeans
group
model.1[[1]]

?table
table(group)

# another example
# taxonomic use of kmeans

taxa<-read.table("taxon.txt",header=T)
taxa
names(taxa)
attach(taxa)

# start by looking at the scatterplot matrix as a whole

pairs(taxa) # get a view of data seperation
pairs(taxa[1:4])

kmeans(taxa,5) # the computer was doing its classification blind
# if using classification decision tree,it can discover 
# a faultless key


########## Hierarchical cluster analysis

pgdata<-read.table("pgfull.txt",header=T)
attach(pgdata)
names(pgdata)
labels<-paste(plot,letters[lime],sep="") # letters: the built in constant a,b,c...


dist(pgdata[,1:54]) # calculate the distance between each row (objects)
hpg<-hclust(dist(pgdata[,1:54])) # use the dist to carry out h-cluster
plot(hpg,lables=labels,main="") # dendrogram, the y axis shows the distance at which
# the clusters merge
plot(hpg)



######## discriminant analysis (pan bie fen xi)

## identity of each individual is known, want to know how the explanatory variables
## contribute to the correct classification of individuals

library(MASS)
tax<-read.table("taxonomy.txt",header=T)
tax
attach(tax)

names(tax)
model<-lda(Taxon~.,tax)
?lda
?qda
dim(tax)
plot(model,col=rep(1:4,each=30)) # four kinds of taxon arranged by order
model

# From the model data, we will bsae the key on sepal first, then leaf,petiole
# by checking how much their LD is

predict(model)$class   #one member of type I was misallocated to type III


#### do training

train<-sort(sample(1:120,60))
train

table(Taxon[train])



################################
## Neural networks  # Ripley book in more details
###############################
#key feature: contain a hidden layer: each node in hte hidden layer receives info
# from each of many inputs, sums the inputs,adds a constant(the bias) then transforms
# the result using a fixed function

# can operate like multiple regression when ouputs are coninuous
# can operate like classifications when outputs are categorical



####################################################
# Chapter 26 Spatial statistics
####################################################

# Can be used to solve three kinds of problems
# 1. point processes (locations and spatial patterns of individuals)
# 2. maps of a continuous response variable (kriging)
# 3. spatially explicit responses affected by the identity, size and proximity of neighbors

# 26.1 Point process 
### Three broad classes of spatial pattern on a continuum from complete regularity
# 1. random pattern: the distribution of each individual in completely independent of the distribution 
#    of every other
# 2. regular pattern:individuals are more spaced out than in a randome one, presumably because of some 
#    mechanism that eliminates individuals taht are too close together
# 3. aggreated pattern: individuals are more clumped than in a random one,presumably because of 
#    some process such as reproduction with limited dispersal , or because of underlying spatial 
#    heterogeneity(eg. good patches)

# Example: random points in a circle

#random length and randome angle
point<-function(r){
  angle<-runif(1)*2*pi
  length<-runif(1)*r
  x<-length*sin(angle)
  y<-length*cos(angle)
  return(data.frame(x,y))
  
}

e0<-10
n0<-10
plot(e0,n0,ylab="",xlab="",ylim=c(0,2*n0),xlim=c(0,2*e0),type='n')  # type n means no plotting
n<-1000
r<-10
for(i in 1:n){
  a<-point(r)
  e<-e0+a[1]
  n<-n0+a[2]
  points(e,n,pch='.',col='blue') # pch ranges 0 to 25
}
# More points are around the center

## cookie cutter problem
n<-10000
side<-10
# install.packages('maptools')
# install.packages('sp')
library(sp)
library(maptools)

space<-cbind((runif(n)*side),(runif(n)*side))
plot(space)
circle<-function(e,n,r){
  angle<-seq(0,2*pi,2*pi/360) # 360 angles
  x<-r*sin(angle)
  y<-r*cos(angle)
  return(cbind((x+e),(y+n)))
}

xc<-9
yc<-9
rc<-1
# generate a circle with center at (9,9) with radius 1
outline<-circle(xc,yc,rc)
wanted<-point.in.polygon(space[,1],space[,2],outline[,1],outline[,2])
# return 1 if the point is in the polygon, 0 if not
points(space[,1][wanted==1],space[,2][wanted==1],col='blue',pch=16)
# There is no clustering in this circle


### 26.2 Nearest neighbors
x<-runif(100)
y<-runif(100)

# par cane be used to set or query graphical parameters 

par(pty='s') #pty,a character string specifying the type of plot region to be used, 
# s generates square region, m generates maximal ploting region
plot(x,y,pch=21,bg='red')
distance<-function(x1,y1,x2,y2) sqrt((x2-x1)^2+(y2-y1)^2)
r<-numeric(100)
nn<-numeric(100)
d<-numeric(100)
for (i in 1:100){
  for(k in 1:100) d[k]<-distance(x[i],y[i],x[k],y[k])
  # r stores the minimal distance from the i the point
  r[i]<-min(d[-i])
  # nn stores the subscript of point which has the minimal distance from the i-th point
  nn[i]<-which(d==min(d[-i]))
}
#connect to the nearest point for each point
for (i in 1:100) lines(c(x[i],x[nn[i]]),c(y[i],y[nn[i]]),col='green')


# Next work out how many points are closer to the edges than their nearest neighbors 
topd<-1-y
# distance to the top edge
rightd<-1-x
#distance to the right edge
?pmin
# pmin takes vectors as arguments
edge<-pmin(x,y,topd,rightd)
edge
sum(edge<r)
edge<r

# show the points at the plot
id<-which(edge<r)
points(x[id],y[id],col='green',cex=1.5,lwd=2)
# cex: the amount by which plotting text and symbols shoudl be magnified
# lwd: line width, default is 1
# bg: color to be used for background of the device region


# split the two dimensional surface into a mosaic by halving the distance between neighbor pairs of points
install.packages('tripack')
library(tripack)
plot(x,y,pch=16,add=TRUE)
map<-voronoi.mosaic(x,y)
plot.voronoi(map,pch=16,add=TRUE)
#add: if TURE, add to a current plot


## 26.3 Tests for spatial randomness 

# 1. Clark and Evans (first order estimates), assume that the population density of the individuals,p(rou)
#     is known, then the expected mean distance to the nearest neighbor is E(r)=sqrt(p)/2
#     An index of randomness: mean(r)/E(r), is 1 for ranom patterns, more than 1 for regular patterns, 
#      and less than 1 for aggregated patterns 

# Disadvantage: can give no info for the way that the spatial distribution changes within the area 

# 2.  K function (second order estimates ), describes the way the spatial interactions change through space
#     K(d)=1/lambda*(E(number of points <=distance d of an arbitray point))
#     Interpretation: If there is no spatial dependence, the expected number of points that are within
#                     a distance of d is Pi*d^2*mean density. 
#                     lambda is the man number of points per unit area.

# 3. Ripley's K  
#    K(d)=1/n^2*|A|*sum_i*sum_j(I_d(d_ij)/w_ij)
#    n is the number of points in region A, |A| is the area of region A, w_ij is a factor related to 
#    edge effects. 
# Plot K(d) vs. d and compare with Pi*d^2 vs. d 

distances<-numeric(100)
for (i in 1:100) distances[i]<-distance(x[1],y[1],x[i],y[i])
distances

dd<-numeric(10000)
dd<-matrix(dd,nrow=100)
for (j in 1:100){
  for (i in 1:100) dd[j,i]<-distance(x[j],y[j],x[i],y[i])
}
# Or use the sapply function to realize the above matrix 
dd1<-sapply(1:100,function (i,j=1:100) distance(x[j],y[j],x[i],y[i]))
??saaply
dd==dd1

count<-numeric(100)
d<-seq(0.01,1,0.01)
for (i in 1:100) count[i]<-sum(dd<d[i])-100
count
K<-count/10000
K
plot(d,K,type='l',col='red')
lines(d,pi*d^2,col='blue')
# The two lines are inconsistent for longer distances for a random pattern, that's because the algorithm
# is counting for too few neighbors for the points near edges, so edge correct for Ripley's K is needed. 
# And it can be realized by the "Spatial" package. 

library(spatial)
pines<-ppinit('pines.dat')
pines
??ppinit #Read a Point Process Object from a File
# A point process is a type of random process for which any one realisation consists of a set of 
# isolated points either in time or geographical space, or in even more general spaces. 
library(spatial)
?par
par(mfrow=c(1,2),pty='s')
plot(pines,pch=16,col='blue')
plot(Kfn(pines,5),type='s',xlab='distance',ylab='L(t)')
lims<-Kenvl(5,100,Psim(71))
lines(lims$x,lims$lower,lty=2,col='red')
lines(lims$x,lims$upper,lty=2,col='red')
#  It shows a regular pattern for small distances as the line lies below the lower envelope of the CSR line
?Kfn 
# Kfn = sqrt(K(d)/Pi), so Kfn vs d should be a linear line with slope one if it's random
??Kenvl # Computes envelope (upper and lower limits) and average of simulations of K-fns
??Psim #Simulate Binomial spatial point process.
# Psim(n), n is number of points





# 4. Quadrat-based methods 
# Count number of individuals in quadrats of different sizes 


# Example of an random pattern
par(mfrow=c(1,1))
plot(x,y,pch=16,col='red')
grid(10,10,lty=1)
?grid #grid adds an nx by ny rectangular grid to an existing plot.

?cut
#cut divides the range of x into intervals and codes the values in x 
# according to which interval they fall. 

# Next count the number of individuals in each cell 
xt<-cut(x,seq(0,1,0.1))
yt<-cut(y,seq(0,1,0.1))
xt
yt

count<-as.vector(table(xt,yt))
count
table(count)
# shows about 40 cells empty, 1 cell has 5 count, consistent with Poison with lambda=1:
expected<-100*exp(-1)/sapply(0:5,factorial)
expected

# Example of aggregated pattern 
rm(list=ls())
trees<-read.table('trees.txt',header=T)
head(trees)
names(trees)
attach(trees)
plot(trees$x,trees$y,pch=16,col='blue')
abline(v=seq(0,1,.1),col='lightgray',lty=1)
abline(h=seq(0,1,.1),col='gray',lty=2)
xt<-cut(x,seq(0,1,0.1),right=FALSE)
yt<-cut(y,seq(0,1,0.1),right=FALSE)
xt
yt
?cut
# right	
# logical, indicating if the intervals should be closed on the right 
# (and open on the left) or vice versa.
count<-as.vector(table(xt,yt))
count
table(count)
mean(count)
# a preliminary estimate of the departure from randomness is the variance mean ratio
var(count)/mean(count)
# the aggregation parameter K: 
k_para<-mean(count)^2/(var(count)-mean(count))
expected<-dnbinom(0:8,size=k_para,mu=mean(count))*100
expected

ht<-numeric(18)
observed<-table(count)
ht[seq(1,17,2)]<-observed
ht[seq(2,18,2)]<-expected
names<-rep('',18)
names[seq(1,18,2)]<-as.character(0:10)
# Show the expected and observed counts on the barplot
barplot(ht,col=c('darkgray','lightgray'),names=names,
        ylab='frequency',xlab='trees per quadrat')
legend(locator(1),legend=c('observed','expected'),
       fill=c('darkgray','lightgray'))
# locator{graphics} Reads the position of the graphics cursor when the (first) mouse button is pressed.



## 5.  counting data on maps 
# Convention: A point falls exactly on the x axis or y axis is counted as being INSIDE on the left and 
#             bottom edge, as OUTSIDE on the right and top edges as in the following plot: 

plot(c(0,2),c(0,2),type='n',xlab='',ylab = '')
# type=n avoids producing the two points and results in a blank canvas with the specified limits
lines(c(0,1,1,0,0),c(0,0,1,1,0))
points(c(0.5,0),c(0,0.5),pch=16,col='green')
points(c(0.5,1),c(1,0.5),pch=16,col='red')
legend('topright',legend=c('inside','outside'),col=c('green','red'),pch=16)

#type=
#"p" for points,
#"l" for lines,
# "b" for both,
# "c" for the lines part alone of "b",
# "o" for both ‘overplotted’,
# "h" for ‘histogram’ like (or ‘high-density’) vertical lines,
# "s" for stair steps,
# "S" for other steps, see ‘Details’ below,
# "n" for no plotting


## 26.4 Brief introduction for packages for spatial statistics

# Package 1: spatstat: statistical analysis for spatial point patterns
# Package 2: spdep: for spatial analysis of data from mapped regions
install.packages("spatstat")
install.packages("spdep")
library(spatstat)
beginner
demo(spatstat)
library(help=spatstat)

library(spdep)

## The spatstat package 
?ppp #Creates an object of class "ppp" representing a point pattern dataset in the two-dimensional plane.

?quadratcount #Divides window into quadrats and counts the numbers of points in each quadrat.

??density.pp #Compute a kernel smoothed intensity function from a point pattern.

?Kest # Ripley's K 
?Kest.fft
?distmap
?ppm #Fits a point process model to an observed point pattern.


data<-read.table('ragwortmap2006.txt',header=T)
names(data)
head(data)
dim(data)
summary(data$xcoord)
attach(data)

# Use the function ppp to convert the coordinate data into an object of class ppp 
#  representing a point pattern data 

# The third and fourth arguments are boundary coordinates for x and y respectively
# MARKS (optional) mark values for the points. A vector or data frame.

ragwort<-ppp(xcoord,ycoord,c(0,3000),c(0,1500),marks=type)
head(ragwort)
summary(ragwort)
plot(split(ragwort),main='')
# 
?split
?ppp

# quadratcount divides window into quadrats and count the number of points in each quadrat 
plot(quadratcount(ragwort),main='') # default setting 
plot(quadratcount(ragwort,
                  xbreaks=c(0,500,1000,1500,2000,2500,3000),
                  ybreaks=c(0,500,1000,1500)),main='')
# quadratcount(X, nx=5, ny=nx, ...,xbreaks=NULL, ybreaks=NULL, tess=NULL)
# nx, ny are number of rectangular quadrats in x and y respectively 
# xbreaks ,ybreaks are coordinates for x and y boundries

# plot density plot 
z<-density.ppp(ragwort)
plot(z,main='')

# 
K<-Kest(ragwort)
plot(K,main='K function')
# Red dotted line is the theoretical line of expected number of plants within distance r vs r with 
#  the assumption of complete spatial randomness,
# Black line is the observed one
# The observed line is always above the red one indicates that strong spatial aggregation at all scales


# pair correlation of the data
?pcf
pc<-pcf(ragwort)
plot(pc,main='Pair correlation')
# There is a strong pair correlation for small scales and much less for large scales 

#distance map for individual plants
z<-distmap(ragwort)
plot(z,main='')

# fit a point process model
?polynom
model<-ppm(ragwort,~marks+polynom(x,y,2),Poisson())
#ppm.ppp(Q = ragwort, trend = ~marks + polynom(x, y, 2), interaction = Poisson())

plot(model)

summary(model)



## spdep package 
??knearneigh
??knn2nb

library(spdep)
# still use the ragwort data
myco<-cbind(xcoord,ycoord)
head(myco)
myco<-matrix(myco,ncol=2)
head(myco)
myco.knn<-knearneigh(myco,k=4) # knn object
str(myco.knn) 
names(myco.knn)
head(myco.knn)
?knearneigh

myco.nb<-knn2nb(myco.knn) # neighbor object
myco.nb
?knn2nb 
# converts a knn object returned by knearneigh into a neighbours list of class nb 
# with a list of integer vectors containing neighbour region number ids.
plot(myco.nb,myco)
# plot the data with connection to the four nearest neighbors

myco.lw<-nb2listw(myco.nb,style='W')
myco.lw


## Three classic tests based on spatial cross product 
# 1. Moran C(i,j)=z(i)*z(j)
# 2. Geary C(i,j)=(z(i)-z(j))^2
# 3. Sokal C(i,j)=|z(i)-z(j)|
# With z(i)=(x(i)-mean(x))/sd(x)

# Moran
?moran
moran(1:3359,myco.lw,length(myco.nb),Szero(myco.lw))

# Geary
?geary
geary(1:3359,myco.lw,length(myco.nb),length(myco.nb)-1,Szero(myco.lw))

# Mantel's permutation test
?sp.mantel.mc
permuatation<-sp.mantel.mc(1:3359,myco.lw,nsim=99,type='sokal') #nsim: number of permutations
plot(permuatation)

###Polygon lists 
data("columbus")
polys
# polys is an object of class polylist which comprises a list of 49 polygons

# First takes a list of polygons and works out which regions are neighbors by looking for shared boundaries
?poly2nb
class(polys)
xx <- poly2nb(columbus)
example(columbus)
coords <- coordinates(columbus)
xx <- poly2nb(columbus)
dxx <- diffnb(xx, col.gal.nb)
plot(columbus, border="grey")
plot(col.gal.nb, coords, add=TRUE)
plot(dxx, coords, add=TRUE, col="red")
title(main=paste("Differences (red) in Columbus GAL weights (black)",
                 "and polygon generated queen weights", sep="\n"))



## 26.5 Geostatistical data 
# Fundamental statistical tool of spatial statistics: variogram(or semivariogram). 
# It measures how quickly spatial autocorrelation falls off with increasing distance 
# Two functions with the same name : variogram in spatial package, Variogram in nlme package 

# variogram in spatial package : need to create a trend surface or kriging object with columns x, y, z:
# x,y are spatial coordinates and the third column is the response data 

library(spatial)
data<-read.table('ragwortmap2006.txt',header=T)
head(data)
attach(data)

dts<-data.frame(x=xcoord,y=ycoord,z=diameter)
head(dts)
dts<-na.omit(dts)
head(dts)
# create a trend surface using function surf.ls
?surf.ls #Fits a trend surface by least-squares.
#surf.ls(np, x, y, z)
# np: degree of polynomial surface
surface<-surf.ls(2,dts)
?variogram
variogram(surface,300) # 300 is number of bins used
# It computes the average squared difference for pairs with separation in each bin, returning results
# for bins that contain six or more pairs.
?correlogram
#Compute spatial correlograms of spatial data or residuals.
correlogram(surface,300)
# Positive correlations disappeared by about 100cm and correlations are spurious edge effects at xp=3000
detach(data)

## Variogram in nlme
library(nlme)
head(dts)
attach(dts)
model<-gls(z~x+y,data=dts)
summary(model)
?Variogram
plot(Variogram(model,form=~x+y))

## 26.6 Regression models with spatially correlated errors: generalized least squares 
# Variogram is characterized by :
# Nugget: small scale variation plus measurement error 
# Range: the threshold distance beyond which the data are no longer autocorrelated
# Sill: the asymptotic value of r(h) as h goes to infinity, representing the variance of teh randome filed


library(nlme)
spatialdata<-read.table('spatialdata.txt',header=T)
names(spatialdata)
attach(spatialdata)
# Compare the yield of 56 varieties of wheat. 

# visual inspection of the effect of location 
par(mfrow=c(1,2))
plot(latitude,yield,pch=21,col='blue',,bg='red')
plot(longitude,yield,pch=21,col='blue',bg='red')

#bar plot of the mean yields for each variety 
par(mfrow=c(1,1))
barplot(sort(tapply(yield,variety,mean)),col='green')

# check the block effect 
tapply(yield,Block,mean)
barplot(tapply(yield,Block,mean),col='red')
# substantial block effects 

# One way analysis of variance 
model1<-aov(yield~variety)
summary(model1)
# No significant difference among the yields of different varieties


#A split plot analysis 
Block<-factor(Block)
model2<-aov(yield~Block+variety+Error(Block))
# If the formula contains a single Error term, this is used to specify error strata, 
# and appropriate models are fitted within each error stratum.
summary(model2)
# still not significant 

# Fit latitude and longitude as covariates 
model3<-aov(yield~Block+variety+latitude+longitude)
summary(model3)
coef(model3)
# variety effect is close to significant 

#use GLS model to introduce spatial covariance between yields from locations taht are close together 
?groupedData
space<-groupedData(yield~variety|Block)
head(space)
class(space)
space
model4<-gls(yield~variety-1,space) # remove the intercept by '-1' (Page 398), so the variety means are given
summary(model4)

#include the spatial covariance 
plot(Variogram(model4,form=~latitude+longitude))
# There appears to be a nugget of about 0.2. 
# Nugget: small scale variation plus measurement error 
# Range: the threshold distance beyond which the data are no longer autocorrelated
# Sill: the asymptotic value of r(h) as h goes to infinity, representing the variance of teh randome filed

## Try a spherical correlation structure using corSpher class 
# Other options include corExp, corGaus, corLin, corRatio(rational quadratic spatial correlation),
# corSpher(spherical spatial correlation), corSymm(generic correlation matrix, w/o additional structure)
?update
model5<-update(model4,corr=corSpher(c(28,0.2),form=~latitude+longitude,nugget=T))
#28 is about the distance at which the semivariogram first reaches 1,
# 0.2 is approximate nugget 
summary(model5)
model5
coef(model5)

# Try other spatial correlation 
model6<-update(model4,corr=corRatio(c(12.5,0.2),form=~latitude+longitude,nugget=T))
?corRatio
# corRatio(value,...). If nugget is TRUE, value meaning that a nugget effect is present, 
# value can contain one or two elements, the first being the "range" 
# and the second the "nugget effect". 

# 12.5 is the range distance between the semivariogram is (1+nugget)/2=0.6 .

summary(model6)

anova(model5,model6)
anova(model4,model6) # model4: spatially independent errors; model6: spatial model 

# Check the adequacy of model6
plot(Variogram(model6,resType='n'))
?Variogram
# No pattern in the plot of the sample variogram, so conclude that the rational quadratic correltaion is adequate.

plot(model6,resid(.,type='n')~fitted(.),abline=0)
?plot

qqnorm(model6,~resid(.,type='n'))
?qqnorm


# Next, investigate the significance of any differences between the varieties 
model7<-update(model6,model=yield~variety) # change the model structure 
anova(model7)

# Specific contrasts can be obtained through L argument in anova
anova(model6,L=c(1,0,-1)) # model 6 is used 



## 26.7 create dot-distribution map from a relational database 
library(RODBC)
# To be continued 

#################################################
# Chapter 27 Survival analysis
#################################################



# 27.1 A Monte Carlo experiment for demo

death1<-numeric(30) # generate 30 zeros
death1

# generate a set of time to death  of sample size 30 
for (i in 1:30){
  rnos<-runif(100) # generate 100 randome numbers to indicate 100 weeks
  death1[i]<-which(rnos<=0.1)[1] 
  #find the week where the value first smaller than .1 (dies)
}
death1


# generate another set 
death2<-numeric(30)
for (i in 1:30){
  rnos<-runif(100)
  death2[i]<-which(rnos<=.2)[1]
}
death2

mean(death1)
mean(death2)


## Do analysis for the above pseudo data
death<-c(death1,death2)
factory<-factor(c(rep(1,30),rep(2,30)))
plot(factory,death,xlab="Factory",ylab="Age at failure",col="wheat")

# fit a model

model1<-glm(death~factory,family=Gamma)
summary(model1)

# Can conclude that the factories (types) are marginally significantly different 
# in mean age at failure

rm(death)


###### Some background knowledge of survial analysis 

# Three interchangeable concepts:
# 1. surviorship 2. age at death 3. instantaneous risk of death

# Three patterns of surviorship
# 1. Type I most of the mortality occurs late in life
# 2. Type II mortality occurs at a roughly constant rate
# 3. Type III most of the mortality occurs early in life 

# Hazard function: instantaneous risk of death: f(t)/(1-F(t))
# Survial function: S(t)=1-F(t)
# Density function: f(t)

## Kaplan Meier survial distribution
# K-M estimate S(t)_hat=Product((r_i-d_i)/r_i) for i <=t;
# r_i is the number at risk(surviors)
# d_i is the number of deaths at time i

# A response varible in survial analysis are consist of the survial part and the staus part(censored?die?...)


# Three cases we are concerned:
# 1. constant hazard with no censoring: use generalized linear model with gamma errors
# 2. constant hazard with censoring: exponential survival model with censoring indicator 1
# 3. age-specific hazard,w/o censoring: choose either parametric models based on Weibull distrbution
#    or non-parametric techniques based on Cox proportional hazards

# Parametric models: such as gamma, exponential, log-normal, extreme value,Weibull...
#                    for model prediction

# Cox proportional hazards model: the hazard lambda(t|X)=lambda_0(t)*exp(beta*X)
#               the estimate of beta will not depend on lambda_0,only depends on the ranks 
#               of the survival times
#               no extrapolation


## Parametric analysis w/o censoring: use survfit,survreg

seed<-read.table("seedlings.txt",header=T)
head(seed)
summary(seed)
library(survival)
attach(seed)
names(seed)

# create an indicator to show which data are censored
status<-1*(death>0)
plot(survfit(Surv(death,status)~1),ylab="Survivorship",xlab="Weeks",col=4)
plot(survfit(Surv(death,status)~1+cohort),ylab="Survivorship",xlab="Weeks",col=c(4,3))

# fit the model by different cohort types

model<-survfit((Surv(death,status)~cohort))
summary(model)
model<-survfit((Surv(death,status)~1+cohort)) # no difference
model
plot(model,col=c("red","blue"),ylab="Survivorship",xlab="Week")

model1<-survfit((Surv(death,status)~cohort+gapsize)) # no difference
model1



## Cox's porportional hazards: use coxph
model1<-coxph(Surv(death,status)~strata(cohort)*gapsize)
model1
summary(model1)

# fit model w/o interaction
model2<-coxph(Surv(death,status)~strata(cohort)+gapsize)
# strata :This is a special function used in the context of the Cox survival model.
# It identifies stratification variables when they appear on the right hand side of a formula.

# separate baseline hazard functions are fit for each strata


summary(model2)
model3<-coxph(Surv(death,status)~gapsize)
summary(model3)

anova(model1,model2)




##### Models with censoring 
# Set up an extra vector called status to show if it's censored or not

rm(status)
detach(seed)
cancer<-read.table("cancer.txt",header=T)
cancer
names(cancer)
attach(cancer)
library(survival)
plot(survfit(Surv(death,status)~treatment),col=c(1:4),ylab="Survivorship",xlab="Time")
tapply(death[status==1],treatment[status==1],mean)
?tapply(x,index,fucntion)
tapply(death[status==1],treatment[status==1],var)

# fit parametric model with constant hazard function,ie,exponential distribution

model1<-survreg(Surv(death,status)~treatment,dist='exponential')
model1
summary(model1)
# no significant treatment effects

# try nonconstant hazard function
model2<-survreg(Surv(death,status)~treatment,dist='weibull')
model2
summary(model2)

model3<-survreg(Surv(death,status)~treatment) # the default dist is Weibull
model3         # small scale parameter indicates that the hazard decreases with age
summary(model3)

anova(model1,model2)
# small p value reject null(no difference on the two models)

# Try some model simplification
treat2<-treatment
treat2
levels(treat2)
# merge the first two levels
levels(treat2)[1:2]<-'DrugsAB'
treat2
model4<-survreg(Surv(death,status)~treat2)
model4
summary(model4) #all significant
anova(model2,model4) # large p value accepts the hypothesis that two models no difference


#Now try merge drug C and placebo
levels(treat2)[2:3]<-"placeboC"
treat2
model5<-survreg(Surv(death,status)~treat2)
model5
summary(model5)
anova(model2,model5)
anova(model4,model5)

?predict
# predict the mean age at death taking account of the censoring
predict(model5,type='response')
predict(model5)

tapply(predict(model5,type='response'),treat2,mean)

tapply(death[status==1],treat2[status==1],mean)

# From the comparison, the greater censoring, the bigger difference

detach(cancer)
rm(death,status)


########## Compare coxph and survreg survial analysis
# Analyze the same data set and compare

insects<-read.table('roaches.txt',header=T)
insects
names(insects)
str(insects)
attach(insects)
plot(survfit(Surv(death,status)~group),col=c(1:3))

# Fit parametric model1
model1<-survreg(Surv(death,status)~weight*group,dist='exponential')
summary(model1)

# Fit parametric weibull model
model2<-survreg(Surv(death,status)~weight*group)
summary(model2)

anova(model1,model2)  # small p value shows that Weibull is better, significant difference between
# the two, note here weibull has higher error df

?step # Select a formula-based model by AIC.
names(model2)
model3<-step(model2) # -group means AIC without group

model3
summary(model3)  # even group has higher AIC than none, still chosen

# Given a set of candidate models for the data, the preferred model is 
# the one with the minimum AIC value. 
# AIC rewards goodness of fit (as assessed by the likelihood function), 
# but it also includes a penalty that is an increasing function of the 
# number of estimated parameters. 
# AIC=2k-log(theta|x)
# want to choose the minimum 

tapply(predict(model3),group,mean)
tapply(death,group,mean) # original data
tapply(death[status==1],group[status==1],mean)


## Try coxph, non parametric fit 
model0<-coxph(Surv(death,status)~weight*group)
model0
summary(model0) # see no significant interaction

# step selection
model00<-step(model0)
model00
names(model00)
summary(model00)

############## 
# Summary: One fundamental difference between Kaplan-Meier surviorship curves and Cox proportional 
#         hazards are, K-M refers to the population while Cox refers to an individual in a particular
#         group.
##############






###################################################
#### Chapter 28 Simulation Models
###################################################



#########################################
# Chapter 29 Change the Look of Graphics 
#########################################

# 29.1 Graphs for publication 
??las
??cex
par(mfrow=c(2,2))
x<-seq(0,150,10)
x
y<-16+x*0.4+rnorm(length(x),0.6)
y

plot(x,y,pch=16,col='blue',xlab='label for x axis',
     ylab='label for y axis')

plot(x,y,pch=16,col='blue',xlab='label for x axis',
     ylab='label for y axis',
     las=1,
     cex.lab=1.2,
     cex.axis=1.1)
plot(x,y,pch=16,col='blue',xlab='label for x axis',
     ylab='label for y axis',
     las=2,
     cex=1.5)
plot(x,y,pch=16,col='blue',xlab='label for x axis',
     ylab='label for y axis',
     las=3,
     cex=0.7,
     cex.lab=1.3,
     cex.axis=1.3)
par(mfrow=c(1,1))

# Functions used: 
# las: determines the orientation of the numbers on the tick marks (eg. in the above case, the 20,30,...
#      label numbers for x y axes)
#       las=1 (all vertical), las=2(numbers parallel with axes on both axes)
#       las=3(numbers both 90 degrees with axes),las=0(default,x axis parallel,y axis vertical)
# cex: determines the size of plotting characters (size of pch)
# cex.lab: determines the size of text labels 
# cex.axis: determines the size of the numbers on tick marks(eg.size of 20,30 ...)


## 29.2 Color 
# Four ways to specify a color in R 
# 1. by color number (1-7,8 is light grey)
# 2. by color name (the elements in colors())
# 3. by hexadecimal string of the form #rrggbb
# 4. by an integer subscript i, on the current palette()[i]

# by method 2
# All the colors in R : 657 types 
colors()

# by method 1 
plot(0:8,0:8,type='n',xlab='',ylab='color number')
axis(side=2,at=1:7)
axis(side=1,seq(1,7,1))
?axis
for(i in 1:8) lines(c(0,8),c(i,i),col=i) # from 9, the pattern repeats itself, i.e. col=9 ~ col=1

# by method 3
par(mfrow=c(2,3))
?rgb
rgb(0, 1, 0)
# rr, gg,bb represents hexadecimal digits of value in the range 00-FF for red green and blue respectively
plot(0:1,0:1,type='n',xlab='red',ylab='blue',main='green=0')
for(red in seq(0,1,0.1)){
  green<-0
  for (blue in seq(0,1,0.1)){
    points(red,blue,pch=16,col=rgb(red,green,blue))
  }
}

plot(0:1,0:1,type='n',xlab='red',ylab='blue',main='green=0.2')
for(red in seq(0,1,0.1)){
  green<-0.2
  for (blue in seq(0,1,0.1)){
    points(red,blue,pch=16,col=rgb(red,green,blue))
  }
}

plot(0:1,0:1,type='n',xlab='red',ylab='blue',main='green=0.4')
for(red in seq(0,1,0.1)){
  green<-0.4
  for (blue in seq(0,1,0.1)){
    points(red,blue,pch=16,col=rgb(red,green,blue))
  }
}

plot(0:1,0:1,type='n',xlab='red',ylab='blue',main='green=0.6')
for(red in seq(0,1,0.1)){
  green<-0.6
  for (blue in seq(0,1,0.1)){
    points(red,blue,pch=16,col=rgb(red,green,blue))
  }
}

plot(0:1,0:1,type='n',xlab='red',ylab='blue',main='green=0.8')
for(red in seq(0,1,0.1)){
  green<-0.8
  for (blue in seq(0,1,0.1)){
    points(red,blue,pch=16,col=rgb(red,green,blue))
  }
}

plot(0:1,0:1,type='n',xlab='red',ylab='blue',main='green=1.0')
for(red in seq(0,1,0.1)){
  green<-1.0
  for (blue in seq(0,1,0.1)){
    points(red,blue,pch=16,col=rgb(red,green,blue))
  }
}

# 29.2.1 pallette for groups of colors (by method 4)
?rgb
?hsv

par(mfrow=c(2,2))
?par
?rainbow
# mar:  A numerical vector of the form c(bottom, left, top, right) which gives the number of 
#     lines of margin to be specified on the four sides of the plot. The default is c(5, 4, 4, 2) + 0.1.

?pie
par(mar=c(1.5,1.5,1,1)) # margin parameter to optimize the size of the pie diagrams, thus keep labels distinct
pie(rep(1,7),col=rainbow(7),radius=1)
pie(rep(1,14),col=rainbow(14),radius=1)
pie(rep(1,28),col=rainbow(28),radius=1)
pie(rep(1,56),col=rainbow(56),radius=1)

# by method 4 : except for the above palette, there are four other built in functions 
pie(rep(1,14),col=heat.colors(14),
    radius=0.9,main='heat colors')
pie(rep(1,14),col=terrain.colors(14),
    radius=0.9,main='terrain colors')
pie(rep(1,14),col=topo.colors(14),
    radius=0.9,main='topo colors')
pie(rep(1,14),col=cm.colors(14),
    radius=0.9,main='cm colors')

?heat.colors #Create a vector of n contiguous colors.

par(mfrow=c(1,1))
# Create your own palette 
custom<-c(rgb(0.6,0.8,1),rgb(1,0.8,0.2),rgb(1,0.8,0.4),
          rgb(1,0.8,0.6),rgb(1,0.8,0.8),rgb(1,0.8,1),
          rgb(0.8,0.8,1),rgb(0.7,0.8,1))
pie(rep(1/8,8),col=custom)


# the RColorBrewer package 
library(RColorBrewer)
par(mfrow=c(3,3))
par(mar=c(1,1,1,1))
mypalette  <- brewer.pal(8,"Reds")
pie(rep(1,8), col = mypalette, radius = 0.9,main="Reds")
mypalette  <- brewer.pal(8,"Blues")
pie(rep(1,8), col = mypalette, radius = 0.9,main="Blues")
mypalette  <- brewer.pal(8,"Greens")
pie(rep(1,8), col = mypalette, radius = 0.9,main="Greens")
mypalette  <- brewer.pal(8,"BrBG")
pie(rep(1,8), col = mypalette, radius = 0.9,main="BrBG")
mypalette  <- brewer.pal(8,"PiYG")
pie(rep(1,8), col = mypalette, radius = 0.9,main="PiYG")
mypalette  <- brewer.pal(8,"Spectral")
pie(rep(1,8), col = mypalette, radius = 0.9,main="Spectral")
mypalette  <- brewer.pal(8,"Accent")
pie(rep(1,8), col = mypalette, radius = 0.9,main="Accent")
mypalette  <- brewer.pal(8,"Pastel1")
pie(rep(1,8), col = mypalette, radius = 0.9,main="Pastel1")
mypalette  <- brewer.pal(8,"Set2")
pie(rep(1,8), col = mypalette, radius = 0.9,main="Set2")
par(mfrow=c(1,1))

?brewer.pal
# Three types of pallettes:
# 1. sequential palette (suitable for ordered data that progress from low to high)
# 2. diverging .
# 3. qualitative palette (suitable for nominal or categorical data)

# now can refer colors by method 4 
mypalette
# Reset the margin to be the default so the values could show properly 
par(mar= c(5, 4, 4, 2) + 0.1)
plot(x,y,pch=16,col=mypalette[5],xlab='label for x axis',
     ylab='label for y axis',main='')

# reset the palette back to default use 
palette('default')


## 29.2.3 colored plotting symbols with contrasting margins 


##  29.2.4 color in legends 
# the background in the plotting symbols 21–25 is called bg, 
# but in figure legends bg controls the background colour of the whole legend box. 
# So when you are creating figure legends for plotting symbols pch=21 to 25 you
# need to remember to use the argument pt.bg in the legend function in place of
# the usual bg for the interior colour of the symbol.

## 29.2.5 background colors 
# par(col=...) defines the color of the whole plot 
data<-read.table('silwoodweather.txt',header=T)
attach(data)
par(bg='wheat2')
plot(factor(month),lower,col='green4')

par(bg='white')


# 29.2.6 foreground color 
# fg is used to change the color of things like axes and boxes around plots 
par(mar=c(4,4,1,1))
par(mfrow=c(2,2))
plot(1:10,1:10,xlab='x label',ylab='y label')
plot(1:10,1:10,xlab='x label',ylab='y label',fg='blue')
plot(1:10,1:10,xlab='x label',ylab='y label',fg='red')
plot(1:10,1:10,xlab='x label',ylab='y label',fg='green')


# 20.2.7 Different colors and font styles for different parts of the graphs
# col.axis : The color to be used for axis annotation. Defaults to "black".
# col.lab : The color to be used for x and y labels. Defaults to "black".
# col.main : The color to be used for plot main titles. Defaults to "black".
# The color to be used for plot sub-titles. Defaults to "black".

# font: An integer which specifies which font to use for text.
# font.axis:The font to be used for axis annotation.
# font.lab : The font to be used for x and y labels.
# font.main: The font to be used for plot main titles.
# font.sub: The font to be used for plot sub-titles.

# check ?par for more details 
dev.off()
par(mfrow=c(1,1))
plot(1:10,1:10,xlab='x label',ylab='y label',
     pch=16,col='orange',col.lab='green4',col.axis='blue',
     col.main='red',col.sub='navy',
     sub='Subtitle',font.axis=3,font.lab=2,font.main=4,
     font.sub=3,main=expression('Critical elasticity log'[~10]~~'mday'^-1))
?expression
# expression allows more complicated formatting of axis labels:
# [ ] produces subscript 
# ^ produces superscript
# ~ produces wider spacing, the more tildes, the wider


# 29.2.8 full control of colors in plots 
data<-read.table('silwoodweather.txt',header=T)
attach(data)
plot(factor(month),lower,ylab="minimum temperature",xlab="month",
     medlty="blank",medpch=21,medbg="red",medcol="yellow",
     boxcol="red",boxfill="green",outpch=21,outbg="yellow",
     outcol="red",staplecol="blue",whisklty=1,whiskcol='black')
detach(data)


# Cross hatching 
# Can control five aspects of shading: 
# density of lines by density=
# angle of the shading by angle=
# border of the region by border=
# color of the lines and line type by col= lty= 
?barplot
data<-read.table('box.txt',header=T)
attach(data)
names(data)

par(mfrow=c(2,2))
barplot(tapply(response,fact,mean),border=NA)
barplot(tapply(response,fact,mean),density=3:10,border='green',col='red')
barplot(tapply(response,fact,mean),density=3:10,angle=seq(30,60,length=8))
barplot(tapply(response,fact,mean),density=20,axis.lty=3,col='red')
?barplot
dev.off()

barplot(tapply(response,fact,mean))
barplot(tapply(response,fact,mean),density=3:10)
barplot(tapply(response,fact,mean),density=3:10,angle=seq(30,60,length=8))
barplot(tapply(response,fact,mean),density=20)

# 29.4 Gray scale 
# grey scale goes from 0 to 1 and goes from dark to light. 
par(mfrow=c(1,1))
barplot(tapply(response,fact,mean),col=grey(seq(0.8,0.2,length=8)))

# 29.5 Color convex hull and other polygons 
data<-read.table('pgr.txt',header=T)
head(data)
# FR~y, hay and pH are covariates 
attach(data)

# Draw polygons to represent the convex hull for the abundance of y 
# in space defined and two covariates
plot(hay,pH)
x<-hay[FR>5]
y<-pH[FR>5]
polygon(x[chull(x,y)],y[chull(x,y)],col='red')
x<-hay[FR>10]
y<-pH[FR>10]
polygon(x[chull(x,y)],y[chull(x,y)],col='green')
x<-hay[FR>20]
y<-pH[FR>20]
polygon(x[chull(x,y)],y[chull(x,y)],density=10,angle=90,col='blue')
polygon(x[chull(x,y)],y[chull(x,y)],density=10,angle=0,col='blue')
points(hay,pH,pch=16)
?chull #Compute Convex Hull of a Set of Points , i.e. Computes the subset of points 
# which lie on the convex hull of the set of points specified.

# chull(x,y) returns an integer vector giving the indices of the unique points 
# lying on the convex hull, in clockwise order. 
?polygon

# 29.6 Logarithmic axes
# method 1: transform inside the plot(), plot(log(y)~log(x))
# method 2: transform by log='x'

par(mfrow=c(2,2))
data<-read.table('logplots.txt',header=T)
attach(data)
names(data)
plot(x,y,pch=16,main='untransformed data',col='red')
plot(log(x),log(y),pch=16,main='log-log data',col='blue')
plot(x,y,pch=16,log='xy',main="both transformed by log='xy'",col='blue')
plot(x,y,pch=16,log='y',main='only y transformed',col='green')

# 29.7 Different font families for text 
# 
par(mfrow=c(1,1))
plot(1:10,1:10,type='n',xlab='',ylab='')
par(family='sans')
text(5,8,'This is the default font')
?text
par(family='serif')
text(5,6,'This is serif font')
par(family='mono')
text(5,4,'This is mono font')
par(family='HersheySymbol')
text(5,2,'This is the symbol font')
# change back to default font 
par(family='sans')

# 29.8 Mathmatical and other symbols on plots 
# Refer Table 29.1 of 'The R Book' for a list of syntax or use the following help file
?plotmath
x <- seq(-4, 4, len = 101)
y <- cbind(sin(x), cos(x))
?matplot #Plot the columns of one matrix against the columns of another.
matplot(x, y, type = "l", xaxt = "n",
        main = expression(paste(plain(sin) * phi, "  and  ",
                                plain(cos) * phi)),
        ylab = expression(paste("sin" * phi, ' and ',  "cos" * phi)), # only 1st is taken
        xlab = expression(paste("Phase Angle ", phi)),
        col.main = "blue")
axis(1, at = c(-pi, -pi/2, 0, pi/2, pi),
     labels = expression(-pi, -pi/2, 0, pi/2, pi))


## How to combine "math" and numeric variables :
?substitute
plot(1:10, type="n", xlab="", ylab="", main = "plot math & numbers")
theta <- 1.23
mtext(bquote(hat(theta) == .(theta)), line= .25)
for(i in 2:9)
  text(i, i+1, substitute(list(xi, eta) == group("(",list(x,y),")"),
                          list(x = i, y = i+1)))
## note that both of these use calls rather than expressions.
##
text(1, 10,  "Derivatives:", adj = 0)
text(1, 9.6, expression(
  "first: {f * minute}(x) " == {f * minute}(x)), adj = 0)
text(1, 9.0, expression(
  "second: {f * second}(x) "== {f * second}(x)), adj = 0)


plot(1:10, 1:10)
text(4, 9, expression(hat(beta) == (X^t * X)^{-1} * X^t * y))
text(4, 8.4, "expression(hat(beta) == (X^t * X)^{-1} * X^t * y)",
     cex = .8)
text(4, 7, expression(bar(x) == sum(frac(x[i], n), i==1, n)))
text(4, 6.4, "expression(bar(x) == sum(frac(x[i], n), i==1, n))",
     cex = .8)
text(8, 5, expression(paste(frac(1, sigma*sqrt(2*pi)), " ",
                            plain(e)^{frac(-(x-mu)^2, 2*sigma^2)})),
     cex = 1.2)

## some other useful symbols
plot.new(); plot.window(c(0,4), c(15,1))
text(1, 1, "universal", adj = 0); text(2.5, 1,  "\\042")
text(3, 1, expression(symbol("\042")))
text(1, 2, "existential", adj = 0); text(2.5, 2,  "\\044")
text(3, 2, expression(symbol("\044")))
text(1, 3, "suchthat", adj = 0); text(2.5, 3,  "\\047")
text(3, 3, expression(symbol("\047")))
text(1, 4, "therefore", adj = 0); text(2.5, 4,  "\\134")
text(3, 4, expression(symbol("\134")))
text(1, 5, "perpendicular", adj = 0); text(2.5, 5,  "\\136")
text(3, 5, expression(symbol("\136")))
text(1, 6, "circlemultiply", adj = 0); text(2.5, 6,  "\\304")
text(3, 6, expression(symbol("\304")))
text(1, 7, "circleplus", adj = 0); text(2.5, 7,  "\\305")
text(3, 7, expression(symbol("\305")))
text(1, 8, "emptyset", adj = 0); text(2.5, 8,  "\\306")
text(3, 8, expression(symbol("\306")))
text(1, 9, "angle", adj = 0); text(2.5, 9,  "\\320")
text(3, 9, expression(symbol("\320")))
text(1, 10, "leftangle", adj = 0); text(2.5, 10,  "\\341")
text(3, 10, expression(symbol("\341")))
text(1, 11, "rightangle", adj = 0); text(2.5, 11,  "\\361")
text(3, 11, expression(symbol("\361")))


## 29.9 Phase Planes

plot(c(0,1),c(0,1),ylab="",xlab="",xaxt='n',yaxt='n',type='n')
# xaxt, yaxt supress the tick marks 
?abline # abline(a,b,h,v) a,b are intercept and slope repectively
abline(0.8,-1.5,col="blue")
abline(0.6,-0.8, col="red")
axis(1, at = 0.805, lab = expression(1/alpha[21]))
axis(1, at = 0.56, lab = expression(1/alpha[11]))
axis(2, at = 0.86, lab = expression(1/alpha[12]),las=1)
axis(2, at = 0.63, lab = expression(1/alpha[22]),las=1)

text(0.05,0.85, expression(paste(frac("d N"[1],"dt"), " = 0" )))
text(0.78,0.07, expression(paste(frac("d N"[2],"dt"), " = 0" )))
?arrows
arrows(-0.02,0.72,0.05,0.72,length=0.1)
arrows(-0.02,0.72,-0.02,0.65,length=0.1)
arrows(-0.02,0.72,0.05,0.65,length=0.1)
arrows(0.65,-0.02,0.65,0.05,length=0.1)
arrows(0.65,-0.02,0.58,-0.02,length=0.1)
arrows(0.65,-0.02,0.58,0.05,length=0.1)
arrows(0.15,0.25,0.15,0.32,length=0.1)
arrows(0.15,0.25,0.22,0.25,length=0.1)
arrows(0.15,0.25,0.22,0.32,length=0.1)
arrows(.42,.53,.42,.46,length=0.1)
arrows(.42,.53,.35,.53,length=0.1)
arrows(.42,.53,.35,.46,length=0.1)
axis(1, at = 1, lab = expression(N[1]))
axis(2, at = 1, lab = expression(N[2]),las=1)



## 29.10 Fat Arrows 
# Defined a function called fat.arrow to add arrows to plots
fat.arrow <- function(size.x=0.5,size.y=0.5,ar.col="red"){ 
  size.x <- size.x*(par("usr")[2]-par("usr")[1])*0.1
  size.y <- size.y*(par("usr")[4]-par("usr")[3])*0.1
  pos <- locator(1)
  xc <- c(0,1,0.5,0.5,-0.5,-0.5,-1,0)
  yc <- c(0,1,1,6,6,1,1,0) 
  polygon(pos$x+size.x*xc,pos$y+size.y*yc,col=ar.col) 
}
?polygon
# usr: A vector of the form c(x1, x2, y1, y2) giving the extremes of the user coordinates
# of the plotting region.

# Example of the use of fat.arrow 
plot(0:10,0:10,type="n",xlab="",ylab="")
fat.arrow()
fat.arrow(ar.col="green")
fat.arrow(ar.col="blue",size.x=0.8)
fat.arrow(ar.col="orange",size.x=0.8,size.y=0.3)
fat.arrow(ar.col="gray",size.x=0.2,size.y=0.3)

## 29.11 three dimensional plots with "akima" package 
data<-read.table('pgr.txt',header=T)
attach(data)
names(data)
# install.packages('akima')
library(akima)
zz<-interp(hay,pH,FR)
# now zz can be used in any of contour, filled.contour, image or persp
image(zz,col=topo.colors(12),xlab='biomass',ylab='pH')
contour(zz,add=T)
contour(zz)

filled.contour(zz,col=topo.colors(25),xlab='biomass',ylab='pH')
# I like this more as it has a clear color key for reference. 

?persp #This function draws perspective plots of a surface over the x–y plane. 
# persp is a generic function.
# Allows angled view of a 3D-like object

persp(zz,xlab='biomass',ylab='pH',zlab='Festuca Rubra',
      theta=45, phi=30,col='green')
detach(data)
x<-seq(0,10,0.1)
y<-seq(0,10,0.1)
func <-  function(x,y) 3 * x * exp(0.1*x) * sin(y*exp(-0.5*x))
func
image(x,y,outer(x,y,func))
contour(x,y,outer(x,y,func),add=T)
?outer # returns values(outer product) of func(x,y) 

## 29.12 Complex 3D plots with wireframe in package lattice 
library(lattice)
wireframe(volcano, shade =TRUE, aspect = c(61/87,0.4), screen = list(z = -120, x = -45),
          light.source = c(0,0, 10), distance = 0.2,
          shade.colors = function(irr, ref,height, w = 0.5) 
            grey(w * irr + (1 - w) * (1 - (1 - ref)^0.4))
)

wireframe(volcano, shade = TRUE,
          aspect = c(61/87, 0.4),
          light.source = c(10,0,10))
?wireframe

summary(volcano)


n <- 50
tx <- matrix(seq(-pi, pi, len = 2 * n), 2 * n, n)
ty <- matrix(seq(-pi, pi, len = n)/2, 2 * n, n, byrow = T)
xx <- cos(tx) * cos(ty)
yy <- sin(tx) * cos(ty)
zz <- sin(ty)
zzz <- zz
zzz[, 1:12 * 4] <- NA
wireframe(zzz ~ xx * yy, shade = TRUE, light.source = c(3,3,3))


# 29.13 Alphabetical tour of the graphics parameters 

# usr the limits of the current axes 
??usr
#A vector of the form c(x1, x2, y1, y2) giving the extremes of the user 
#coordinates of the plotting region. 
par('usr')

par('mar') # give the number of lines of the margin 

par(adj=0.5)
# adj=0 for left justified text, 0.5 for center text (default), 1 for right justified.

# ask=TRUE , logical. If TRUE (and the R session is interactive) the user is asked for input, 
# before a new figure is drawn


# bty: box type 

?par  # use par to check all the other options 


# 29.13.4 Control over the axes by axis 

plot(1:10,10:1,type='n',axes=FALSE,xlab='',ylab='') # No axes 
axis(1,1:10,LETTERS[1:10],col.axis='blue')
axis(2,1:10,letters[10:1],col.axis='red')
axis(3,lwd=3,col.axis='green')
axis(4,at=c(2,5,8),labels=c('one','two','three'))
box()  # if box() is not used, there will be gaps between the ends of each axis 


#29.13.5 Background color 
par(bg='cornsilk')
plot(1:10,10:1,type='n',axes=FALSE,xlab='',ylab='') # No axes 
axis(1,1:10,LETTERS[1:10],col.axis='blue')
axis(2,1:10,letters[10:1],col.axis='red')
axis(3,lwd=3,col.axis='green')
axis(4,at=c(2,5,8),labels=c('one','two','three'))
box()  # if box() is not used, there will be gaps between the ends of each axis 


# default 
par(bg='transparent')


# 29.13.6 Boxs around plots, bty 

par(mfrow=c(2,3))
plot(1:10,10:1,type='n',main='default complete box')
plot(1:10,10:1,type='n',bty='n',main='no box')
plot(1:10,10:1,type='n',bty='l',main='open on the left')
plot(1:10,10:1,type='n',bty='c',main='open on the right')
plot(1:10,10:1,type='n',bty='u',main='open on the top')
plot(1:10,10:1,type='n',bty='7',main='top and right only')



# 29.13.7 Size of plotting symbols using the character expansion function, cex 
par(mfrow=c(1,1))
plot(0:10,0:10,type='n',xlab='',yalb='')
for (i in 1:10) points(2,i,cex=i)
for(i in 1:10) points(6,i,cex=10+2*i)

# 29.13.8 Changing of the shape of the plotting region, plt 
# coordinates of the plot region as fractions of the current figure region
default_plt<-par('plt')
par(plt=c(0.15,0.94,0.3,0.7))
plot(c(0,3000),c(0,1500),type='n')
par(plt=default_plt)


# 29.13.9 Locating multiple graphs in non standard layout by fig 

# fig : coordinates of the figure region 

default_fig<-par('fig')

par(fig=c(0.5,1,0.5,1))
plot(0:10,25*exp(-0.1*0:10),col='blue',type='l')
par(fig=c(0,0.5,0,0.5))
plot(0:10,25*exp(-0.1*0:10),col='blue',type='l')


# 29.13.10 Two graphs with a common x scale but different y scales using fig 

data<-read.table('gales.txt',header=T)
attach(data)
names(data)


par(fig=c(0,1,0.5,1))
default_mar<-par('mar')

par(mar=c(0,5,2,2)) # set bottom margin to zero so the plot on top sit right on top of lower graph
plot(year,number,xlab='',xaxt='n',type='b',pch=16,col='blue') 
# xaxt set up axis but not plot 
par(fig=c(0,1,0,0.5),new=T) # figure for the bottom plot 
par(mar=c(2,5,0,2))
plot(year, February,xlab='Year',type='h',col='red') # h for histogram like vertical lines


# reset fig and mar 
par(mar=default_mar)
par(fig=default_fig)


# 29.13.11 The layout function 
# Another way to configure both location and shape of multip plotting regions independently

?pmin # Returns the (parallel) maxima and minima of the input values.
x<-pmin(3,pmax(-3,rnorm(50)))
x
# First compare -3 with 50 random variables and return the max one , then compare with 3 and 
# return the min one 

y<-pmin(3,pmax(-3,rnorm(50)))
xhist<-hist(x,breaks=c(-3,3,0.5),plot=FALSE)
yhist<-hist(y,breaks=c(-3,3,0.5),plot=FALSE)

top<-max(c(xhist$counts,yhist$counts))
xrange<-c(-3,3)
yrange<-c(-3,3)

matrix(c(2,0,1,3),2,2,byrow=TRUE)

nf<-layout(matrix(c(2,0,1,3),2,2,byrow=TRUE),c(3,1),c(1,3),TRUE) #2,0,1,3 is the corresponding figure name
layout.show(nf)

# layout(matrix,widths,heights,respect=TRUE)
# matrix specifies the location of the next n figures, left to right, top to bottom 


par(mar=c(3,3,1,1))
plot(x,y,xlim=xrange,ylim=yrange,pch=21,col='blue',bg='red')
par(mar=c(0,3,1,1))
barplot(xhist$counts,axes=FALSE,col='green',ylim=c(0,top),space=0,bin=5)
# space : the space between bars 

par(mar=c(3,0,1,1))
barplot(yhist$counts,axes=FALSE,col='green',xlim=c(0,top),space=0,horiz=TRUE)


# 29.13.12 Create and control multiple screens on a single device by split.screen
# Should complete each graph before moving on to the next screen 
# erase.screen is used clear plots 
# close.screen removes the specified screen 
# close.screen(all=TRUE) exit split screen 
default_mar<-c(5.1,4.1,4.1,2.1)
par(mar=default_mar)
par(fig=default_fig)
fig.mat<-c(0,0,0.5,0.5,1,0.5,1,1,0.7,0,0.35,0,1,0.7,0.7,0.35)
fig.mat<-matrix(fig.mat,nrow=4)
fig.mat

dev.off()

split.screen(fig.mat)
par(mar=c(2,5,1,2))
screen(1)
plot(year,number,type='l',col='blue')
screen(2)
par(mar=c(4,5,0,1))
plot(year,February,type='h',col='red')
par(mar=c(4,5,0,1))
screen(3)
plot(1:10,0.5*(1:10)^0.5,xlab='Concentration',ylab='rate',type='l',col='green4')
par(mar=c(4,5,0,1))
screen(4)
plot(1:10,600*exp(-0.5*(1:10)),xlab='Time',ylab='residue',type='l',col='green4')
close.screen(all=TRUE)


detach(data)


# 29.13.14 Shapes for the ends and joins of lines, lend and join 
plot(0:10,0:10,type='n',xlab='',ylab='')
lines(c(2,5,8),c(8,2,8),lwd=50,lend='square',ljoin='mitre')
lines(c(2,5,8),c(8,2,8),lwd=50,ljoin='round',lend='round',col='green')
lines(c(2,5,8),c(8,2,8),lwd=50,lend='butt',ljoin='bevel',col='red')

# a random walk 
x<-numeric(100)
y<-numeric(100)
x[1]<-1
y[1]<-1
for (i in 2:100){
  a<-runif(1)*2*pi
  d<-runif(1)*2*pi
  x[i]<-x[i-1]+d*sin(a)
  y[i]<-y[i-1]+d*cos(a)
}

# To show the road map of a random walk, first draw lines with a bigger width 
# then draw same thing with smaller width 
plot(c(min(x),max(x)),c(min(y),max(y)),type='n',xaxt='n',yaxt='n',xlab='',ylab='')
lines(x,y,lwd=13,lend='round',ljoin='round')
lines(x,y,lwd=10,lend='round',ljoin='round',col='red')


# line type 1...6 


# 29.13.20  Two graphs on the same plot with different scales in y, new=T 
gales<-read.table('gales.txt',header=T)
attach(gales)
default_mar<-par('mar')
par(mar=c(5,4,4,4)+0.1)
plot(year,number,type='l',lwd=2,las=1,col='blue')
par(new=T)
plot(year,February,type='h',axes=F,ylab='',lty=2,col='red')
axis(4,las=1)
mtext(side=4,line=2.5,'Feb Gales')
detach(gales)

par(mar=default_mar)
par(new=FALSE)

# 29.13.21 Out margin, oma 
# Useful when extra space needed for labels/titles 

default_outer<-par('oma')

attach(anscombe)
par(mfrow=c(2,2),oma=c(0,0,2,0))
plot(x1,y1,main='set 1',col='red',pch=21,bg='orange',xlim=c(0,20),ylim=c(0,16))
abline(lm(y1~x1),col='navy')
plot(x2,y2,main='set 2',col='red',pch=21,bg='orange',xlim=c(0,20),ylim=c(0,16))
abline(lm(y2~x2),col='navy')
plot(x3,y3,main='set 3',col='red',pch=21,bg='orange',xlim=c(0,20),ylim=c(0,16))
abline(lm(y3~x3),col='navy')
plot(x4,y4,main='set 2',col='red',pch=21,bg='orange',xlim=c(0,20),ylim=c(0,16))
abline(lm(y4~x4),col='navy')
mtext('Anscombe regression',outer=T,cex=1.5) # outer=T allows writing title in outer margin
detach(anscombe)


# 29.13.22 Packing graphs closer together 
par(mfrow=c(3,3))
par(mar=c(0.2,0.2,0.2,0.2))
par(oma=c(5,5,0,0))
for (i in 1:9) plot(sort(runif(100)),sort(runif(100)),
                    xaxt='n',yaxt='n',pch=21,bg='green')
title(xlab='time',ylab='distance',outer=T,cex.lab=2)

dev.off()


# 29.13.24 Character rotation, srt (string rotation)

plot(1:10,1:10,type='n',xlab='',ylab='')
for (i in 1:10) text(i,i,LETTERS[i],srt=(20*i),col='red')
for (i in 1:10) text(10-i+1,i,letters[i],srt=(20*i),col='blue')


# 29.13.25 Rotating the axis labels 
spending<-read.table('spending.txt',header=T)
spending
attach(spending)
par(mar=c(7,4,4,2)+0.1)
xvals<-barplot(spend,ylab='Spending',col='wheat2')
text(xvals,par('usr')[3]-0.25,srt=45,adj=1,labels=country,xpd=T)
# adj=1 for right justified
??xpd
#  xpd, If FALSE, all plotting is clipped to the plot region, 
# if TRUE, all plotting is clipped to the figure region, 
# and if NA, all plotting is clipped to the device region.
dev.off()
# 29.13.26 Tick marks on the axes 
par(mfrow=c(2,2))
plot(1:10,1:10,xlab='',ylab='',type='n',main='default ticks')
plot(1:10,1:10,xlab='',ylab='',type='n',main='maximum ticks',tck=1)
plot(1:10,1:10,xlab='',ylab='',type='n',main='no ticks',tck=0)
plot(1:10,1:10,xlab='',ylab='',type='n',main='interior ticks',tck=0.05)
??tck
# tck, The length of tick marks as a fraction of the smaller of the width or
# height of the plotting region. If tck >= 0.5 it is interpreted as a fraction
# of the relevant side, so if tck = 1 grid lines are drawn. 
# The default setting (tck = NA) is to use tcl = -0.5.
# tcl length of the tick 

dev.off()
plot(1:10,1:10,xlab='',ylab='',main='default ticks',xaxs='i')
# xaxs = r, i


# 29.14 Trellis graphics
# Mainly used to produce multiple plots per page and multi-page plots, particularly in the 
# context of mixed effects modelling. 

data<-read.table('panels.txt',header=T)
data
attach(data)
library(lattice)
xyplot(weight~age|gender)
# xyplot produces bivariate scatterplots or time-series plots, 
# bwplot produces box-and-whisker plots, 
# dotplot produces Cleveland dot plots,
# barchart produces bar plots, 
# and stripplot produces one-dimensional scatterplots
detach(data)

# Change par() settings has NO effect on lattice plots. 

# From xyplot help files
require(stats)

## Tonga Trench Earthquakes

Depth <- equal.count(quakes$depth, number=8, overlap=.1)
xyplot(lat ~ long | Depth, data = quakes)
update(trellis.last.object(),
       strip = strip.custom(strip.names = TRUE, strip.levels = TRUE),
       par.strip.text = list(cex = 0.75),
       aspect = "iso")

## Examples with data from `Visualizing Data' (Cleveland, 1993) obtained
## from http://cm.bell-labs.com/cm/ms/departments/sia/wsc/

EE <- equal.count(ethanol$E, number=9, overlap=1/4)

## Constructing panel functions on the fly; prepanel
xyplot(NOx ~ C | EE, data = ethanol,
       prepanel = function(x, y) prepanel.loess(x, y, span = 1),
       xlab = "Compression Ratio", ylab = "NOx (micrograms/J)",
       panel = function(x, y) {
         panel.grid(h = -1, v = 2) #horizontal and vertical space 
         panel.xyplot(x, y)
         panel.loess(x, y, span=1)
       },
       aspect = "xy")
# loess add a smooth line based on data 


## Extended formula interface 

xyplot(Sepal.Length + Sepal.Width ~ Petal.Length + Petal.Width | Species,
       data = iris, scales = "free", layout = c(2, 2),
       auto.key = list(x = .6, y = .7, corner = c(0, 0)))


## user defined panel functions

states <- data.frame(state.x77,
                     state.name = dimnames(state.x77)[[1]],
                     state.region = state.region)
xyplot(Murder ~ Population | state.region, data = states,
       groups = state.name,
       panel = function(x, y, subscripts, groups) {
         ltext(x = x, y = y, labels = groups[subscripts], cex=1,
               fontfamily = "HersheySans")
       })



## Grouped dot plot showing anomaly at Morris

dotplot(variety ~ yield | site, data = barley, groups = year,
        key = simpleKey(levels(barley$year), space = "right"),
        xlab = "Barley Yield (bushels/acre) ",
        aspect=0.5, layout = c(1,6), ylab=NULL)

stripplot(voice.part ~ jitter(height), data = singer, aspect = 1,
          jitter.data = TRUE, xlab = "Height (inches)")

## Interaction Plot

xyplot(decrease ~ treatment, OrchardSprays, groups = rowpos,
       type = "a",
       auto.key =
         list(space = "right", points = FALSE, lines = TRUE))

## longer version with no x-ticks

bwplot(decrease ~ treatment, OrchardSprays, groups = rowpos,
       panel = "panel.superpose",
       panel.groups = "panel.linejoin",
       xlab = "treatment",
       key = list(lines = Rows(trellis.par.get("superpose.line"),
                               c(1:7, 1)),
                  text = list(lab = as.character(unique(OrchardSprays$rowpos))),
                  columns = 4, title = "Row position"))

# 29.14.1 Panel box-and-whisker plots 
data<-read.table('daphnia.txt',header=T)
attach(data)
names(data)
bwplot(Growth.rate~Water+Daphnia|Detergent)
detach(data)
# 29.14.2 Panel scatterplot
results<-read.table('fertilizer.txt',header=T)
attach(results)
names(results)
xyplot(root~week|plant)
# the panels are shown in alphabetical order by plant name 

xyplot(root~week|plant, pch=16)


# If we want to fit a separate linear regression for each individual plant, need to use 'panel'

xyplot(root~week|plant,
       panel=function(x,y){
         panel.xyplot(x,y,pch=16)
         panel.abline(lm(y~x))
       })

# If we want to do different things in different panels 

# example, draw a highlighted line at the location of the fourth data point in each panel
xyplot(root~week|plant,
       panel=function(x,y){
         panel.xyplot(x,y,pch=16)
         panel.abline(lm(y~x))
         panel.abline(h=y[4],col='red',lty=3)
       })


# Add a text label to each panel to show the panel number in purple 
xyplot(root~week|plant,
       panel=function(x,y){
         panel.xyplot(x,y,pch=16)
         panel.abline(lm(y~x))
         panel.abline(h=y[4],col='red',lty=3)
         panel.text(8,2,panel.number(),col='purple',cex=0.7)
       })

detach(results)
# 29.14.3 Panel barplots 
## Stacked bar chart

barchart(yield ~ variety | site, data = barley,
         groups = year, layout = c(2,3), stack = TRUE,
         #col=c('cornflowerblue','blue'),
         auto.key = list(space = "right"), # a suitable legend will be drawn by groups
         ylab = "Barley Yield (bushels/acre)",
         scales = list(x = list(rot = 45)))
# layout shows that it produces 3 rows with 2 plots each row 
# scales is used to rotate the long x labels 
bwplot(voice.part ~ height, data=singer, xlab="Height (inches)")

dotplot(variety ~ yield | year * site, data=barley)

# 29.14.5 Panel histograms 
data<-read.table('SilwoodWeather.txt',header=T)
attach(data)
names(data)

histogram(~lower|as.factor(month),type='count',
          xlab='minimum temperature',
          ylab='frequency',
          breaks=seq(-12,28,2),
          strip=strip.custom(bg='lightgrey',par.strip.text=list(col="black", cex=.8, font=3))
)
detach(data)


library(lattice)
histogram(~height | voice.part, data = singer,
          strip = strip.custom(bg="lightgrey",
                               par.strip.text=list(col="black", cex=.8, font=3)),
          main="Distribution of Heights by Voice Pitch",
          xlab="Height (inches)") 

# 29.14.6 Effect sizes 
# Use effects package which takes a model object and provides trellis plots of specified effects
library(effects)
data<-read.table('daphnia.txt',header=T)
attach(data)
model<-lm(Growth.rate~Water*Detergent*Daphnia)
daph.effects<-allEffects(model) # calculate all effects and plot by specifying interactions in quote
plot(daph.effects,'Water:Detergent:Daphnia',main='Effects')
detach(data)

# 29.14.7 More panel functions 

# Use built-in data OrchardSprays
# groups=rowpos assigns different color for each group
#
xyplot(decrease~treatment,OrchardSprays,groups=rowpos,
       type='a',
       auto.key=
         list(space='right',points=TRUE,lines=TRUE))
xyplot(decrease~treatment,OrchardSprays,groups=rowpos,
       type='a',
       panel='panel.superpose',
       auto.key=
         list(space='right',points=TRUE,lines=TRUE))
