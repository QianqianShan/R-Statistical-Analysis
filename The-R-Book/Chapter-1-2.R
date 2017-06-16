setwd("~/Dropbox/R-Statistical-Analysis/The-R-Book/therbookdata")
getwd()
##Chapter 1 Getting Started 
############################
citation()
contributors()

# 1.5 Getting help in R 
help.search("ggplot2")
find("lowess") # tells what package the quoted name in 

apropos("lm") # gives a character vector giving the names of all objects that match the (partial) equiry

# see a worked example
example(lm)

# see demenstrations of r functions
demo(persp)
demo(graphics)
demo(Hershey)
demo(plotmath)


## Install packages

# lattice graphics for panel plot or trellis (gezi) graphs
# install.packages("lattice")

# Modern applied statistics using s plus
# install.packages("MASS")

# generalized additive model
# install.packages("mgcv")

# mixed effects model both linear and non-linear
# install.packages("nlme")

# feed forward neural networks and multinomial log linear models
# install.packages("nnet")

# functions for kriging and point pattern analysis
# install.packages("spatial")

# survival analysis including penalized likelihood
# install.packages("survival")

# get contents in the packages

###################################
# 1.6.1 Contents of packages 
library(help=spatial)

# Packages to be used 
# install.packages("akima")
# install.packages("boot")
# install.packages("car")
# install.packages("lme4")
# install.packages("meta")
# install.packages("mgcv")
# install.packages("deSolve")
# install.packages("R2jags")
# install.packages("RColorBrewer")
# install.packages("RODBC")
# install.packages("rpart")
# install.packages("spatstat")
# install.packages("spdep")
# install.packages("tree")


# 1.10 Good housekeeping 
objects() # show variables that have been created in the current session
search()  # see the currently attached packages 

#remove (rm) any variables names and detach any dataframes you have created at the end session
rm(x)

detach()
## The detach command does not make the dataframe called worms disappear;
# it just means that the variables within worms, such as Slope and Area, are no longer accessible directly by name. 
#To get rid of everything, including all the dataframes, type

rm(list=ls()) # Get rid of everything 

##########################################
# Chapter 2: Essentials of the R language
##########################################

# 2.1 Calculations 
2+3;4+5;

#complex number
z<-3.5+8i
Re(z)
Im(z)

# Modulus of complex number
Mod(z)

# Argument 
Arg(z)

Arg(z)*180/pi

# Conjugate
Conj(z)

is.complex(z)

as.complex(z)

# Rounding 

floor(5.7)
ceiling(5.7)

signif(3.45678,3)

# log to the base e
log(10)

# log to the base 10
log10(10)

# log with other bases

log(9,3) # log to base 3 of 9

# other mathmatical functions

choose(10,3) #choose (n,x), binomial coefficients

lgamma(10) #natural log of gamma(x)

trunc(3.4) # closest integer to 0

acos(1/2)*180/pi


119%/%9 #quotient 

119%%9 #remainder,modulo

# set up integer array
x<-c(5,3,7,8)
class(x)
x<-as.integer(x)
is.integer(x)

# Logical operations
T==F

# usage of all.equal
# unlike "identical", all.equal allows for insignificant differences
x<-.3-.2
y<-.1
all.equal(x,y)
identical(x,y)
#summarize differences using all.equal

x<-factor(c('1','2'))
x
as.vector(unclass(x)) # turn factor into integer
# unclass returns (a copy of) its argument with its class attribute removed.


# 2.2 Logical operations 


# Equality of floating number using all.equal 
x<-0.3-0.2
y<-0.1
x==y
identical(x,y)
all.equal(x,y)
# only all.equal gives the TRUE as it allows for insignificant differences

# Note: don't use all.equal directly in if expressions, use isTRUE(all.equal(...)) or identical 

# summarize differnces between objects using all.equal 
a<-c('cat','dog','goldfish') # character array 
b<-factor(a)

class(b)
mode(b) # factors are stored internally as integers 
# MODE Get or set the type or STORAGE mode of an object.
all.equal(a,b)
# "target is character, current is factor"     

attributes(b) # is a list with levels and class, corresponds to
# "Attributes:< target is NULL, current is list >" from the results of the above all.equal()



# combination of T&F
# Always write TRUE and FALSE in FULL, never use T and F as variable names to avoid confusions.
x<-c(NA,TRUE,FALSE)
names(x)<-as.character(x)
names(x)
x

outer(x,x,"&") # outer product of arrays
outer(x,x,"|")


# 2.3 Generating sequences 
sequence(c(4,3,4,4,4,5)) #first generate seq(1,4,1),then seq(1,3,1)...
 #generate a vector make up of sequence of unequal lengths

#genearate repeats,differentiate the following commands

rep(1:4,2)

rep(1:4,each=2)

rep(1:4,each=2,times=3)

rep(1:4,1:4)

rep(1:4,c(4,1,4,2))

rep(c("cat","goldfish","dog","rat"),c(3,2,5,1))

# generate factor levels by "gl(generate levels)"
gl(4,3) # 4 is the number of levels, 3 is the number of replications
?gl
gl(2, 8, labels = c("Control", "Treat"))

temp<-gl(2,2,24,labels=c('low','high'))
temp
soft<-gl(3,8,24,labels=c('hard','medium','soft'))
data.frame(temp,soft)


# 2.4 Membership: Testing and coercing in R  is.  and as. 

lv<-c(T,F,T)
is.logical(lv)
levels(lv)
# lv is logical, and has no levels 

fv<-as.factor(lv) # coerce logical variables to factor
is.factor(fv)
levels(fv)

nv<-as.numeric(lv) # coerce to numeric values  
nv
levels(nv)

# Summary testing with is.logical and coercing with as.logical .

## Types 
# array 
# character
# complex 
# dataframe 
# double 
# factor 
# list 
# logical 
# matrix 
# numeric 
# raw 
# ts (time series)
# vector 



# 2.5 Missing values, infinity and things that are not numbers 
3/0 # plus infinity
0/0 # NaN

# Note: Difference between NaN and NA 
# NaN : not a number 
# NA: not available, missing values 

# built in functions to check if a number is finite/infinite/NaN/NA 
is.finite(10)
is.infinite(Inf)
is.na(NA)
is.na(NaN)
is.nan(NA)
is.nan(NaN)

# produce a vector with NA stripped out 
y<-c(4,NA,7)
y[!is.na(y)]

x<-c(1:8,NA)
mean(x)
mean(x,na.rm=TRUE) # remove NA to make it working 


# Use "which" to locate NAs 
vmv<-c(1:6,NA,NA,8:9)
which(is.na(vmv))


# replace NA values with 0 , method 1
vmv1<-vmv
vmv1[is.na(vmv1)]<-0
vmv1

# replace NA values with 0 , method 2
ifelse(is.na(vmv),0,vmv)



# 2.6 Vectors and subscripts 
# A vector is a variable with one or more values of the SAME type. 

# Create a vector method 1
peas<-c(3,4,5,6,7,7)
quantile(peas)
quantile(peas,probs=seq(0,1,0.15))

# Create a vector method 2
peas<-scan() # hit Enter without any values to indicate the end of input 

# 2.6.3 Naming elements within vectors 

# add labels for vector 
counts<-c(25,35,46,4,3,2)
names(counts)<-0:5
counts

# remove the names 
st<-table(rpois(2000,3))
st
st<-as.vector(st) # convert to vector types 

sort(st)
rev(sort(st))
rev(sort(st))[1:3]

which(st==max(st))
which.max(st)
# similarly, which.min


# 2.7 Vector functions 
# Evaluate functions over entire vectors to avoid loop and subscripts 

y<-c(12,2,323,3,23,4,23,2,1)
range(y)
var(y)
cumprod(y)
cumsum(y)

fivenum(y)
# Tukey's five number summary: min, lower hinge,median,upper hinge,max 

counts<-rnbinom(10000,mu=0.92,size=1.1)
table(counts)

# table means/max/min... using tapply
data<-read.table('SilwoodWeather.txt',header=T)
names(data)
attach(data)
tapply(lower,month,mean)
# tapply(x,index, function)
tapply(lower,list(yr,month),mean) # multiple dim tables 

tapply(lower,yr,mean,trim=0.2)

detach(data)
# trim allows us to specify the fraction of the data (0-0.5) that you want to be omitted 
# from the left and right hand tails of the sorted vector values before computing the function


# 2.7.2 The aggregate function for grouped summary statistics 
# Useful for removing pseudoreplication from dataframes
data<-read.table('daphnia.txt',header=T)
names(data)
aggregate(Growth.rate~Water,data,mean) # one to one
?aggregate

aggregate(Growth.rate~Water+Detergent,data,mean) # one to many 
# mean is applied to Growth.rate with different classes of Water and Detergent. 


# 2.7.3 Parallel minima and maxima: pmin,pmax 

x<-sample(1:9,6)
y<-sample(2:7,6)
z<-sample(4:10,6)
cbind(x,y,z)
pmin(x,y,z)
pmax(x,y,z)
# produce a vector with the same length as x, y ,z 

# 2.7.4 Summary information from vectors by groups 

attach(data)
tapply(Growth.rate,Detergent,mean)
# tapply means apply a function to produce a Table from the values in the Table
tapply(Growth.rate,list(Water,Detergent,Daphnia),mean)

# 2.7.5 Addresses within vectors : which 
y<-sample.int(11,8)
which(y>5)
y
y[y>5]
length(y[y>5])
detach(data)
# 2.7.6 Finding closest values 
xv<-rnorm(1000,100,10)

# find the closest value to 108 
which(abs(xv-108)==min(abs(xv-108)))
xv[which(abs(xv-108)==min(abs(xv-108)))]

# write in the form of a function 
closest<-function(xv,sv){
  xv[which(abs(xv-sv)==min(abs(xv-sv)))]
}

closest(xv,108)

# 2.7.7 Sorting, ranking and ordering 

houses<-read.table('houses.txt',header=TRUE)
attach(houses)
names(houses)
ranks<-rank(Price) # returns a vector which shows the ranks of the corresponding Price
sorted<-sort(Price) # returns the sorted Price in ascending order BUT may uncouple with the 
                    # other variables, a dangerous way to sort
ordered<-order(Price) # returns the subscripts of the ascending ordered Price
view<-data.frame(ranks,sorted,ordered,Price)
view
?rank
?sort
?order

# Use order the obtain price ranked list of locations 
Location[ordered]
Location[order(Price,decreasing=TRUE)] # reversed order

detach(houses)

# 2.7.8 Understanding the difference between 'unique' and 'duplicated'

names<-c('cat','dog','dog','cat','dog','fish','food')
table(names)
unique(names)
# unique returns a vector, data frame or array like x 
# but with duplicate elements/rows removed.
duplicated(names)
# duplicated() determines which elements of a vector or data frame are duplicates 
# of elements with smaller subscripts, 
# and returns a logical vector indicating which elements (rows) are duplicates.
?duplicated
?unique

# mimic what 'unique' does by 'duplicated'
names[!duplicated(names)]

# 2.7.9 Looking for runs of numbers within vectors : 'rle'
# rle : run length encoding 
poison<-rpois(150,0.7)
rle(poison)

# Find the longest run 
max(rle(poison)[[1]])
which(rle(poison)[[1]]==max(rle(poison)[[1]]))
rle(poison)[[2]][which(rle(poison)[[1]]==max(rle(poison)[[1]]))]

# 2.7.10 Sets: union, intersect and setdiff 
setA<-c(letters[1:5])
setA
setB<-c(letters[4:7])
setB

union(setA, setB)
intersect(setA,setB)

setdiff(setA,setB) # elements in the first set but not in the second one 
setdiff(setB,setA) # order dependent 

setequal(setA,setB) # test if two sets are equal 

# use %in% to test 
setA %in% setB

setA[setA %in% setB]==intersect(setA,setB)
all.equal(setA[setA %in% setB],intersect(setA,setB))
identical(setA[setA %in% setB],intersect(setA,setB))




# 2.8 Matrices and arrays 
y <- 1:24 
dim(y) <- c(2,4,3)  # 3 2*4 tables 
y

dim(y) <- c(3,2,4)
y

# A matrix is a two dim array containing numbers 
# A data frame is a two dim list containing numbers, text or logical variables 


# 2.8.1 Matrices 

# Create matrix: method 1
x <- matrix(c(1,0,0,0,1,2,2,2,2),nrow=3) # arrange by column by default 
x
class(x)
attributes(x)

y <- c(seq(1:8))
v <- matrix(y,byrow=T,nrow=2)
v

# Create matrix: method 2, convert a vector to matrix 
dim(y) <- c(4,2)
y # column wise arranged 
t(y)
is.matrix(y)

# 2.8.2 Naming the rows and columns of matrices 

# method 1: rownames, colnames 
x <- matrix(rpois(20,1.6),nrow=4)
x
rownames(x) <- rownames(x,do.NULL = FALSE, prefix = "Trial")
x

# add columns names 
drug.names <- c("aspirin", "paracetamol", "nurofen", "hedex", "placebo")
colnames(x) <- drug.names
x

# method 2: use dimnames 
dimnames(x) <- list(NULL, paste("drug.", 1:5, sep = '')) # list ( rownames, colnames)
x
dimnames(x) <- list(paste("Trial.",1:4, sep = ''), paste("drug.", 1:5, sep = ''))
x

# 2.8.3 Caculations on rows or columns of the matrix 

mean(x[, 5])

var(x[4, ])

# calculate summary statistics on matrices 
rowSums(x)
colSums(x)
colMeans(x)
rowMeans(x)

apply(x, 2, mean) 

# sum groups of rows within columns 

# method 1 : rowsum 
group <- c('A', 'B', 'A', 'B')
rowsum(x,group)

# method 2 : tapply
tapply(x, list(group[row(x)], col(x)), sum)
list(group[row(x)], col(x))

aggregate(x, list(group), sum)
list(group)

# shuttfle the elements of each column of a matrix independently 
apply(x, 2, sample)
x



# 2.8.4 Adding rows and columns to the matrix   

# adding rows and columns to the matrix 
x <- rbind(x, apply(x, 2, mean))
x
x <- cbind(x,apply(x, 1, var))
x

# add names for the added row and column 
colnames(x) <- c(drug.names, "variance")
rownames(x) <- c(paste("Trial",1:4), "mean")
x

# Add drop = FALSE to prevent it becoming a vector form a matrix 
rowmatrix <- x[2, , drop=FALSE]
rowmatrix

rowmatrix <- x[2, ]
rowmatrix



# 2.8.5 sweep function 

# "sweep" function is used to "sweep out" array summaries from vectors, matrices, arrays or 
#  dataframes. 

matdata <- read.table("sweepdata.txt")
head(matdata)
cols <- apply(matdata, 2, mean) # mean of each column 
cols

head(sweep(matdata, 2, cols))

class(sweep(matdata, 2, cols))
?sweep
# Returns a data frame which shows the departure of each value from the colmean mean 
# 2 (margin) is a vector of indices giving the extent(s) of x which correspond to cols, i.e., 
# sweep by column. 

sweep(matdata, 1, 1:10, function(a, b) b) # subscripts for a columns wise sweep of data
sweep(matdata, 2, 1:4, function(a, b) b) # subscripts for a row wise sweep of the data 


# 2.8.6 Applying functions with apply, sapply, lapply 
x <- matrix(1:24, nrow = 4)
x

apply(x, 1, sum)

apply(x, 2, sum)

apply(x, 1, sqrt)

apply(x, 1, sample)
t(x)

apply(x, 1, function(x) x^2 + x)
x

sapply(3:7, seq)
?sapply

# sapply is a user-friendly version and wrapper of lapply by default returning a vector,
# matrix or, if simplify = "array", an array if appropriate, 
# by applying simplify2array(). sapply(x, f, simplify = FALSE, USE.NAMES = FALSE) 
# is the same as lapply(x, f).


sapdecay <- read.table("sapdecay.txt", header = TRUE)
attach(sapdecay)
names(sapdecay)
sapdecay
x <- sapdecay$x
y <- sapdecay$y

sumsq <- function(a, xv = x, yv = y){
  yf <- exp(-a*xv)
  sum((yv - yf)^2)
}

lm(log(y)~x)
a <- seq(0.01, 0.2, 0.005)
b <- sapply(a, sumsq)
plot(a, sapply(a, sumsq), type = "l") # shows under which value we could obtain a minmum 

# Extract the minimum value of a , method 1
a[min(sapply(a, sumsq)) == sapply(a, sumsq)]
plot(x, y, pch = 16)
plot(x, y, pch = c(0:18)) # see all different types of points characters 
xv <- seq(0, 50, 0.1)
lines(xv, exp(- 0.055 * xv))

# Extract the minimum value of a, method 2 
fa <- function(a) sum((y - exp(- a * x))^2)
optimize(fa, c(0.01, 0.1)) # searches the interval from lower to upper for minimum or maximum
?optimize
detach(sapdecay)
rm(list=c(x,y,xv))

# 2.8.7 Using the "max.col" function 
?max.col #Find the maximum position for each row of a matrix, breaking ties at random.

data <- read.table("pgfull.txt", header = T)
attach(data)
names(data)
head(data)
dim(data) # 89 rows and 59 columns

species <- data[, 1:54] # dataframe which only contains the species abundance 

max.col(species)

# get the indentity of the dominant 
domi <- names(species)[max.col(species)]
domi

# count up the total number of plots on which each species was dominant 
table(domi)

# the total # of species which are dominant
length(table(domi))

# use max.col to realize the "min.col"
max.col(-species) # NOT working well when tied 


# 2.8.8 Restructuring  a multi-dim array using "aperm"
?aperm # Transpose an array by permuting its dimensions and optionally resizing it.

data <- array(1:24, 2:4) # 4 sub-tables with 2 * 3 dim for each   
data

# add dimnames for rows, columns and tables 
dimnames(data)[[1]] <- list("male", "female")
dimnames(data)[[2]] <- list("young", "mid", "old")
dimnames(data)[[3]] <- list("A", "B", "C", "D")
data

# change the 4 A B.. groups to be the columns, and separate sub-tables by gender 
new.data <- aperm(data, c(2, 3, 1) ) # the the subscript permutation vector in terms of 
                                     # the original subscript of names 
new.data


# 2.9 Random numbers, sampling and shuffling 

set.seed(375) # set.seed can be useful to get the same string of random numbers as last time
runif(3)

# obtain part of the same series of random numbers : use .Random.seed 
?.Random.seed # an integer vector, containing the random number generator (RNG) state for 
              # random number generation in R. It can be saved and restored, but should 
              # not be altered by the user.

current <- .Random.seed
head(current)
runif(3)

runif(3)

# reset .Random.seed 
.Random.seed <- current
runif(3)  # same as the first one 

# the "sample" function

y <- rnorm(10)
sample (y)

sample(y, 5)

# Basis of bootstrapping 
sample(y, replace = TRUE)

# sample with specified different probabilities with which each element is to be sampled 
y <- 1:10
p <- c(seq(1:5), seq(5, 1))
sapply(1:5,function(i) sample(y, 4, prob = p)) # tell by column, return 5 columns for random sample


# 2.10 Loops and repeats 

# Ways to write the factorial:  "for" loop
fac1 <- function(x){
  f <- 1
  if (x <2) return(1)
  for (i in 2:x) {
    f <- f*i
  }
  f
}

sapply(0:5, fac1)

# Ways to write the factorial: "while" loop
fac2 <- function(x) {
  f <- 1
  t <- x
  while (t > 1) {
    f <- f*t
    t <- t-1
  }
  return(f)
}

sapply(0:5, fac2)

# Ways to write the factorial function: "repeat" 
fac3 <- function(x) {
  f <- 1
  t <- x
  repeat {
    if (t < 2) break  # logical escape clause 
    f <- f*t 
    t <- t-1
  }
  return(f)
}

sapply(0:5, fac3)

# Ways to write the factorial function : "cumprod"
cumprod(1:5)
cumprod(0:5)  # all values are ZERO! Not working for 0 case. 

# Ways to write the factorial function: modification of cumprod
fac4 <- function(x) max(cumprod(1:x))

sapply(0:5, fac4)

# Ways to write the factorial function: built-in function "gamma"
fac5 <- function(x) gamma(x+1)

sapply(0:5, fac5)

# Ways to write the factorial function: "factorial"
factorial(0:5)
# end 

# 2.10.1 binary representation of a number 



# generate Fibonacci series 
fibo <- function(n) {
  a <- 1
  b <- 0
  while (n > 0) {
    swap <- a
    a <- a + b
    b <- swap 
    n <- n-1
  }
  b
}

sapply(1:10, fibo)

# 2.10.2 Loop avoidance : avoid using loops whenver possible 
y <- rnorm(10)

z <- ifelse(y < 0 ,-1, 1) 
z

# convert continuous varible to two level factor 
data <- read.table("worms.txt", header = TRUE)
attach(data)
ifelse(Area > median(Area), "big", "smale")

# or use "cut "
cut(Area, 2, labels = FALSE)  # 2 is the number of intervals which x will be cut 
Area
cut(Area, 2)


# ifelse can also be used for natural inclinations 
y <- log(rpois(1000, 1.5))
ifelse(y < 0, NA, y) # replace -Inf with NA 

# 2.10.3 The slowness of loops 
x <- runif(1000000)
system.time(max(x)) # time to find the max x value 
?system.time

pc <- proc.time()
pc
?proc.time # proc.time determines how much real and CPU time (in seconds) the 
           # currently running R process has already taken.

cmax <- x[1] 
for (i in 2:1000000) {
  if (x[i] > cmax) cmax <- x[i]
}

proc.time() - pc # loop is much slower 

# 2.10.4 Do not "grow" data by concatenation or recursive function calls 

test1 <- function() {
  y <- 1:100000
}

test2 <- function() {
  y <- numeric(100000)
  for (i in 1:100000) y[i] <- i
}

test3 <- function() {
  y <- NULL
  for (i in 1:100000) y <- c(y, i)
}

proc.time()
system.time(test1())
system.time(test2())
system.time(test3()) # the longest time 20 seconds 


# 2.10.5 Loops for producing time series 
next.year <- function(x, lambda) {
   lambda * x * (1-x)
  }

next.year(0.6, 3.7)

n <- numeric(30)
n[1] <- 0.6
for (i in 2:30) n[i] <- next.year(n[i-1], 3.7)
n
plot(n, type = "l")
# Plot is quadratic map, related to chaos theory.  sensitve to inital values. 



# 2.11 Lists 
# Lists are important objects in R. 

apple <- c(sample(1:10,replace = TRUE))
orange <- c(TRUE, FALSE, TRUE)
chalk <- c("limeston", "marl")
cheese <- c(3-4i, 7-9i)
data.frame(apple, orange, chalk, cheese)  # ERROR !!!

items <- list(apple1 = apple, orange1= orange, chalk1=chalk, cheese1 = cheese) # Use "=" not "<-"
items # WORKS

 # subscript on list 
items[[3]]
# items[[i , j]] works for two dimenstion data 

# The first element of the thrid list element 
items[[3]][1]
items[3][1] # Not correct ! 

names(items)
items$cheese1


# lapply 
class(items)

mode(items)

length(items)

lapply(items, length) # the length of each element of the list without a loop 

lapply(items, class)

lapply(items, mean) #  mean not working for character class 

summary(items)

str(items)  # structure 

# manipulating and saving lists 
data <- read.csv("bowens.csv")
data1 <- data[1:10,2:3]
data1
rownames(data1) <- data[,1][1:10]
data1


sapply(data1, as.numeric)
sapply(1:10, function(i) which(data1[i, ] > 60)) # shows the columns which satisfy this standard 

spp <- sapply(1:10, function(i) which(data1[i, ] > 60))
spp
sapply(1:10, function(i) names(data1)[spp[[i]]])  # find the column names which satisfy the above standard 

sapply(1:2, function(j) rownames(data1) [data1[, j] > 60]) # get rownames for each column 

# write the file 
spplist <- sapply(1:2, function(j) rownames(data1)[data1[, j] > 60])
for (i in 1:2) {
  slist <- data.frame(spplist[[i]])
  names(slist) <- names(data1)[i]
  write.table(slist,paste(names(data1[i]), ".txt"),sep='') # will write file to the current working directory
}

?stack 
# Stacking vectors concatenates multiple vectors into a single vector along with 
# a factor indicating where each observation originated. Unstacking reverses this operation.

newframe <- stack(data1)
newframe
newframe <- data.frame(newframe, rep(rownames(data1), 2))
newframe
names(newframe) <- c("value", "direction", "location")
head(newframe)
write.table(newframe, "spplists.txt")

# Now it's easy to do analysis 
newframe[newframe$direction == "east" & (newframe$value > 50), ]
newframe$value > 50
newframe$direction == "east" 


# 2.12 Text, character strings and pattern matching 

a <- "abc"
b <- "123"

# Numbers can be coerced to characters, but non numeric characters cannot be coerced to numbers. 

as.numeric(a)
as.numeric("1")

as.numeric(b)

pets <- c('cat', "dog", "gerbil")
length(pets) # length of the vector 
nchar(pets) # length of each character strings 
class(pets) # NOT factors 
df <- data.frame(pets)
is.factor(df$pets) # IS factor 

letters
LETTERS

noquote(letters)  # suppress the double quotes 

# 2.12.1 Pasting character strings together 

c(a, b)

paste(a, b, sep='') # pasate together with no blank 
paste(a, b )  # there is a blank in between by default 

d <- c(a, b, "new")
e <- paste(d, " a long phrase")
e   # Three character strings are produced as d is a vector 
f <- paste(d, " a long phrase")

# 2.12.2 Extracting parts of strings 

phrase <- "the quick brown fox jumps over the lazy dog"
phrase

q <- character(20)
q   # 20 empty strings 

for (i in 1:20) q[i] <- substr(phrase, 1, i)
q
?substr  # substr(x, start, stop)


# 2.12.3 Counting things within strings 

nchar(phrase)  # blanks spaces are included 

strsplit(phrase, split = character(0))
character(0)
?strsplit  # Split the elements of a character vector x into substrings according to 
           # the matches to substring split within them.
strsplit(phrase, split = character(1))
strsplit(phrase, split = NULL)
# Argument split will be coerced to character, so you will see uses with split = NULL
# to mean split = character(0). 

table(strsplit(phrase, split = character(0))) # the first column is the blank space 

# Count the total number of words in phrase by counting the number of blank spaces 
words <- table(strsplit(phrase, split = character(0)))[1] + 1
words

strsplit(phrase, " ")
length(strsplit(phrase, " ")[[1]]) # strsplit returns a list, so need [[1]] to show the length of list elements
length(strsplit(phrase, " "))

strsplit(phrase, " ")[[1]]
lapply(strsplit(phrase, " "), nchar) # number of characters of each word 
table(lapply(strsplit(phrase, " "), nchar))

# reverse the words in the character vector 
lapply(strsplit(phrase, NULL), rev)

# paste them all back together 

sapply(lapply(strsplit(phrase, NULL), rev), paste) 
str(sapply(lapply(strsplit(phrase, NULL), rev), paste) )
sapply(lapply(strsplit(phrase, NULL), rev), paste, collapse = "")
str(sapply(lapply(strsplit(phrase, NULL), rev), paste, collapse = ""))
# collapse reduce back to a single character string 

strsplit(phrase, "the")
phrase
strsplit(phrase, "the")[[1]][2] # the second element 
nchar(strsplit(phrase, "the")[[1]][2] )


# 2.12.4 Upper and Lower case text  
toupper(phrase)  
tolower(toupper(phrase))



# 2.12.5 The "match" function and relational databases 
first <- c(sample(1:10, replace = TRUE))
first
second <- c(3,7,8)
match(first, second)
?match # match returns a vector of the positions of (first) matches 
       # of its first argument in its second.

subjects <- c(sample(LETTERS,size = 10))
subjects

suitable.patients <- c("E", "G", "T")

match(subjects, suitable.patients)

drug <- c("new", "conventional")
drug[ifelse(is.na(match(subjects, suitable.patients)), 2, 1)] # if suitable patients, then "new" 


# 2.12.6 Pattern matching 
wf <- read.table("worldfloras.txt", header = TRUE)
head(wf)
attach(wf)


# Match 1: "grep" searches a pattern in the first argument in the second argument 
as.vector(Country[grep("R", as.character(Country))]) # all countries containing a letter "R"

# Match 2: serach countries with upper case "R" as the first letter of names 
as.vector(Country[grep("^R", as.character(Country))])  # NO space between ^ and R 


# Match 3: search countries with upper case "R" as the first letter of their second or 
#          subsequent names 

as.vector(Country[grep(" R", as.character(Country))]) # search a blank space + R 


# Match 4: search countries with two or more words 
as.vector(Country[grep(" ", as.character(Country))]) # search blank space


# Match 5: names ending with "y"
as.vector(Country[grep("y$", as.character(Country))])


# Match 6: a range of upper case values 
as.vector(Country[grep("[C-E]", as.character(Country))])

# Match 7: a range of upper cases as the first letter 
as.vector(Country[grep("^[C-E]", as.character(Country))])


# Match 8: not ending with specific patterns 

as.vector(Country[-grep("[a-t]$", as.character(Country))])


# Match 9: not ending with specific patterns both lower and upper cases 
as.vector(Country[-grep("[A-T a-t]$", as.character(Country))])





# 2.12.7 .as : the "anything" character 
as.vector(Country[grep("^.y", as.character(Country))]) # . means ONE character of any kind 

as.vector(Country[grep("^..y", as.character(Country))]) # . means ONE character of any kind 

# names with "y" as the sixth letter 
as.vector(Country[grep("^.{5}y", as.character(Country))]) 

# names with 4 OR FEWER letters
as.vector(Country[grep("^.{,4}$", as.character(Country))]) # 

# names with 15 OR MORE characters 
as.vector(Country[grep("^.{15,}$", as.character(Country))]) # . means ONE character of any kind 



# 2.12.8 Substituting text within character strings 
# sub and gsub are used for "search and replace" operations 

text <- c("arm", "leg", "head", "foot", "hand", "hindleg", "elbow")
gsub("h", "H", text)
?gsub # sub and gsub perform replacement of the first and all matches respectively.

sub("o", "O", text)
gsub("o", "O", text)


# replace the first character of every string with uppoer case O 
gsub("^.", "O", text)


# capitalize the first character in each string 
gsub("(\\w*)(\\w*)", "\\U\\1\\L\\2", text, perl = TRUE) # perl = TRUE: use Perl-style regular expressions.

gsub("(\\w*)(\\w*)", "\\U\\1", text, perl = TRUE)

txt <- "a test of capitalizing"
gsub("(\\w)(\\w*)", "\\U\\1\\L\\2", txt, perl=TRUE)
gsub("\\b(\\w)",    "\\U\\1",       txt, perl=TRUE)

txt2 <- "useRs may fly into JFK or laGuardia"
gsub("(\\w)(\\w*)(\\w)", "\\U\\1\\E\\2\\U\\3", txt2, perl=TRUE)
sub("(\\w)(\\w*)(\\w)", "\\U\\1\\E\\2\\U\\3", txt2, perl=TRUE)
detach(wf)


# 2.12.9 Locations of a pattern within a vector using "regexpr" 
?regexpr
# grep, grepl, regexpr, gregexpr and regexec search for matches to argument 
# pattern within each element of a character vector: they differ in the 
# format of and amount of detail in the results. 
text
regexpr("o", text)  # return staring positions of the occurence within each element, 
                    # or -1 if not matched

grep("o", text) # return the subscript of elements which contain "o"

text[grep("o", text)]

?gregexpr 
# gregexpr returns a list of the same length as text each element of which is
# of the same form as the return value for regexpr, except that the starting 
# positions of every (disjoint) match are given.
gregexpr("o", text)
lapply(gregexpr("o", text),length)
unlist(lapply(gregexpr("o", text),length))
freq <- unlist(lapply(gregexpr("o", text),length))
present <- ifelse(regexpr("o", text) < 0, 0, 1)
present * freq # shows if the o is present in each element and how many times if it shows up 

?charmatch
# charmatch seeks matches for the elements of its first argument among those of its second
# An integer vector of the same length as the first argument is returned, giving the
# indices of the elements in table which matched, or nomatch.
charmatch("", "")                             # returns 1
charmatch("m",   c("mean", "median", "mode")) # returns 0
charmatch("med", c("mean", "median", "mode")) # returns 2
charmatch(c("med","mod"), c("mean", "median", "mode")) 


# 2.12.10 Using "%in%" and "which" 

stock <- c("car", "van")
requests <- c("truck", "suv", "van", "sports", "car", "car")

# use which 
which(requests %in% stock)
requests %in% stock
requests[which(requests %in% stock)]

# use match  
match(requests, stock)
stock[match(requests, stock)][!is.na(match(requests, stock))]
stock[!is.na(match(requests, stock))]
# Note: match has to be perfect in order to make it work 



# 2.12.11 More on pattern matching 
# Check "The R book " page 98 for more details or check the help file for the above functions. 

# ? the preceding item is optional and will be matched at most once 
# * the preceding item will be match ZERO or more times 
# + the preceding item will be matched ONE or more times 
# {n} the preceding item is matched exactly n times 
# {n,} the preceding item is matched n or more times 
# {,n} the preceding item is matched up to n times 
# {n, m} the preceding item is matched at least n times, but no more than m times 

# . matches a single character 
# ^ caret matches the empty string at the beginning of a LINE
# $ matches the empty string at the end of a LINE 
# \< matches the empty string at the beginning of a WORD 
# \>                               end 

text

grep('o',text)
grep('o{1}',text, value = TRUE)
# value: if FALSE, a vector containing the (integer) indices of the matches determined by grep is returned, 
# and if TRUE, a vector containing the matching elements themselves is returned.

grep('o{2}',text, value = TRUE)
grep('o{3}',text, value = TRUE)

grep('[[:alnum:]]{4,}',text, value = TRUE)
grep('[[:alnum:]]{5,}',text, value = TRUE)
grep('[[:alnum:]]{6,}',text, value = TRUE)
grep('[[:alnum:]]{7,}',text, value = TRUE)

# [:alnum:] alphanumeric characters 




# 2.12.12 "Perl" regular expressions 
# Perl = TRUE switches to the PCRE library which implements regular expression pattern matching 
# using the same syntax and semantics as Perl 5.6 or later 
?regexp




# 2.12.13 Stipping patterned text out of complex strings 


(entries <- c ("Trial 1 58 cervicornis (52 match)", 
               "Trial 2 60 terrestris (51 matched)", 
               "Trial 8 109 flavicollis (101 matches)"))
entries

# remove the material on numbers of matches including the brackets 
gsub("\\(.*\\)$","",entries) # remove everything between ( ) 
gsub(" *$", "", gsub("\\(.*\\)$","",entries)) # remove trailing blanks 


# strip out the materials in the brackets and ignore the brackets 

pos <- regexpr("\\(.*\\)$", entries) # the starting position of the first match for each element
pos
substring(entries, first = pos + 1, last = pos + attr(pos, "match.length")-2) 
attr(pos, "match.length")
?regexpr
# match.length includes the brackets themselves, need to subtract 2 if we don't want the brackets. 


entries <- c ("Trial 1 58 cervicornis (52 match)", "Trial 2 60
terrestris (51 matched)", "Trial 8 109 flavicollis (101 matches)")
entries






# 2.13 Dates and tiems in R 

Sys.time() 
str(Sys.time())
Sys.timezone()
Sys.Date()
Sys.getenv()


# convert the date to a single numeric repsresentation of the combined date adnd time in SECONDS. 
as.numeric(Sys.time())  # baseline is Jan 1st 1970 in seconds 

# R use POSIX(portable operating system interface) for representing times and dates 

class(Sys.time())
# [1] "POSIXct" "POSIXt"  : ct means continuous time (in seconds); ls means list time (in various 
# categorical descriptions of the time, including day of the week and so forth. )

?as.POSIXlt

time.list <- as.POSIXlt(Sys.time())
time.list
unlist(time.list) # return nine components of a list 
# isdst: is daylight saving time or not 
# mon: month of the year starting on jan 01 


# 2.13.1 Reading time data from files 
?mode
?class
?read.csv
data <- read.csv("dates.csv", header = TRUE)
data
class(data)
class(data$time.value)  # dates are converted to factors 

attach(data)
mode(time.value)
class(time.value) # factors are of mode numeric and class factor 
# R doesn't recognize the date for the monment 




# 2.13.2 The "strptime" function 
?strptime # Functions to convert between character representations and objects of classes
          # "POSIXlt" and "POSIXct" representing calendar dates and times.

# first convert time.value to the date and time format needed here 
n <- length(data$time.value)
n
time.value1 <- NULL
for (i in 1:n) time.value1[i] <- substr(as.character(data$time.value)[i], 2, 11)
time.value1
Rdate <- strptime(as.character(time.value1), "%Y-%m-%d") # use the same format "-" as in the data 
class(Rdate)
data <- cbind(data, Rdate)
Rdate$wday
data

z <- strptime("20/2/06 11:16:16.683", "%d/%m/%y %H:%M:%OS")
z
detach(data)



# A full list of format components : 
# %a  abbreviated weekday name 
# %A full weekday name
# %b  abbreviated month name
# %B  full month name 
# %c date and time , locale specific 
# %d day of the month as decimal number (1-31)
# %H hours of decimal number on 24 hour 
# %I hours of decimal number on 12 hour 
# %j day of the year as decimal number (0-366)
# %m month as decimal number (0-11)
# %M minute as decimal number (00-59)
# %p AM/PM indicator in the locale 
# %S second as decimal number (00-61) allowing for two leap seconds 
# %U week of the year (00-53) using the fist Sunday as day 1 of week 1 
# %w weekday as decimal number (0-6, Sunday is 0) 
# %W week of the year (00-53) using teh first Monday as day 1 of week 1 
# %x Date, locale specific 
# %X Time, locale specific 
# %Y Year with century
# %y year  without century 
# %Z time zone as a character string (output only)


?weekdays
y <- strptime("28/03/2017", format = "%d/%m/%Y")
str(y)
class(y)
weekdays(y)
weekdays(Sys.time())
# weekdays and months return a character vector of names in the locale in use.
months(y)

y$wday


# use abbreviated months
other.dates <- c("1jan99", "30jul05")
strptime(other.dates, "%d%b%y")



# 2.13.3 The "difftime" function 
?difftime
x <- difftime("2014-02-06", "2014-12-21") # first time minus the second time 
x
class(x)

x <- as.character(x)
x



# 2.13.4 calculations with dates and times 

# +, -, logical operator 

y2 <- "2015-10-22"
y1 <- "2018-10-22"
?as.Date
as.Date(y1) - as.Date(y2) # method 1 


y2 <- as.POSIXlt(y2) # method 2: first convert to POSIXlt object 
y1 <- as.POSIXlt(y1)
y1 - y2  



# 2.13.5 the "difftime" and "as.difftime"

difftime("2014-02-06", "2014-12-21")

as.numeric(difftime("2014-02-06", "2014-12-21")) # only the number 


# if only have times no dates , can use as.difftime to create appropriate objects 
t1 <- as.difftime("6:13:23")
t2 <- as.difftime("3:45:23")

t1 - t2



# another example 
times <- read.table("times.txt", header = TRUE)
head(times) 
attach(times)
# paste to a character string using colons as the separator 
paste(hrs, min, sec, sep = ":")

duration <- as.difftime(paste(hrs, min, sec, sep = ":"))
duration

tapply(duration, experiment, mean)




# 2.13.6 generating sequences of dates 


# by day 
seq(as.POSIXlt("2015-11-04"), as.POSIXlt("2015-11-10"), "1 day")


# by week 
seq(as.POSIXlt("2015-11-04"), as.POSIXlt("2015-12-10"), "1 week")

seq(as.POSIXlt("2015-11-04"), as.POSIXlt("2015-12-10"), "2 weeks")


# by month 
seq(as.POSIXlt("2015-11-04"), as.POSIXlt("2016-12-10"), "1 month")
seq(as.POSIXlt("2015-11-04"), as.POSIXlt("2016-12-10"), "3 months")


# by year 
seq(as.POSIXlt("2015-11-04"), as.POSIXlt("2019-12-10"), "1 year")
seq(as.POSIXlt("2015-11-04"), as.POSIXlt("2019-12-10"), "2 years")


# another method 1 
# if the by part in the seq is specified by a number , then the number is in seconds 
seq(as.POSIXlt("2015-11-04"), by = "month", length = 10)

#  another method 2 
# generate the dates to match the length of an existing vector using "along = "

results <- runif(16)

seq(as.POSIXlt("2015-11-04"), by = "month", along = results)
?seq

weekdays(seq(as.POSIXlt("2015-11-04"), by = "month", along = results))


# use of logical subscripts 

# the first 100 days of 2016 
y <- as.Date(1:100, origin = "2015-12-31")
y

x <- as.POSIXlt(y)
x
class(x)
str(x)


x[x$wday == 1] # select all days which are Mondays 


# extract only the first monday for each month of a year 
y <- as.POSIXlt(as.Date(1:365, origin = "2015-12-31"))
head(y)
data.frame(monday = y[y$wday == 1], month = y$mo[y$wday == 1]) # extract all mondays of the year 
wanted <- !duplicated(y$mo[y$wday == 1])
wanted
y[y$wday == 1][wanted]  # use wanted as the subscript to extract the first mondays of each month 

# or 
data.frame(monday = y[y$wday == 1], month = y$mo[y$wday == 1])[wanted, ]




# 2.13.7 calculating time differences between the rows of a dataframe 

class(duration)

# use subscripts to calculate differences 
duration
tmp <- duration[1:15]-duration[2:16]
tmp
diffs <- c(as.vector(tmp), NA)
diffs

# assgin it to the original data 
times$diffs <- diffs
times
times$diffs[8] <- NA
times




# 2.13.8 Regression using dates and times 

# no timereg.txt 
# simulate a fake data set  
survivors <- exp(10:1)
date <- seq(as.POSIXlt("2015-11-04"), by = "1 month", length = 10)
date <- as.character(date)

# data <- read.table("timereg.txt", header = TRUE)
date <- strptime(date, format="%Y-%m-%d")
date

class(date)
mode(date)  # a list 

par(mfrow=c(1,2))
plot(date, survivors)
plot(date, log(survivors))
dev.off()

model <- lm(log(survivors)~date, subset = (survivors > 0)) # ERROR: a list cannot be explanatory variable

# correct way 
date <- as.POSIXct(date) # or as.numeric 
date
model <- lm(log(survivors)~date, subset = (survivors > 0)) 

plot(date, log(survivors))
abline(model)


summary(model)  # the slope is small as it's based on seconds 
# date <- as.numeric(date)
# date

# Summary 
# POSIXlt gives a list containing separate vectors for the year, month day etc 
#          useful as a categorical explanatory variable
# POSIXct gives a vector containing the date and time expressed as a continuous variable
#          useful for regression models 

date <- as.POSIXlt(date)
date
date$mo
weekdays(date)





# 2.14 Environments 
globalenv()
# the variables you create are in the global environment


# 2.14.1 Using "with" rather than "attach"
?with
# with(data, expr, ...)

with(OrchardSprays, boxplot(decrease ~ treatment)) # call built in data sets 

library(MASS)
with(bacteria, tapply(y == "n", trt, sum))
with(mammals, plot(body, brain, log = "xy"))

reg.data <- read.table("regression.txt", header = TRUE)
with(reg.data, { model <- lm(growth ~ tannin) 
                      summary(model)}) # multiple statements are grouped by curly brackets 


data()  # check the built in data sets 
data(package = .packages(all.available = TRUE)) # all available data sets including installed packages



# Use "with" or "data = " argument 
# Avoid using "attach"






# 2.15 Writing R functions 

# 2.15.2 Median of a single sample 
# median is the middle value of the sorted values of a vector of numbers 
y <- seq(1, 20, 2)
sort(y)[ceiling(length(y)/2)]
median(y)

# function to find median 
med <- function(x){
  odd.even <- length(x) %% 2
  if(odd.even == 0) (sort(x)[length(x)/2] + sort(x)[length(x)/2 + 1])/2
  else sort(x)[ceiling(length(x)/2)]
}

med(y)


# or 

med <- function(x){
  ifelse(length(x) %% 2 == 0, (sort(x)[length(x)/2] + sort(x)[length(x)/2 + 1])/2, sort(x)[ceiling(length(x)/2)])
}

med(y)





# 2.15.6 Variance 

variance <- function(x) (sum((x - mean(x))^2)/(length(x) - 1))

variance(y)
var(y)


# 2.15.9 Deparsing: a graphics function for error bars 


# yv: height of bars 
# z: lengths of the error bars 
# nn: lables for the bars on the x axis 
error.bars <- function(yv, z, nn){
  xv <- barplot(yv, ylim = c(0, max(yv) + max(z)), names = nn, ylab = deparse(substitute(yv)))
  # deparse: Turn unevaluated expressions into character strings.
  g = (max(xv) - min(xv))/50
  for(i in 1:length(xv)){
    lines(c(xv[i], xv[i]), c(yv[i] + z[i], yv[i] - z[i]))
    lines(c(xv[i] - g, xv[i] + g), c(yv[i] + z[i], yv[i] + z[i]))
    lines(c(xv[i] - g, xv[i] + g), c(yv[i] - z[i], yv[i] - z[i]))
  }
}


comp <- read.table("competition.txt", header = TRUE)
head(comp, 2)
attach(comp)
se <- rep(28.75, 5)
labels <- as.character(levels(clipping))
ybar <- as.vector(tapply(biomass, clipping, mean))
error.bars(ybar, se, labels)
detach(comp)



# error bars on a scatterplot in both the x and y directions
xy.error.bars <- function(x, y, xbar, ybar) {
  # x, y : centers for the bars , xbar ybar are lengths 
  plot(x, y, xlim = c(min(x - xbar), max(x + xbar)), 
       ylim = c(min(y - ybar), max(y + ybar)), pch = 16)
  arrows(x, y - ybar, x, y + ybar, code = 3, angle = 90, length = 0.1)
  # 
  # angle : angle from the shaft of the arrow to the edge of the arrow head.
  # code	: integer code, determining kind of arrows to be drawn.
  # length : length of the edges of the arrow head (in inches).
  arrows(x - xbar, y, x + xbar, y, code = 3, angle = 90, length = 0.1)
}

x <- rnorm(5, 25, 5)
y <- rnorm(5, 100, 20)

xb <- runif(5) *5
yb <- runif(5)*20

xy.error.bars(x, y, xb, yb)






# 2.15.10 The switch function : do different things in different circumstances 


# Example to calculate any of the four kinds of means 
central <- function(y, measure){
  switch(measure, 
         Mean = mean(y),
         Geometric = exp(mean(log(y))),
         Harmonic = 1/mean(1/y),
         Median = median(y),
         stop("Measure not included") # In the case of no match, if there is a 
                                      #unnamed element of ... its value is returned.
         )
}

central(rnorm(100), "Harmonic") # quotation needed 
central(rnorm(100), "Test")

for(i in c(-1:3, 9))  print(switch(i, 1, 2 , 3, 4)) # numeric EXPR doesn't allow a default, always a NULL
switch(1, invisible(pi), pi)
switch(2, invisible(pi), pi)


# 2.5.11 The evaluation environment of a function 



# 2.15.14 Variable numbers of arguments(...)
# Triple dot is used in the argument list to specify that an arbitrary number of arguments are to be passed to 
#  the function. 

many.means <- function(...){
  data <- list(...)
  n <- length(data)
  means <- numeric(n)
  var <- numeric(n)
  for(i in 1:n){
    means[i] <- mean(data[[i]])
    var[i] <- var(data[[i]])
  }
  print(means)
  print(var)
  invisible(NULL)
}

x <- rnorm(100)
y <- rnorm(100)
many.means(x, y)
many.means() # Error in data[[i]] : subscript out of bounds


# use of invisible
f1 <- function(x) x
f2 <- function(x) invisible(x)
f1(1)  # prints
f2(1)  # does not print 


# R has a form of lazy evaluation of function arguments in which they are not evaluated until they are needed 



# 2.15.15 Returning values from a function 

# return one value : simply return the unassigned line 
parmax <- function(a, b){
  c <- pmax(a,b)
  median(c)
}


parmax(sample(1:10), sample(3:12))


# return multiple values 
parboth <- function(a, b){
  c <- pmax(a, b)
  d <- pmin(a, b)
  answer <- list(median(c), median(d))
  names(answer) <- c("median of max", "median of min")
  return(answer)
}
parboth(sample(1:10), sample(3:12))



# 2.15.16 Anonymous functions : mostly used with apply, tapply, sapply and lapply
(function(x, y){z <- 2*x + y; x+y+z})(0:7, 1)


# 2.15.17 Flexible handling of arguments to functions 
# The lazy evaluation in R makes it simple to deal with missing arguments 
plotx2 <- function(x, y = z^2){
  z <- 1:x
  plot(z, y, type="l")
}

par(mfrow= c(1,2))
plotx2(12)
plotx2(12, 1:12)
dev.off()




# 2.15.18 Structure of an object: str

data <- read.table("spplists.txt",header = T)
str(data)

# see more details about the levels of variables 
levels(data$direction)
levels(data$location)



# know the structure of a model objects 
reg <- read.table("tannin.txt", header = TRUE)
reg.model <- lm(growth ~ tannin, data = reg)
str(reg.model)


# know the structure of a generalized linear model 

data <- read.table("Trout.txt", header=TRUE)
str(data)
o=glm(cbind(tumor,total-tumor)~dose,
      family=binomial(link=logit),
      data=data)
summary(o) 
# str(o) # list of 30 



# 2.16 Writing from R to file 


# 1. save the current R sesscion and continue where left off 
# save(list = ls(all = TRUE), file = "test.session")
# load(file = "test.session")


# 2. save history 
history(Inf) # to see all of your lines of input code 

# or alternatively, 
# savehistory(file = "session")
# laodhistory(file = "session")



# 3. save graphics 
# pdf("filename")
# dev.off()


# 4. save data produced in R to a file 

numbers <- rnbinom(1000, size=1,mu = 1.2)
write(numbers, "numbers.txt", 1) # 1 is number of columns 

 # save a table or matrix 
xmat <- matrix(rpois(1000, 0.75), nrow = 100)
write.table(xmat, "table.txt", col.names = FALSE, row.names = FALSE)



