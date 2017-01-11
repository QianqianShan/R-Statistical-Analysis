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
  "             first: {f * minute}(x) " == {f * minute}(x)), adj = 0)
text(1, 9.0, expression(
  "     second: {f * second}(x) "        == {f * second}(x)), adj = 0)


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

