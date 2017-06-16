setwd("~/Dropbox/R-Statistical-Analysis/The-R-Book/therbookdata")

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
for (i in 1:10) text(i,i,LETTERS[i],x=(20*i),col='red')
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
