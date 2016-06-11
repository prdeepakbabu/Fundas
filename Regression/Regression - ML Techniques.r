
#lets read a dataset for training a machine learning model along with all R packages/libraries loaded
rm(list=ls())
library(sqldf)
library(nnet)
setwd("/home/deepak/Documents/Projects/Product")
inp = read.csv("dataset_coverage.csv")

str(inp)

options(repr.plot.width=4, repr.plot.height=3)
plot(inp$x,inp$y,type='p',lwd=0,col='blue',cex.lab=0.75, cex.axis=0.7,xlab='x',ylab='y',main='Training Dataset',
     cex.main=0.75)
lines(inp$x,inp$y,col='blue')
grid()

ols = lm(y ~ x,inp)
summary(ols)

x = seq(0,25,0.1)
y = 0.660799 + 0.011049 * x
options(repr.plot.width=8, repr.plot.height=3)
par(mfrow=c(1,2))
res = resid(ols)
plot(x,y,type='p',lwd=0,col='blue',cex.lab=0.75, cex.axis=0.7,xlab='x',ylab='predicted y=f(x)',
     cex.main=0.75, main = 'Actual vs. Predicted')
lines(x,y,)
lines(inp$x,inp$y,col='blue')
grid()
x=seq(0,23,1)
plot(x,res,col='blue',pch=18,cex.lab=0.75, cex.axis=0.7,xlab='x',ylab='residuals',
     cex.main=0.75,main='Residuals')
abline(0, 0)  

inp$x1 = inp$x*inp$x
inp$x2 = inp$x*inp$x*inp$x
ols1 = lm(y ~ x + x1 + x2,inp)
summary(ols1)

x = seq(0,23,0.1)
y = 0.660799 + 0.011049 * x
y1 = 5.413e-01 + 5.640e-02 * x -3.586e-03 * x^2 + 7.690e-05 * x^3
predicted_ols = y1
options(repr.plot.width=8, repr.plot.height=3)
par(mfrow=c(1,2))
res = resid(ols)
res1 = resid(ols1)
plot(x,y,type='p',lwd=0,col='blue',cex.lab=0.75, cex.axis=0.7,xlab='x',ylab='predicted y=f(x)',
cex.main=0.75)
lines(x,y,)
lines(inp$x,inp$y,col='red')
lines(x,y1,col='blue')
grid()
x=seq(0,23,1)
plot(x,res1,col='blue',pch=18,cex.lab=0.75, cex.axis=0.7,xlab='x',ylab='residuals',
cex.main=0.75)
abline(0, 0)  

#Lets Calculate RMSE (root mean square error - so we can compare other models using this metric). There are 
#other error metrics one could look at, depeneding on the problem at hand. Check out kaggle for how they measure 
#error for various competitions
actual = inp$y
x = inp$x
y1 = 5.413e-01 + 5.640e-02 * x -3.586e-03 * x^2 + 7.690e-05 * x^3
predicted = y1
RMSE = (mean((actual - predicted)^2))^0.5 / nrow(inp)
RMSE

source("gradientdescent.R")

# Setting up for gradient descent
X <- matrix(c(inp$x,inp$x*inp$x), nrow=nrow(inp), byrow=FALSE)
y <- as.vector(inp$y)
f <- function(X,y,b) {
   (1/2)*norm(y-X%*%b,"F")^{2}
}
grad_f <- function(X,y,b) {
   t(X)%*%(X%*%b - y)
}
simple_ex <- gdescent(f,grad_f,X,y,alpha=0.01,iter=120000)

library(ggplot2)
#options(repr.plot.width=12, repr.plot.height=3)
#par(mfrow=c(1,3))
#plot_loss(simple_ex)
#plot_iterates(simple_ex)
#plot_gradient(simple_ex)
x = seq(0,23,0.1)
predicted = 0.5821183 +(0.0325076650*x) -(0.0009329719*(x*x))
predicted_gd = predicted
options(repr.plot.width=4, repr.plot.height=3)
plot(x,predicted,type='p',lwd=0,col='blue',cex.lab=0.75, cex.axis=0.7,xlab='x',ylab='predicted y=f(x)',
cex.main=0.75)
lines(inp$x,inp$y,col='red')
lines(x,predicted,col= "blue")
grid()

predicted = 0.5821183 +(0.0325076650*inp$x) -(0.0009329719*(inp$x*inp$x))
actual = inp$y
RMSE = (mean((actual - predicted)^2))^0.5 / nrow(inp)
RMSE

library(nnet)
library(neuralnet)

nn <- neuralnet(inp$y ~ inp$x,train,hidden=c(3,3),linear.output=T)

plot.nn(nn,file="nn.png")
dev.off()

nn
#jj <- readJPEG("nnet_vis_1.jpg",native=TRUE)
#plot(0:1,0:1,type="n",ann=FALSE,axes=FALSE)
#rasterImage(jj,0,0,1,1)

#RMSE
actual = inp$y
predicted = compute(nn,inp$x)$net.result
RMSE = (mean((actual - predicted)^2))^0.5 / nrow(inp)
RMSE

x = seq(0,23,0.1)
predicted = compute(nn,x)$net.result
predicted_nn = predicted
options(repr.plot.width=4, repr.plot.height=3)
plot(x,predicted,type='p',lwd=0,col='blue',cex.lab=0.75, cex.axis=0.7,xlab='x',ylab='predicted y=f(x)',
cex.main=0.75)
lines(inp$x,inp$y,col='red')
lines(x,predicted,col= "blue")
grid()

x = seq(0,23,0.1)
options(repr.plot.width=8, repr.plot.height=4)
plot(x,predicted_ols,type='o',lwd=0,col='blue',cex.lab=0.75, cex.axis=0.7,xlab='x',ylab='predicted y=f(x)',main="Curve Fitting - OLS, GD & NN"
,cex.main=0.75)
lines(inp$x,inp$y,col='black')
lines(x,predicted_ols,col= "blue")
lines(x,predicted_gd,col= "red")
lines(x,predicted_nn,col= "green")
legend('bottom',cex=0.75,c("Actual","Predicted - OLS","Predicted - GD","Predicted - NN"),horiz=TRUE,bg='lightblue',pt.cex=0.75,bty='n',lty=c(1,1), lwd=c(2.5,2.5),col=c("black","blue","red","green")) 


