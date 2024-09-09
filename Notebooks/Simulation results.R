############################################################

#### Analyse results from DDM simulations

# Author - David Hughes

############################################################



# Power calculations for exponential model alternative
rm(list=ls())
setwd("~/Dropbox (MIT)/RA Work/Whitney - DDM")
source('ddm_functions_boot.R')
library(pbapply)
library(latex2exp)


##### Test statistic simulations
file <- paste("/Users/davidhughes/Dropbox (MIT)/RA Work/Whitney - DDM/Statistic sims/",
              "bootstat_n250j5knots0sim100deg0sb100_ddm5_3",
              ".Rdata", sep="")
load(file=file)

#Print out rejection rates
sum(out>qchisq(0.8,3))/length(out)
sum(out>qchisq(0.9,3))/length(out)
sum(out>qchisq(0.95,3))/length(out)
sum(out>qchisq(0.99,3))/length(out)







####  Match moments of Poisson model to DDM and compare CDFs

delta <- 0.5
sig   <- 1
b_fun <- function(t){
  b <- rep(1,length(t))
  return(b)
}

int   <- 1000
step  <- 500
tmax  <- 10^10
ndat      <- 1000
nsim.est   <- 1000

data <- pbreplicate(10000, sim_txy_step(delta, sig, b_fun, int, tmax, step))
y.ddm   <- data[1,]
tau.ddm <- data[2,]

#mean FPT
mu.ddm <- mean(tau.ddm)
#choice prob
p.ddm <- exp(2*delta*1)/(1+exp(2*delta*1))


#### Match moments of exponential model
a.exp <- log(p.ddm/mu.ddm)
b.exp <- a.exp - 2*delta*1

#Generate exponential model data
gen_data_exp <- function(num,lambda, p){
  
  tau <- rexp(num,lambda)
  y   <- rbinom(num,1,p)
  
  return(cbind(y,tau))
  
}

dat.exp <- gen_data_exp(100000,1/mu.ddm,p.ddm)
y.exp <- dat.exp[,1]
tau.exp <- dat.exp[,2]

#Check moments of simulated data
mean(tau.ddm)
mean(tau.exp)
mean(y.ddm)
mean(y.exp)


#### Compare CDFs for tau in each model
cdf.ddm <- sort(tau.ddm)
cdf.exp <- sort(tau.exp)
ind     <- (1:length(cdf.ddm))/length(cdf.ddm)

plot(cdf.ddm,ind,type='l',col='blue',xlim=c(0,6))
lines(cdf.exp,ind,col='red')

legend("bottomright",bty="n",c("DDM","Poisson"),col=c('blue','red'),lty=1)

dens.ddm <- density(tau.ddm)
dens.exp <- density(tau.exp)
plot(dens.ddm$x,dens.ddm$y,col='blue',xlim=c(0,6),type='l')
lines(dens.exp$x,dens.exp$y,col='red')
legend("topright",bty="n",c("DDM","Poisson"),col=c('blue','red'),lty=1)