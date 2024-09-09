############################################################

#### Functions for DDM simulations

# Author - David Hughes

############################################################

library(pbapply)
library(statmod)
library(splines)
library(boot)

#SIMULATION FUNCTIONS

# This function generates a FPT and choice pair. Inputs are:
# delta - the drift of the brownian motion
# sig   - the standard deviation of the brownian motion
# b     - the boundary (a constant)
# int   - the number of steps per unit of time in the brownian motion
# tmax  - an upper bound for the crossing time
# step  - the number of steps to generate before checking for a crossing

sim_txy_true <- function(delta, FUN.b, sig, params){
  
  int  <- params[4]
  step <- params[5]
  smax <- params[6]
  
  xlast <- 0   # brownian motion
  y     <- NA  # choice indicator (1-top boundary, 0-bottom boundary)
  s     <- 0   # time 
  
  while(s <= smax){
    
    x     <- xlast + (1:step)*delta/int + cumsum(sqrt(1/int)*sig*rnorm(step,0,1))
    t     <- (s + (1:step))/int   #return time in orginal units
    b     <- FUN.b(t)
    
    cross <- (abs(x) >= b) #check for crossings
    if(sum(cross)>0){
      pos <- min(which(cross==TRUE)) #if a crossing occurred, find first instance
      tau <- (s + pos)/int   #return time in orginal units
      y   <- (x[pos] >= b[pos])
      return(c(y,tau))
    }
    
    s     <- s + step #else update time counter and xlast
    xlast <- x[step]
  }
  
  return(c(NA,NA))
  
}

# This function generates a FPT and choice pair, given an estimated boundary.
sim_txy_est <- function(delta, sig, b.hat, params, tmax){
  
  int  <- params[4]
  step <- params[5]
  smax <- tmax*int - step
  
  xlast <- 0   # brownian motion
  y     <- NA  # choice indicator (1-top boundary, 0-bottom boundary)
  s     <- 0   # time 
  
  while(s <= smax){
    
    x     <- xlast + (1:step)*delta/int + cumsum(sqrt(1/int)*sig*rnorm(step,0,1))
    t     <- s + (1:step) 
    b     <- b.hat[t]
    
    cross <- (abs(x) >= b) #check for crossings
    if(sum(cross)>0){
      pos <- min(which(cross==TRUE)) #if a crossing occurred, find first instance
      tau <- (s + pos)/int   #return time in orginal units
      y   <- (x[pos] >= b[pos])
      return(c(y,tau))
    }
    
    s     <- s + step #else update time counter and xlast
    xlast <- x[step]
  }
  
  return(c(NA,NA))
  
}

#Generates data from the Poisson model
gen_data_exp <- function(num,lambda, p){
  
  tau <- rexp(num,lambda)
  y   <- rbinom(num,1,p)
  
  return(t(cbind(y,tau)))
  
}

#-----------ESTIMATION FUNCTIONS

# Generates a b-spline basis for a given dataset and chosen number of knots
# Knots are placed at evenly spaced quantiles of the data

gen_spline <- function(G,num.knots,deg,spline.knots){
  
  #Generate spline functions
  if(num.knots==0){
    basis <- bs(G,degree=deg, intercept=TRUE, knots=NULL)
  } else{
    basis <- bs(G,degree=deg, intercept=TRUE,knots=spline.knots)
  }
  
  return(basis)
  
}

gen_Q <- function(G, basis, gmin){
  
  Q           <- t(predict(basis,G-gmin))
  const       <- ifelse(predict(basis,0)<0.5,0,1)
  Q[,G<=gmin] <- const
  return(t(Q))
  
}


# Function estimates the propensity score from y and tau
est_propensity <- function(y,tau,g.par,basis,gmin){
  
  G <- pinvgauss(tau,g.par[1],g.par[2])
  
  if(sum(is.na(basis))){
    phat <- lm(y~1)$fitted
  } else{
    X    <- gen_Q(G, basis, gmin)
    phat <- lm(y~X-1)$fitted
  }
  
  phat[phat>=0.99] <- 0.99
  phat[phat<=0.01] <- 0.01
  return(phat)
  
}

#computes boundary estimate from propensity function coefficients
betas_to_boundary <- function(t, beta, delta, g.par, basis, gmin){
  
  if(sum(is.na(basis))){
    phat <- outer(rep(1,length(t)),beta)
  } else{
    G     <- pinvgauss(t,g.par[1],g.par[2])
    X     <- gen_Q(G, basis, gmin)
    phat  <- X%*%beta
  }
  
  phat[phat>=0.99] <- 0.99
  phat[phat<=0.01] <- 0.01
  
  bhat    <- 0.5*log(phat/(1-phat))/delta
  
  return(bhat)
  
}


# Estimates the drift parameter from estimated propensities and crossing times
est_delta <- function(phat, tau){
  
  I_hat <- phat*log(phat/(1-phat)) + (1-phat)*log((1-phat)/phat)
  delta <- sqrt(0.5*mean(I_hat)/mean(tau))
  return(delta)
  
}

#BOOTSTRAP STATISTIC FUNCTIONS

#Compute proportion of data in each bin
test_funs <- function(t, J, g.par, bins){
  
  G <- bins
  H <- c(0,G,max(t)+1)
  
  x    <- matrix(t,nrow=length(t),ncol=J)
  ints <- matrix(rep(H,each=length(t)),ncol=(J+1)) 
  out  <- as.matrix((x<ints[,2:(J+1)]) - (x<ints[,1:J])) 
  
  return(out[,2:(J-1)])
 
}

#Estimate drift and boundary together
par_est <- function(dat, params, knots){
  
  tau <- dat[2,]
  y   <- dat[1,]
  g.par <- c(mean(tau), mean(tau)^3/var(tau))
  
  num.knots <- params[1]
  deg       <- params[2]
  gmin      <- params[3]
  
  #estimate boundary coefficients
  G     <- pinvgauss(tau,g.par[1],g.par[2])
  if(deg==0){
    basis <- NA
    beta <- lm(y~1)$coefficients
  } else{
    basis <- gen_spline(G,num.knots,deg,knots)
    X     <- gen_Q(G, basis, gmin)
    beta  <- lm(y~X-1)$coefficients
  }
  K     <- length(beta)
  
  #estimate drift
  phat <- est_propensity(y,tau,g.par,basis,gmin)
  delta.hat <- est_delta(phat,tau)
  
  return(list("propensity"=phat, "betas"= beta, "drift"=delta.hat,  "basis"=basis))
  
}

#estimate full model, simulate new data, and compute difference in CDF proportions
dist_dif <- function(boot, data, params, sig, nsim, bins, knots){
  
  if(boot==TRUE){
    smpl <- sample(1:ncol(data), ncol(data), replace=TRUE)
    dat.new <- data[,smpl]
  } else {
    dat.new <- data
  }
  
  tau <- dat.new[2,]
  y   <- dat.new[1,]
  g.par <- c(mean(tau), mean(tau)^3/var(tau))
  tmax <- 20
  
  #estimate boundary and drift
  par.hat <- par_est(dat.new, params, knots)
  int     <- params[4]
  tvec    <- seq(1/int, tmax, 10/int)
  b.hat   <- betas_to_boundary(tvec, par.hat$betas, par.hat$drift, g.par, par.hat$basis, params[3])
  b.hat   <- rep(b.hat, each=10)
  
  #simulate from model
  sim.out  <- replicate(nsim, sim_txy_est(par.hat$drift, sig, b.hat, params, tmax))
  sim.tau  <- sim.out[2,]
  
  #test vector
  J    <- params[7]
  m1   <- colMeans(test_funs(tau,J,g.par,bins), na.rm=TRUE)
  m2   <- colMeans(test_funs(sim.tau,J,g.par,bins), na.rm=TRUE)
  
  return(m1-m2)
  
}

#Bootstrap the difference stat to compute variance and create statistic
test_stat <- function(b.sim, data, params, sig, nsim, nboot){
  
  if(b.sim==TRUE){
    smpl <- sample(1:ncol(data), ncol(data), replace=TRUE)
    dat.new <- data[,smpl]
  } else {
    dat.new <- data
  }
  
  J      <- params[7]
  bins   <- quantile(data[2,], seq(1,J-1,1)/J,na.rm=TRUE)
  g.par <- c(mean(data[2,]), mean(data[2,])^3/var(data[2,]))
  G     <- pinvgauss(data[2,],g.par[1],g.par[2])
  spline.knots <- quantile(G,(1:params[1])/(params[1]+1))
  
  mdif   <- dist_dif(boot=FALSE, dat.new, params, sig, nsim, bins, spline.knots)
  mdif.b <- replicate(nboot, dist_dif(boot=TRUE, dat.new,params, sig, nsim, bins, spline.knots))
  V      <- var(t(mdif.b))
  stat   <- t(mdif)%*%solve(V)%*%mdif
  
  return(stat)
  
}


#SIMULATE TEST STATISTIC

#Simulate statistic, as well as bootstrap critical values for the statistic
sim_test <- function(dgp, FUN.b, delta, sig, params, ndat, nsim, nboot){
  
  #generate data set
  if(dgp=="ddm"){
    data  <- replicate(ndat, sim_txy_true(delta, FUN.b, sig, params))
  } else if(dgp=="poisson"){
    data  <- gen_data_exp(ndat,delta,sig)
  }
  
  #compute test statistic
  stat   <- test_stat(b.sim=FALSE, data, params, sig, nsim, nboot)
  
  #bootstrap cricitcal values
  stat.b <- replicate(nboot, test_stat(b.sim=TRUE,data,params,sig,nsim,nboot))
  cv     <- quantile(stat.b, c(0.99, 0.95, 0.90, 0.80))
  rej    <- (rep(stat,4)>cv)
  
  #return(stat)
  return(rej)
  
}

#Simulate statistic only
sim_test_simple <- function(dgp, FUN.b, delta, sig, params, ndat, nsim, nboot){
  
  #generate data set
  if(dgp=="ddm"){
    data  <- replicate(ndat, sim_txy_true(delta, FUN.b, sig, params))
  } else if(dgp=="poisson"){
    data  <- gen_data_exp(ndat,delta,sig)
  }
  
  #compute test statistic
  stat   <- test_stat(b.sim=FALSE, data, params, sig, nsim, nboot)

  return(stat)
  
}

#Just simulate the drift statistic
sim_drift_stat <- function(dgp, FUN.b, delta, sig, params, ndat, nsim, nboot){
  
  #generate data set
  if(dgp=="ddm"){
    data  <- replicate(ndat, sim_txy_true(delta, FUN.b, sig, params))
  } else if(dgp=="poisson"){
    data  <- gen_data_exp(ndat,delta,sig)
  }
  
  #estimate boundary and drift
  drift.hat <- drift_stat(boot=FALSE,data, params, sig, nsim)
  
  drift.b <- replicate(nboot, drift_stat(boot=TRUE, data, params, sig, nsim))
  sd       <- sd(t(drift.b))
  return(c(drift.hat,sd))
  
}

drift_stat <- function(boot, data, params, sig, nsim){
  
  if(boot==TRUE){
    smpl <- sample(1:ncol(data), ncol(data), replace=TRUE)
    dat.new <- data[,smpl]
  } else {
    dat.new <- data
  }
  
  tau <- dat.new[2,]
  y   <- dat.new[1,]
  g.par <- c(mean(tau), mean(tau)^3/var(tau))
  tmax <- 20
  
  #estimate boundary and drift
  par.hat <- par_est(dat.new, params)
  
  return(par.hat$drift)
  
}