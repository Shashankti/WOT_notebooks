############################################################

#### Run DDM simulations and store results

# Author - David Hughes

############################################################



rm(list=ls())

#setwd("~/DDMsims")
source('ddm_functions_boot.R')

#SIMULATION OF TEST STATISTIC

#model parameters
delta <- 0.5
sig   <- 1
FUN.b <- function(t){
  
  #b <- 2*exp(-1.5*t) + 0.5
  b <- rep(1,length(t))
  return(b)
  
}

#simulation parameters
int   <- 1000
step  <- 500
tmax  <- 10^10
ndat      <- 1000
nsim.bm   <- 1000
nsim.stat <- 500
nboot     <- 250

#estimation tuning parameters
num.knots <- 0
deg       <- 0
gmin      <- 0
J         <- 5

params <- c(num.knots, deg, gmin, int, step, tmax, J)

#SIMULATION
set.seed(20390)

#######ALL SIMS
nk.vec <- c(0,0)
d.vec  <- c(0,1)

#DGP is DDM
dgp <- "ddm"

for(i in 1:2){
  set.seed(20390)
  params[c(1,2)] <- c(nk.vec[i],d.vec[i]) 
    
  out <- pbreplicate(nsim.stat,
                     sim_test_simple(dgp,FUN.b, delta, sig, params, ndat, nsim.bm, nboot))
  
  filename <- paste0('bootstat_n1000j5knots',nk.vec[i],
                     'sim100deg',d.vec[i],'sb100_ddm.Rdata',sep='')
  
  
  save(out,file=filename)

}

#DGP is Possion model (Power calculations)
dgp <- "poisson"
delta <- 1.043878 #matched to delta=0.5 and B=1
sig <- 0.7310586

params.vec <- params
for(i in 1:4){
  params.vec[c(1,2)] <- c(nk.vec[i],d.vec[i]) 
  
  out <- pbreplicate(nsim.stat,
                     sim_test_simple(dgp,FUN.b, delta, sig, params.vec, ndat, nsim.bm, nboot))
  
  filename <- paste0('bootstat_n1000j5knots',nk.vec[i],
                     'sim100deg',d.vec[i],'sb100_poisson.Rdata',sep='')
  
  
  save(out,file=filename)
  
}



#Simulate drift statistic with bootstrap t-statistic

dgp <- "ddm"
params.vec <- params
out <- pbreplicate(nsim.stat,
                   sim_drift_stat(dgp,FUN.b, delta, sig, params.vec, ndat, nsim.bm, nboot))

filename <- paste0('driftstat_n1000j5knots2',
                   'sim100deg1','sb100_cons.Rdata',sep='')


save(out,file=filename)

#Just simulate drift, no bootstrap variance estimate
delta <- 0.5
sig   <- 1
b_fun <- function(t){
  b <- rep(1,length(t))
  return(b)
}

int   <- 1000
params <- c(num.knots,deg,gmin,int,step,tmax,J)

delta_sim <- function(delta, sig, b_fun, params){
  
  dat.new  <- replicate(ndat, sim_txy_true(delta, b_fun, sig, params))
  tau <- dat.new[2,]
  y   <- dat.new[1,]
  g.par <- c(mean(tau), mean(tau)^3/var(tau))
  par.hat <- par_est(dat.new, params)
  return(par.hat$drift)
  
}

knots.vec <- c(0,0,1,2)
deg.vec <- c(0,1,1,1)
delta.hat <- matrix(NA,nrow=nsim,ncol=4)

for(i in 1:4){
  set.seed(20390)
  params[1] <- knots.vec[i]
  params[2] <- deg.vec[i]
  delta.hat[,i] <- pbreplicate(nsim, delta_sim(delta, sig, b_fun, params))
}


d.mean <- apply(delta.hat,2,mean)
d.median <- apply(delta.hat,2,median)
d.sd <- apply(delta.hat,2,sd)
d.mad <- apply(delta.hat,2,function(x){mean(abs(x-0.5))})

d.mean
d.median
d.mad
d.sd