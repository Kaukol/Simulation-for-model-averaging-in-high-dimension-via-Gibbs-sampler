###Here is a simulation study for case 1 linear models

#ensure empty environment
rm(list = ls())

#library required
library(doParallel)
library(MASS)
library(IBGS)

#set up working directory
#setwd("/nfs/ms_home/home/ad/student.unimelb.edu.au/lizhongc/myGit/iterated-block-gibbs/ma")
setwd("/mnt/h/UbuntuRv2/Gibbs-sampler-algorithm/MA-new")
#parallel setting
registerDoParallel(12)
set.seed(101)

#simulation set up
#200 train samples and 200 test samples with 1000 predictors
n <- 400
p <- 1000

#correlation matrix M with rho_ij = rho^|i-j|
M <- diag(1,p)
rho <- 0
for (i in 1:p)
{
  for (j in 1:i)
  {
    M[j,i] <- rho^{i-j}
    M[i,j] <- M[j,i]
  }
}

### 100 times model averaging results
beta <- (1:15)/5

### 100 times model averaging results
#Ando, MMA, ZMA, AIC k=1, 1.5, BIC k=1, 1.5
MSE_BIC_k1.train <- vector()
MSE_BIC_k1.test  <- vector()

EP_BIC  <- vector()

j <- 0
while(j < 5){

  #data matrix
  x0 <- mvrnorm(n,rep(0,p),M)
  colnames(x0) <- 1:p

  y0 <- x0[,1:15]%*%beta + sin(rnorm(n)*pi) + cos(rnorm(n)*pi) + rnorm(n)

  #training
  x <- x0[1:(n/2),]
  y <- y0[1:(n/2)]

  #test
  x.t <- x0[(n/2+1):n,]
  y.t <- y0[(n/2+1):n]

  m.block <- BlockGibbsSampler(y, x, info = "exBIC", gamma = 0.5, family = gaussian())

  w  <- m.block$c.models$weights

  #fitted
  MFit  <- vector()
  MPred <- vector()
  MRank <- vector()
  for(i in 1:length(w)){
    fit   <- m.block$c.models$models[[i]]
    MFit  <- cbind(MFit, fitted.values(fit))
    v0.s  <- as.numeric(colnames(fit$model)[-1])
    MPred <- cbind(MPred, as.matrix(cbind(1,x.t[,v0.s]))%*% fit$coefficients)
    MRank <- c(MRank, fit$rank-1)
  }

  EFit1   <- MFit %*% w
  EPred.1 <- MPred %*% w

  MSE.g1 <- sum((EFit1 - y)^2)/(n/2)
  MSE.t1 <- sum((EPred.1 - y.t)^2)/(n/2)
  MSE_BIC_k1.train <- c(MSE_BIC_k1.train, MSE.g1)
  MSE_BIC_k1.test  <- c(MSE_BIC_k1.test, MSE.t1)

  EP_BIC <- c(EP_BIC, sum(MRank * w))

  j <- j+1

  print(c(j,"loop"))
  save.image("simu_case1_ibgs_exbic_0.RData")
}

save.image("simu_case1_ibgs_exbic_0.RData")
