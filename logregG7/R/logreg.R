library(ggplot2)
library(dplyr)

#Implementing the sigmoid function
sigmoid <- function(g){
  1/(1+exp(-g))
}

#Implementing the estimator
est <- function(theta, X, y) {
  p <- sigmoid(X %*% theta)
  b_hat <- ((-y)%*%log(p)-(1-y)%*%log(1-p))
  b_hat
}

#initial values
#X <- cbind(c(1,2,3,4,5), c(6,7,8,9,10))
#y <- c(0,1,0,1,0)
init_val <- function(X,y){
  solve(t(X)%*%X)%*%(t(X)%*%y)
}

#Implementing gradient function
gradient <- function(theta, X, y){
  m <- length(y)
  p <- sigmoid(X%*%theta)
  grad <- (t(X)%*%(p - y))/m
  grad
}

