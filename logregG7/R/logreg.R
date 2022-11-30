library(ggplot2)
library(dplyr)

#Implementing the pi function
pi <- function(g){
  1/(1+exp(-g))
}

sigmoid <- function(x, beta){
  as.matrix(x)
  return(1/(1 + exp(- (x%*%beta))))
}

#Implementing the loss function
est <- function(theta, X, y) {
  m <- length(y)
  p <- sigmoid(X, theta)
  b_hat <- (t(-y)%*%log(p)-t(1-y)%*%log(1-p))
  b_hat
}

#initial values
X <- matrix(rnorm(10), nrow = 5) #cbind(c(1,2,3,4,5), c(6,7,8,9,10))
y <- c(0,1,0,1,0)

init_val <- function(X,y){
  solve(t(X)%*%X)%*%(t(X)%*%y)
}


#Implementing the logistic function
logisticReg <- function(X, y){
  #remove NA rows
  X <- na.omit(X)
  y <- na.omit(y)
  #converting y to matrix
  y <- as.matrix(y)
  #use the optim function to perform gradient descent
  costOpti <- optim(init_val(X,y), fn = est, X = X, y = y)
  #return coefficients
  return(c(costOpti$par, costOpti$value))
}


#Implemeting bootstrap

bootstrap_confi <- function(X, y, b=20, alpha = 0.05){
  n <- dim(X)[1]
  p <- dim(X)[2]
  #concat <- cbind(x,y)
  beta <- matrix(nrow = b, ncol = p+1)
  for (i in 1:b) {
    draw <- sample(1:n, n, replace = TRUE)
    boot_x <- X[draw,]
    boot_y <- y[draw]

    beta[i,] <- logisticReg(boot_x, boot_y)
  }
  beta_mean <- apply(beta, 2, mean)
  beta_std_dev <- apply(beta, 2, sd)
  lower_bound <- beta_mean - qnorm(1 - alpha/2)*beta_std_dev
  upper_bound <- beta_mean + qnorm(1 - alpha/2)*beta_std_dev
  confi_interval <- cbind(beta_mean,lower_bound, upper_bound)
  return(confi_interval)
}



#implementing the plot
plot.lsoptim <- function(object, ...) {

  hold <- object
  response <- hold$response
  predictors <- hold$predictors
  beta_hat <- hold$beta_hat

  dm <- ncol(predictors)

  par(mfrow = c(1, dm - 1))

  for(i in 2:dm) {

    plot(predictors[, i], (response - predictors[, -c(1, i)]%*%beta_hat[-c(1, i)]), xlab = paste0("x",i), ylab = "response")
    abline(beta_hat[1], beta_hat[i], col = "red")

  }
