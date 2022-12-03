#Importing needed libraries
library(caret)

#' @importFrom graphics lines
#' @importFrom stats optim qnorm sd
#' @importFrom utils tail
NULL


#Generating the sigmoid function
sigmoid <- function(x){
  return(1/(1 + exp(- x)))
}

#Simulating random sample of x and y
set.seed(233)
n <- 1000
p <- 1
rx <- rnorm(n*p, 0, 4)
x <- matrix(rx,ncol=p)
beta <- rpois(p+1,1.56)
y <- as.vector(round(sigmoid(beta%*%t(cbind(1,x))+rnorm(n, 0, 1))))


#' @title Cost function
#' @description This function computes the cost function that is to be optimized
#' @param beta A \code{vector}
#' @param X A \code{matrix} that holds the independent variables
#' @param y A \code{vector} representing the dependent variable
#' @return A \code{matrix} containing the following objects
#' @author Manisha Parajuli, Favour Onyido, Festus Attah
#' @export
cost <- function(beta, X, y){
  m <- length(y) #number of rows  of the training data
  h <- sigmoid(X %*% beta)
  J <- (t(-y)%*%log(h)-t(1-y)%*%log(1-h))/m #log likelihood function
  return(J)
}

#' @title Gradient function
#' @description This function computes the gradient function that is to be optimized
#' @param beta A 2 digit \code{vector}
#' @param X A \code{matrix} that holds the independent variables
#' @param y A \code{vector} representing the dependent variable
#' @return A \code{matrix} containing the following objects
#' @author Manisha Parajuli, Favour Onyido, Festus Attah
#' @export
grad <- function(beta, X, y){
  m <- length(y)

  h <- sigmoid(X%*%beta)
  grad <- (t(X)%*%(h - y))/m
  return(grad)
}

#' @title Optim log
#' @description This function minimizes the cost function computed
#' @param x A \code{matrix} that holds the independent variables
#' @param y A \code{vector} representing the dependent variable
#' @return A \code{matrix} containing the intercept and coefficient
#' @author Manisha Parajuli, Favour Onyido, Festus Attah
#' @export
Optim_log <- function(x, y)
{
  if(!is.matrix(x)){
    x = as.matrix(x)
  }
  m <- dim(x)[1]
  intercept <- rep(1, m)
  x = cbind(intercept, x)
  x_inverse <- solve(t(x)%*%x)
  beta <- x_inverse%*%t(x)%*%y
  costOpti <- optim(beta, fn = cost, gr = grad, X = x, y = y)

  return(costOpti$par)
}

#Bootstrap confidence interval
#' @title Bootstrap confidence interval
#' @description This function uses the bootstrap resampling method with replacement to compute the confidence interval
#' @param x A \code{matrix} that holds the independent variables
#' @param y A \code{vector} representing the dependent variable
#' @param b A \code{numeric} representing the number of bootstrap replications
#' @param alpha A \code{numeric} representing the confidence level
#' @return A \code{matrix} containing the following objects:
#' \describe{
#'  \item{beta_mean}{This \code{numeric} represents the mean of the parameter estimates}
#'  \item{lower_bound}{This \code{numeric} represents 95% confidence that beta_mean is equal to or greater than this lower bound}
#' \item{upper_bound}{This \code{numeric} represents 95% confidence that beta_mean is equal to or lower than this upper bound}
#' }
#' @author Manisha Parajuli, Favour Onyido, Festus Attah
#' @export
bootstrap_confi <- function(x, y, b=20, alpha = 0.05){
  n <- dim(x)[1]
  p <- dim(x)[2]
  beta <- matrix(nrow = b, ncol = p+1)

  for (i in 1:b) {
    draw <- sample(1:n, n, replace = TRUE)
    boot_x <- x[draw,]
    boot_y <- y[draw]
    beta[i,] <- Optim_log(boot_x, boot_y)
  }

  beta_mean <- apply(beta, 2, mean)
  beta_std_dev <- apply(beta, 2, sd)
  lower_bound <- beta_mean - qnorm(1 - alpha/2)*beta_std_dev
  upper_bound <- beta_mean + qnorm(1 - alpha/2)*beta_std_dev
  confi_interval <- cbind(beta_mean,lower_bound, upper_bound)
  return(confi_interval)
}


#' @title Plot of the predicted probabilities vs the true data points
#' @description This function plots the predicted probabilities vrs. the true Data Points
#' @param x A \code{matrix} that holds the independent variables
#' @param y A \code{vector} representing the dependent variable
#' @return A \code{plot} of the predicted probabilities vrs. the true Data Points
#' @author Manisha Parajuli, Favour Onyido, Festus Attah
#' @export
plot_predict <- function(x,y){
  intercept <- rep(1, n)
  px = cbind(intercept, x)
  #data <- data.frame(y,intercept,x)
  z <- px%*%Optim_log(x, y) #predicted probabilities
  for (i in 1:dim(x)[2]){
    temp <- x[,i]
    otemp=temp[order(temp)]
    oz=z[order(temp)]
    plot(y ~ temp)
    lines(sigmoid(oz)~otemp, lwd=2, col="green")
  }
}


#' @title Confusion matrix
#' @description This function generates a confusion matrix for the predicted and actual values
#' @param x A \code{matrix} that holds the independent variables
#' @param y A \code{vector} representing the dependent variable
#' @param cutoff A \code{numeric} representing the threshold to classify the predicted probabilities
#' @return A \code{table} containing the true positive, true negative, false positive, false negative values.
#' @author Manisha Parajuli, Favour Onyido, Festus Attah
#' @export
conf_mat <- function(x,y, cutoff = 0.5){
  n=dim(x)[1]
  beta <- Optim_log(x,y)
  px <-cbind(rep(1, n),x)
  y_hat <- as.vector(sigmoid(px%*%beta))
  pred <- factor(ifelse(y_hat<cutoff,0,1))
  y <- factor(y)
  levels(y) <- c('Negative', 'Positive')
  levels(pred) <- c('Negative', 'Positive')
  return (caret::confusionMatrix(pred,y)$table)
}


#' @title Calculations from the confusion matrix
#' @description This function calculates various values from the confusion matrix
#' @param mat_tab A confusion matrix \code{table}
#' @param cal A \code{character} representing the value to be computed. "pre" calculates prevelance, "acc" calculates accuracy, "sen" calculates sensitivity, "spe" calculates "specificity, "fdr" calculates false discovery ratio, "dor" calculates diagnostic odds ratio
#' @return A \code{character} of the computer value
#' @author Manisha Parajuli, Favour Onyido, Festus Attah
#' @export
cal_confusion_matrix <- function(mat_tab, cal="acc"){
  TN <- mat_tab[1]
  FN <- mat_tab[2]
  FP <- mat_tab[3]
  TP <- mat_tab[4]

  #Prevelance pre
  pre <- (FN+TP)/(TP+TN+FP+FN)

  #Accuracy acc
  acc <- (TP+TN)/(TP+TN+FP+FN)

  #Sensitivity sen
  sen <- TP/(TP+FN)

  #Specificity spe
  spe <- TN/(TN +FP)

  #False discovery ratio fdr
  fdr <- FP/(FP+TP)

  #Diagnostic odds ratio dor
  #getting lr+
  fpr <- 1-spe
  lrp <- sen/fpr

  #getting lr+
  fnr <- 1-sen
  lrn <- fnr/spe

  #getter dor
  dor <- lrp/lrn

  switch(cal,
         "pre"= return(paste("Prevelence is", pre)),
         "acc"= return(paste("Accuracy is", acc)),
         "sen"= return(paste("Sensitivity is",sen)),
         "spe"= return(paste("Specificity is",spe)),
         "fdr"= return(paste("False discovery ratio is",fdr)),
         "dor"= return(paste("Diagnostic odds ratio is",dor)))

}

#Calculates results for cutoff between 0.1-0.9
#' @title Range of calculations from the confusion matrix
#' @description This function calculates various values from the confusion matrix using series of cuttoffs between 0.1-0.9
#' @param x A \code{matrix} that holds the independent variables
#' @param y A \code{vector} representing the dependent variable
#' @param cal A \code{character} representing the value to be computed. "pre" calculates prevelance, "acc" calculates accuracy, "sen" calculates sensitivity, "spe" calculates "specificity, "fdr" calculates false discovery ratio, "dor" calculates diagnostic odds ratio
#' @return A \code{character} of the computer value
#' @author Manisha Parajuli, Favour Onyido, Festus Attah
#' @export
cal2 <- function(x,y,cal){
  cutoff <- seq(0.1, 0.9, 0.1)
  s <- length(cutoff)
  nums <- rep(NA,s)
  for (i in 1:s){
    mat_tab <- conf_mat(x,y,cutoff=cutoff[i])
    calculation <- cal_confusion_matrix(mat_tab,cal)
    nums[i] <- as.numeric(tail(strsplit(calculation,split=" ")[[1]],1))
  }

  plot(nums,cutoff,ylab="Cutoffs between 0.1-0.9", xlab="Calculated figures")
}
