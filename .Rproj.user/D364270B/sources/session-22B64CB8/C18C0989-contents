sigmoid <- function(x){
  return(1/(1 + exp(- x)))
}

#Simulating random data
set.seed(233)
n <- 1000
p <- 1
rx <- rnorm(n*p, 0, 1)
x <- matrix(rx,ncol=p)
beta <- rpois(p+1,3)
y <- as.vector(round(sigmoid(beta%*%t(cbind(1,x))+rnorm(n, 0, 1))))

#Using our function
Optim_log(x,y)

#Generating bootstrap confidence interval
bootstrap_confi(x,y)

#Plot of the predicted probabilities vs the true data points
plot_predict(x,y)

#confusion matrix
conf_mat(x,y)

#Calculations from confusion matrix
cal_confusion_matrix(conf_mat(x,y))

#Calculations from confusion matrix using a range of cuttoffs then plotting the values
cal2(x,y,"acc")
