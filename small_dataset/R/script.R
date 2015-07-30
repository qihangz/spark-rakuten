data <- read.csv("/home/qihangz/Documents/spark-1.4.0-bin-hadoop2.6/dense_data.csv", sep="\t", header=FALSE)
data <- data[,-387]
y <- data[,128]
x <- data[,-128]
start <- proc.time()
glm.out = glm(y~as.matrix(x), family=binomial(logit), data=cbind.data.frame(x,y))
end <- proc.time()


summary(glm.out)
