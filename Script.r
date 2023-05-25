## Starting with local environmnet
renv::init() ### For more info check https://rstudio.github.io/renv/articles/renv.html # nolint: line_length_linter.

library(tidyverse)
library(ggplot2)
library(httpgd) ### For better plots in VS code
library(reshape2)

renv::snapshot() ### To save the environment

### Data from kaggle   
train <- read.csv('data/train.csv')
test <- read.csv('data/test.csv')

#### Training datasets ####
xtrain <- train[1:floor(nrow(train)*0.6),]
xtest <- train[(floor(nrow(train)*0.8)+1):nrow(train),]
xval <- train[(floor(nrow(train) * 0.6) + 1) : floor(nrow(train) * 0.8),]


train[train == 0] <- 10
ytrain <- train[1:floor(nrow(train)*0.6), 1]
yval <- train[(floor(nrow(train)*0.6) + 1):floor(nrow(train)*0.8), 1]
ytest <- train[(floor(nrow(train)*0.8) + 1):nrow(train), 1]


### visualizating function ###
# visualize recreates a square image from the unrolled matrix that represents it

visualize <- function(imgvec) {

    n <- length(imgvec)
    
    # Reshape the vector into a matrix
    img <- matrix(imgvec, sqrt(n))
    
    # Reformat the data for plotting
    imgmelt <- melt(img)
    
    p <- ggplot(imgmelt, aes(x = Var1, y = -Var2 + sqrt(n) + 1)) +
        geom_raster(aes(fill = imgmelt$value)) +
        scale_fill_gradient(low = "white", high = "black", guide = FALSE) +
        labs(x = NULL, y = NULL, fill = NULL)
  
    print(p)       
}

visualize(
    as.numeric(xtrain[10, -1])
)

