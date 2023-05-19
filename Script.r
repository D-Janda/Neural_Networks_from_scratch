## Starting with local environmnet
renv::init() ### For more info check https://rstudio.github.io/renv/articles/renv.html # nolint: line_length_linter.

library(tidyverse)

install.packages('httpgd')
library(httpgd) ### For better plots in VS code


renv::snapshot() ### To save the environment

### Data from kaggle digit recognizer
train_data <- read.csv('data/train.csv')
test_data <- read.csv('data/test.csv')


train <- as.matrix(train_data)  #### Create matrix to do the math
train <- as.array(train)

