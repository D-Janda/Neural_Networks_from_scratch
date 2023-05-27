library(tidyverse)


# Step 1: Load the data
train_data <- read.csv("data/train.csv")
test_data <- read.csv("data/test.csv")


# Step 2: Prepare the training features and labels
train_labels <- train_data[, 1]  # Extract the labels (first column)
train_features <- train_data[, -1]  # Extract the features (all columns except the first)


# Step 3: Split the data into training and testing sets
set.seed(123)  # Set seed for reproducibility
train_indices <- sample(1:nrow(train_data), size = 0.8 * nrow(train_data))  # 80% of data for training
test_indices <- setdiff(1:nrow(train_data), train_indices)  # Remaining data for testing

# Split the data based on the indices
x_train <- as.matrix(train_features[train_indices, ])
y_train <- train_labels[train_indices]
x_test <- as.matrix(train_features[test_indices, ])
y_test <- train_labels[test_indices]


# Step 4: Initialize the weights and biases
input_size <- ncol(x_train)  # Number of input features
hidden_size <- 128  # Number of units in the hidden layer
output_size <- 10  # Number of output classes

# Initialize the weights
W1 <- matrix(runif(input_size * hidden_size, -0.01, 0.01), nrow = input_size, ncol = hidden_size)
W2 <- matrix(runif(hidden_size * output_size, -0.01, 0.01), nrow = hidden_size, ncol = output_size)


# Initialize the biases
b1 <- rep(0, hidden_size)
b2 <- rep(0, output_size)


# Step 5: Forward propagation
hidden_layer_output <- x_train %*% W1 + matrix(rep(b1, nrow(x_train)), ncol = hidden_size)
hidden_layer_activation <- apply(hidden_layer_output, 2, function(x) ifelse(x > 0, x, 0))

output_layer_output <- hidden_layer_activation %*% W2 + matrix(rep(b2, nrow(hidden_layer_activation)), ncol = output_size)

# Apply the log-sum-exp trick for numerical stability
max_output <- apply(output_layer_output, 1, max)
output_layer_exp <- exp(output_layer_output - matrix(rep(max_output, ncol(output_layer_output)), ncol = output_size))
output_layer_activation <- output_layer_exp / rowSums(output_layer_exp)



# Step 6: Implement the backward pass and update the weights and biases
learning_rate <- 0.001  # Learning rate for gradient descent

# Compute the gradients
output_error <- output_layer_activation
output_error[cbind(1:nrow(output_error), y_train)] <- output_error[cbind(1:nrow(output_error), y_train)] - 1

grad_W2 <- t(hidden_layer_activation) %*% output_error
grad_b2 <- colSums(output_error)

hidden_error <- output_error %*% t(W2) * ifelse(hidden_layer_output > 0, 1, 0)

grad_W1 <- t(x_train) %*% hidden_error
grad_b1 <- colSums(hidden_error)

# Update the weights and biases
W1 <- W1 - learning_rate * grad_W1
W2 <- W2 - learning_rate * grad_W2
b1 <- b1 - learning_rate * grad_b1
b2 <- b2 - learning_rate * grad_b2

# Step 7: Repeat steps 5 and 6 for multiple iterations
num_epochs <- 10  # Number of iterations

for (epoch in 1:num_epochs) {
  # Forward pass
  hidden_layer_output <- x_train %*% W1 + matrix(rep(b1, nrow(x_train)), ncol = hidden_size)
  hidden_layer_activation <- apply(hidden_layer_output, 2, function(x) ifelse(x > 0, x, 0))

  # Apply the log-sum-exp trick for numerical stability
  max_output <- apply(output_layer_output, 1, max)
  output_layer_exp <- exp(output_layer_output - matrix(rep(max_output, ncol(output_layer_output)), ncol = output_size))
  output_layer_activation <- output_layer_exp / rowSums(output_layer_exp)

  
  # Backward pass
  output_error <- output_layer_activation
  output_error[cbind(1:nrow(output_error), y_train)] <- output_error[cbind(1:nrow(output_error), y_train)] - 1
  
  grad_W2 <- t(hidden_layer_activation) %*% output_error
  grad_b2 <- colSums(output_error)
  
  hidden_error <- output_error %*% t(W2) * ifelse(hidden_layer_output > 0, 1, 0)
  
  grad_W1 <- t(x_train) %*% hidden_error
  grad_b1 <- colSums(hidden_error)
  
  # Update the weights and biases
  W1 <- W1 - learning_rate * grad_W1
  W2 <- W2 - learning_rate * grad_W2
  b1 <- b1 - learning_rate * grad_b1
  b2 <- b2 - learning_rate * grad_b2
}


# Step 8: Make predictions on the test set
# Apply the same preprocessing steps as the training data
x_test <- x_test / 255  # Normalize pixel values

hidden_layer_output_test <- x_test %*% W1 + matrix(rep(b1, nrow(x_test)), ncol = hidden_size)
hidden_layer_activation_test <- apply(hidden_layer_output_test, 2, function(x) ifelse(x > 0, x, 0))


output_layer_output_test <- hidden_layer_activation_test %*% W2 + matrix(rep(b2, nrow(hidden_layer_activation_test)), ncol = output_size)
output_layer_activation_test <- exp(output_layer_output_test) / rowSums(exp(output_layer_output_test))

test_predictions <- apply(output_layer_activation_test, 1, which.max) - 1

test_predictions


accuracy <- round(sum(test_predictions == y_test) / length(y_test) * 100, 2)
print(paste("Accuracy:", accuracy, "%"))
