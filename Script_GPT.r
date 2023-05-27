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

# Further split the training data into training and validation sets
train_indices <- sample(nrow(x_train), size = round(0.8 * nrow(x_train)), replace = FALSE)
val_indices <- setdiff(1:nrow(x_train), train_indices)

x_val <- as.matrix(x_train[val_indices, ])
y_val <- y_train[val_indices]

x_train <- as.matrix(x_train[train_indices, ])
y_train <- y_train[train_indices]



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






#### Training the model with hyperparameters ###
# Function to build the model
build_model <- function(input_size, hidden_size, output_size) {
  model <- list(
    W1 = matrix(rnorm(input_size * hidden_size), nrow = input_size, ncol = hidden_size),
    b1 = matrix(0, nrow = 1, ncol = hidden_size),
    W2 = matrix(rnorm(hidden_size * output_size), nrow = hidden_size, ncol = output_size),
    b2 = matrix(0, nrow = 1, ncol = output_size)
  )
  return(model)
}


# Function for forward propagation
forward_propagation <- function(model, x) {
  W1 <- model$W1
  b1 <- model$b1
  W2 <- model$W2
  b2 <- model$b2
  
  hidden_layer_output <- relu(x %*% W1 + matrix(rep(b1, nrow(x)), ncol = dim(W1)[2]))
  output_layer_output <- hidden_layer_output %*% W2 + matrix(rep(b2, nrow(hidden_layer_output)), ncol = dim(W2)[2])
  output_layer_activation <- softmax(output_layer_output)
  
  return(list(hidden_layer_output = hidden_layer_output, output_layer_activation = output_layer_activation))
}


# Function for backward propagation
backward_propagation <- function(model, x, y, hidden_layer_output, output_layer_activation, learning_rate) {
  W1 <- model$W1
  b1 <- model$b1
  W2 <- model$W2
  b2 <- model$b2
  
  num_examples <- nrow(x)
  
  # Compute gradients
  dZ2 <- output_layer_activation
  dZ2[cbind(1:num_examples, y + 1)] <- dZ2[cbind(1:num_examples, y + 1)] - 1
  dZ2 <- dZ2 / num_examples
  
  dW2 <- t(hidden_layer_output) %*% dZ2
  db2 <- colSums(dZ2)
  
  dA1 <- dZ2 %*% t(W2)
  dA1[hidden_layer_output <= 0] <- 0
  
  dW1 <- t(x) %*% dA1
  db1 <- colSums(dA1)
  
  # Update model parameters
  model$W1 <- model$W1 - learning_rate * dW1
  model$b1 <- model$b1 - learning_rate * db1
  model$W2 <- model$W2 - learning_rate * dW2
  model$b2 <- model$b2 - learning_rate * db2
  
  return(model)
}


# ReLU activation function
relu <- function(x) {
  return(pmax(x, 0))
}

# Softmax activation function
softmax <- function(x) {
  e_x <- exp(x - max(x))  # Subtracting max(x) for numerical stability
  return(e_x / sum(e_x))
}

# Define hyperparameter search space
learning_rate_list <- c(0.1, 0.01, 0.001)
hidden_size_list <- c(64, 128, 256)
batch_size_list <- c(32, 64, 128)
num_epochs_list <- c(50, 100, 200)

# Hyperparameter tuning
best_accuracy <- 0
best_hyperparameters <- list()

# Loop over hyperparameters
for (learning_rate in learning_rate_list) {
  for (hidden_size in hidden_size_list) {
    for (batch_size in batch_size_list) {
      for (num_epochs in num_epochs_list) {
        
        # Train the model with current hyperparameters
        model <- build_model(input_size, hidden_size, output_size)  # Build the model
        
        num_batches <- ceiling(nrow(x_train) / batch_size)
        
        for (epoch in 1:num_epochs) {
          # Shuffle the training data
          shuffle_indices <- sample(nrow(x_train))
          shuffled_x <- x_train[shuffle_indices, ]
          shuffled_y <- y_train[shuffle_indices]
          
          # Mini-batch training
          for (batch in 1:num_batches) {
            start_index <- (batch - 1) * batch_size + 1
            end_index <- min(start_index + batch_size - 1, nrow(x_train))
            batch_x <- shuffled_x[start_index:end_index, ]
            batch_y <- shuffled_y[start_index:end_index]
            
            # Forward propagation
            fp_result <- forward_propagation(model, batch_x)
            hidden_layer_output <- fp_result$hidden_layer_output
            output_layer_activation <- fp_result$output_layer_activation
            
            # Backward propagation
            model <- backward_propagation(model, batch_x, batch_y, hidden_layer_output, output_layer_activation, learning_rate)
          }
        }
        
        # Evaluate the model on the validation set
        validation_results <- forward_propagation(model, x_val)
        validation_predictions <- apply(validation_results$output_layer_activation, 1, which.max) - 1
        validation_accuracy <- sum(validation_predictions == y_val) / length(y_val)
        
        # Check if current hyperparameters yield better accuracy
        if (validation_accuracy > best_accuracy) {
          best_accuracy <- validation_accuracy
          best_hyperparameters$learning_rate <- learning_rate
          best_hyperparameters$hidden_size <- hidden_size
          best_hyperparameters$batch_size <- batch_size
          best_hyperparameters$num_epochs <- num_epochs
        }
        
        # Print the validation accuracy for current hyperparameters
        print(paste("Hyperparameters:", "Learning Rate:", learning_rate, "Hidden Size:", hidden_size, "Batch Size:", batch_size, "Num Epochs:", num_epochs))
        print(paste("Validation Accuracy:", validation_accuracy))
      }
    }
  }
}

# Print the best hyperparameters and accuracy
print("Best Hyperparameters:")
print(best_hyperparameters)
print(paste("Best Accuracy:", best_accuracy))s

