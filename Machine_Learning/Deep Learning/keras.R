# install.packages("keras")
library("keras")
# install_keras()

mnist <- dataset_mnist()

image(as.matrix(mnist$train$x[2,,]))

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# flatten 
dim(x_train) <- c(60000, 784)
dim(x_test) <- c(10000, 784)

# normalizar el input (de 0 a 255) a (0-1)
x_train <- x_train/255
x_test <- x_test/255

# one-hot encoding o dummy variable
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


# Modelo!

modelo <- keras_model_sequential()
modelo %>% 
  layer_dense(unit=200, activation = "relu", input_shape = c(784), name = "jestrudis") %>%
  layer_dropout(rate = .4) %>%
  layer_dense(unit=100, activation = "relu") %>%
  layer_dropout(rate = .3) %>%
  layer_dense(unit=10, activation = "softmax")


# Funcion de coste
modelo %>%
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(),
    metric = c("accuracy")
  )


resultado <- fit(modelo, x_train, y_train, epochs = 50,
                 batch_size = 128, validation_split = 0.2)

