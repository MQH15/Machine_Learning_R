install.packages('keras')
library(keras)
install_keras()


mnist <- dataset_mnist()

image(mnist$train$x[2,,])

x_train <- mnist$train$x
y_train <- mnist$train$y

x_test <- mnist$test$x
y_test <- mnist$test$y

# flatten
dim(x_train) <- c(60000,784)
dim(x_test) <- c(10000,784)

x_train <- x_train / 255
x_test <- x_test / 255


y_train <- to_categorical(y_train,10)
y_test <- to_categorical(y_test,10)


model_snn <- keras_model_sequential()
model_snn %>% 
  layer_dense(units = 200,activation = "relu",input_shape = c(784)) %>% 
  layer_dropout(rate = .4) %>% 
  layer_dense(units = 100,activation = "relu") %>% 
  layer_dropout(rate = .3) %>% 
  layer_dense(units = 10,activation = "softmax")

summary(model_snn)  
# layer_dropout(rate = .4) quita de manera aleatoria neuronas, rate es la probabilidad de quitar la neurona - esto evita overfiting
# units: numero de variables
# activation: funcion que se aplica - relu dobla el espacio

# funcion de coste
model_snn %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(),
    metric = c("accuracy")
  )
#    optimizer = optimizer_sgd(lr = ) # este lr es learning rate

resultado <- fit(model_snn,x_train,y_train,epochs = 50,batch_size = 128,validation_split = 0.2)

plot(resultado)
# epoch numero de iteraciones
# batch_size numero de registros en cada iteracion - cuanto mas grande mejor ademas ponerlo en base 2 elevado a un digito
# truquillo de red neuronal ejecutar el modelo con lr alto hasta cierto punto, luego ejecutar bajando este parametro

model_snn %>% evaluate(x_test,y_test)
model_snn %>% predict_classes(x_test)

y_test_real <- mnist$test$y

data_pred <- data.frame(cbind(y_real = y_test_real,y_estimado = predict_classes(model_snn,x_test)))

nrow((data_pred[data_pred$y_real != data_pred$y_estimado,]))

image(mnist$test$x[321,,])


######################################################################################################
## Convolucional NN
######################################################################################################
## sin flatten

x_train <- x_train %>% array_reshape(c(60000,28,28,1))
x_test <- x_test %>% array_reshape(c(10000,28,28,1))


layer_conv_2d(kernel_size = c(3,3),activation = "relu",filters = 32,input_shape = (28,28,1))
# kernel_size no deberia tener tamaño parecido a las dimensiones de la imagen
# filters numero de imagenes convolucionadas( imagen acotada desde la original) salidas, es mejor potencias de 2
# tercer valor de input_shape (1: blanco negro , 3: color)

model_scnn <- keras_model_sequential()
model_scnn %>% 
  layer_conv_2d(kernel_size = c(3,3),activation = "relu",filters = 32,input_shape = c(28,28,1)) %>% 
  layer_conv_2d(kernel_size = c(3,3),activation = "relu",filters = 64) %>% 
  
  layer_max_pooling_2d(pool_size=c(3,3)) %>% 
  
  layer_dropout(rate = 0.25) %>% 
  
  layer_flatten() %>% 
  
  layer_dense(units = 128,activation = "relu") %>% 
  
  layer_dropout(rate = 0.5) %>% 
  
  layer_dense(units = 10,activation = "softmax")
# con el parametro strides mas alto puede ejecutar mas rapido

# funcion de coste
model_scnn %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adadelta(),
    metric = c("accuracy")
  )


resultado_cnn <- fit(model_scnn,x_train,y_train,epochs = 10,batch_size = 128,validation_split = 0.2)

modelo_snn %>% save_model_hdf5("modelo_snn.h5")
modelo_snn %>% load_model_hdf5("modelo_snn.h5")

##############################################################################################################
# Tarea

# ejecutar el modelo en esta data dataset_fashion_mnist()
