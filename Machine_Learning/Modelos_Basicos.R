rm(list=ls())
gc(reset = TRUE)
memory.limit()
memory.size()


data("Titanic")
View(Titanic)
titanic <- data.frame(Titanic)
#Modelo de Benchmark
?glm
model_benchmark <- glm(Survived ~ Class+Sex+Age,data = titanic,family = binomial(link = "logit"),weights = titanic$Freq)
summary(model_benchmark)

library(e1071)

model_NaiveBayes <- naiveBayes(Survived ~ Class+Sex+Age,data = Titanic) # solo acepta tabla de frecuencias, o en todo caso un data.frame de los campos repetidos
model_NaiveBayes

data.frame(cbind(titanic,
                 Class_Bayes = predict(model_NaiveBayes,newdata = Titanic),
                 Class_RL = predict(model_benchmark,newdata = Titanic,type = "response")>0.6))

# hist(predict(model_benchmark,newdata = Titanic,type = "response"))

model_NaiveBayes2 <- naiveBayes(Survived ~ Class+Sex+Age+Sex*Class,data = Titanic)
model_benchmark2 <- glm(Survived ~ Class+Sex+Age+Sex*Class,data = titanic,family = binomial(link = "logit"),weights = titanic$Freq)
summary(model_benchmark2)
model_NaiveBayes2

data.frame(cbind(titanic,
                 Class_Bayes = predict(model_NaiveBayes,newdata = Titanic),
                 Class_RL = predict(model_benchmark,newdata = Titanic,type = "response")>0.6,
                 Class_Bayes2 = predict(model_NaiveBayes2,newdata = Titanic),
                 Class_RL2 = predict(model_benchmark2,newdata = Titanic,type = "response")>0.6
                )
           )


list.of.packages <- c("tree", "dplyr", "randomForest", "titanic", "caret", "ROCR")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(tree)
library(dplyr)
library(randomForest)
library(titanic)
library(caret)
library(ROCR)

model_Arbol <- tree(Survived ~ Age+Pclass,data = titanic_train) # este modelo no es muy bueno cuando quieres hacer particiones de modo distinto a las que normalmente realiza el arbol (es decir no puede realizar particiones diagonales o particiones circulares ejem: base iris las relaciones son diagonales), en estos casos knn es mucho mejor.
predict(miArbol,newdata = titanic_test)

titanic_train <- titanic_train %>% filter(!is.na(Age)) %>% mutate(Survived = factor(Survived))

model_Random <- randomForest(Survived ~ Age+Pclass,data = titanic_train)

library(class)

muestra <- sample(1:nrow(iris),110)
# iris.train <- iris %>% sample_n(110)

iris.train <- iris[muestra,]
iris.test <- iris[-muestra,]

cl = iris.train$Species
# train <- iris.train[,colnames(iris.train) != "Species"]
train <- iris.train %>% select(-Species)
test <- iris.test %>% select(-Species)

model_knn <- knn(train = train,test = test,cl = cl,k = 10,prob = T) # este modelo no es muy bueno con la interaccion de variables que no tengan sentido ejem: interaccion entre edad, nivel socioeconomico, sexo

