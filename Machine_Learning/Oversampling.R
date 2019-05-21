library(imbalance)
library(ggfortify)
library(RColorBrewer)

data(glass0)

ann.df=data.frame(select(glass0,-Class),
                  class=glass0$Class)

pca<-prcomp(select(glass0,-Class))
autoplot(pca, colour = "class",size=5,data=ann.df)
autoplot(pca, x=2,y=3,colour = "class",size=5,data=ann.df)

# Oversample glass0 to get an imbalance ratio of 0.8
imbalanceRatio(glass0)
# 0.4861111

newDataset <- oversample(glass0, ratio = 0.8, method = "SMOTE")
imbalanceRatio(newDataset)

new.glass=bind_rows(glass0,newDataset) %>% 
            mutate(real=c(rep("2",length(glass0[,1])),rep("1",length(newDataset[,1]))))
         
ann.df=data.frame(select(new.glass,-Class),
                  class=new.glass$Class,
                  data.type=new.glass$real)
         
pca<-prcomp(select(new.glass,-Class,-real))
autoplot(pca,colour = "class",size=2,shape="real",data=ann.df)
autoplot(pca,2,3, colour = "class",size=2,shape="real",data=ann.df)

newDataset <- oversample(glass0, method = "ADASYN")
imbalanceRatio(newDataset)

new.glass=bind_rows(glass0,newDataset) %>% 
  mutate(real=c(rep("2",length(glass0[,1])),rep("1",length(newDataset[,1]))))

ann.df=data.frame(select(new.glass,-Class),
                  class=new.glass$Class,
                  data.type=new.glass$real)

pca<-prcomp(select(new.glass,-Class,-real))
autoplot(pca,2,3, colour = "class",size=2,shape="real",data=ann.df)

newDataset<-SMOTE(Class~., data=glass0,perc.over = 50, k = 5)
imbalanceRatio(newDataset)

new.glass=bind_rows(glass0,newDataset) %>% 
  mutate(real=c(rep("2",length(glass0[,1])),rep("1",length(newDataset[,1]))))

ann.df=data.frame(select(new.glass,-Class),
                  class=new.glass$Class,
                  data.type=new.glass$real)

pca<-prcomp(select(new.glass,-Class,-real))
autoplot(pca, colour = "class",size=2,shape="real",data=ann.df)
autoplot(pca,2,3, colour = "class",size=2,shape="real",data=ann.df)

new.neater<-neater(glass0, newDataset, k = 3, iterations = 100, smoothFactor = 1,
       classAttr = "Class")

ann.df=data.frame(select(new.neater,-Class),
                  class=new.neater$Class)

pca<-prcomp(select(new.neater,-Class))
autoplot(pca, colour = "class",size=2,data=ann.df)
autoplot(pca,2,3, colour = "class",size=2,shape="real",data=ann.df)
