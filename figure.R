library(ggplot2)
library(grid)
library(grDevices)

setwd("~/Desktop")
rand1=readLines("rand_result_1.txt")
rand2=readLines("rand_result_2.txt")
rand3=readLines("rand_vader_result.txt")
w2v1=readLines("w2v_result_1.txt")
w2v2=readLines("w2v_result_2.txt")
w2v3=readLines("w2v_vader_result.txt")

#rand之间的比较。包含三个方面：交叉验证的准确率、交叉验证的损失函数值、训练时间
rand1=rand1[(2:51)*2-1]
rand2=rand2[(2:51)*2-1]
rand3=rand3[(2:51)*2-1]
mfun1=function(x,i) return(strsplit(gsub(" - ","#",gsub(": ","#",x)),"#")[[1]][i])
val_acc1=sapply(rand1,mfun1,9)
val_acc2=sapply(rand2,mfun1,9)
val_acc3=sapply(rand3,mfun1,9)
val_loss1=sapply(rand1,mfun1,7)
val_loss2=sapply(rand2,mfun1,7)
val_loss3=sapply(rand3,mfun1,7)
time1=sapply(rand1,mfun1,1)
time2=sapply(rand2,mfun1,1)
time3=sapply(rand3,mfun1,1)

pd1=data.frame(epoch=1:50,acc=c(val_acc1,val_acc2,val_acc3),loss=c(val_loss1,val_loss2,val_loss3),time=c(time1,time2,time3),fill=rep(c("rand_1","rand_2","rand_vader"),each=50))
pd1$acc=as.numeric(as.character(pd1$acc))
pd1$loss=as.numeric(as.character(pd1$loss))
pd1$time=gsub("s","",pd1$time)
pd1$time=as.numeric(as.character(pd1$time))
pfun1=function(){
  ph1=ggplot(data=pd1,aes(x=epoch,y=acc,color=fill))+
    geom_line(size=0.8,alpha=0.8)+
    geom_point(aes(shape=fill),size=2,alpha=0.8)+
    labs(x="Epoch",y="Validation Accuracy",title="Random Embedding")+
    scale_color_manual(values=c("darkred","darkblue","darkgreen"))+
    theme(plot.title=element_text(hjust=0.5),
          legend.title=element_blank(),
          legend.position=c(0.8,0.13),
          legend.text=element_text(size=15),
          axis.title=element_text(size=15),
          title=element_text(size=18))
  #print(ph1)
  ph2=ggplot(data=pd1,aes(x=epoch,y=loss,color=fill))+
    geom_line(size=0.8,alpha=0.8)+
    geom_point(aes(shape=fill),size=2,alpha=0.8)+
    labs(x="Epoch",y="Validation Loss",title="Random Embedding")+
    scale_color_manual(values=c("darkred","darkblue","darkgreen"))+
    theme(plot.title=element_text(hjust=0.5),
          legend.title=element_blank(),
          legend.position=c(0.8,0.13),
          legend.text=element_text(size=15),
          axis.title=element_text(size=15),
          title=element_text(size=18))
  #print(ph2)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(1,2)))
  vplayout <- function(x,y){
    viewport(layout.pos.row = x, layout.pos.col = y)
  }
  print(ph1, vp = vplayout(1,1))
  print(ph2, vp = vplayout(1,2))
}
pdf("rand.pdf",width=9,height=5)
pfun1()
dev.off()

#w2v之间的比较。包含三个方面：交叉验证的准确率、交叉验证的损失函数值、训练时间
w2v1=w2v1[(2:101)*2-1]
w2v2=w2v2[(2:101)*2-1]
w2v3=w2v3[(2:101)*2-1]
mfun1=function(x,i) return(strsplit(gsub(" - ","#",gsub(": ","#",x)),"#")[[1]][i])
val_acc1=sapply(w2v1,mfun1,9)
val_acc2=sapply(w2v2,mfun1,9)
val_acc3=sapply(w2v3,mfun1,9)
val_loss1=sapply(w2v1,mfun1,7)
val_loss2=sapply(w2v2,mfun1,7)
val_loss3=sapply(w2v3,mfun1,7)
time1=sapply(w2v1,mfun1,1)
time2=sapply(w2v2,mfun1,1)
time3=sapply(w2v3,mfun1,1)

pd2=data.frame(epoch=1:100,acc=c(val_acc1,val_acc2,val_acc3),loss=c(val_loss1,val_loss2,val_loss3),time=c(time1,time2,time3),fill=rep(c("w2v_1","w2v_2","w2v_vader"),each=50))
pd2$acc=as.numeric(as.character(pd2$acc))
pd2$loss=as.numeric(as.character(pd2$loss))
pd2$time=gsub("s","",pd2$time)
pd2$time=as.numeric(as.character(pd2$time))
pfun2=function(){
  ph1=ggplot(data=pd2,aes(x=epoch,y=acc,color=fill))+
    geom_line(size=0.8,alpha=0.8)+
    geom_point(aes(shape=fill),size=1.5,alpha=0.8)+
    labs(x="Epoch",y="Validation Accuracy",title="Word2vec Embedding")+
    scale_color_manual(values=c("darkred","darkblue","darkgreen"))+
    theme(plot.title=element_text(hjust=0.5),
          legend.title=element_blank(),
          legend.position=c(0.79,0.13),
          legend.text=element_text(size=15),
          axis.title=element_text(size=15),
          title=element_text(size=18))
  #print(ph1)
  ph2=ggplot(data=pd2,aes(x=epoch,y=loss,color=fill))+
    geom_line(size=0.8,alpha=0.8)+
    geom_point(aes(shape=fill),size=1.5,alpha=0.8)+
    labs(x="Epoch",y="Validation Loss",title="Word2vec Embedding")+
    scale_color_manual(values=c("darkred","darkblue","darkgreen"))+
    theme(plot.title=element_text(hjust=0.5),
          legend.title=element_blank(),
          legend.position=c(0.79,0.87),
          legend.text=element_text(size=15),
          axis.title=element_text(size=15),
          title=element_text(size=18))
  #print(ph2)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(1,2)))
  vplayout <- function(x,y){
    viewport(layout.pos.row = x, layout.pos.col = y)
  }
  print(ph1, vp = vplayout(1,1))
  print(ph2, vp = vplayout(1,2))
}
pdf("w2v.pdf",width=9,height=5)
pfun2()
dev.off()
