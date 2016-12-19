date <- read.table("date_totalemo_tweetnum_avgemo_pre.csv",sep=",")
tweetlen <- read.table("tweetlen_emodis_pre.csv",sep=",")
usr <- read.table("usr_avgtweetlen_avgemo_tweetnum_pre.csv",sep=",")
usr.at <- read.table("user_avgat_pre.csv",sep=",")

usr$at <- usr.at$V2

colnames(date) <- c("day","hour","totpol","num","pol")
colnames(tweetlen) <- c("len","pol","num")
colnames(usr) <- c("name","len","pol","num","at")

date$day <- ordered(factor(date$day), levels = c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")) 
date <- date[order(date$day, date$hour),]

library(ggplot2)



dualaxisplot <- function(p1,p2){ 
  require(grid)
  library(gtable)
  # extract gtable
  g1 <- ggplot_gtable(ggplot_build(p1))
  g2 <- ggplot_gtable(ggplot_build(p2))
  
  # overlap the panel of 2nd plot on that of 1st plot
  pp <- c(subset(g1$layout, name == "panel", se = t:r))
  g <- gtable_add_grob(g1, g2$grobs[[which(g2$layout$name == "panel")]], pp$t, 
                       pp$l, pp$b, pp$l)
  
  # axis tweaks
  ia <- which(g2$layout$name == "axis-l")
  ga <- g2$grobs[[ia]]
  ax <- ga$children[[2]]
  ax$widths <- rev(ax$widths)
  ax$grobs <- rev(ax$grobs)
  ax$grobs[[1]]$x <- ax$grobs[[1]]$x - unit(1, "npc") + unit(0.15, "cm")
  g <- gtable_add_cols(g, g2$widths[g2$layout[ia, ]$l], length(g$widths) - 1)
  g <- gtable_add_grob(g, ax, pp$t, length(g$widths) - 1, pp$b)
  
  ia <- which(g2$layout$name == "ylab")
  ax <- g2$grobs[[ia]]
  # str(ax) # you can change features (size, colour etc for these - 
  # change rotation below 
  ax$rot <- 270
  g <- gtable_add_cols(g, g2$widths[g2$layout[ia, ]$l], length(g$widths) - 1)
  g <- gtable_add_grob(g, ax, pp$t, length(g$widths) - 1, pp$b)
  
  if ("guide-box" %in% g1$layout$name){
    leg1 <- g1$grobs[[which(g1$layout$name == "guide-box")]]
    leg2 <- g2$grobs[[which(g2$layout$name == "guide-box")]]
    
    g$grobs[[which(g$layout$name == "guide-box")]] <- 
      gtable:::cbind_gtable(leg1, leg2, "first")
  }
  
  return(g)
  # draw it
}
  pdf("pol_num.pdf", width=18, height=9)
  require(grid)
  grid.newpage()
  
  ticks <- data.frame (t = seq(0,173,6),  
                       l=paste(c(rep("SUN",times=4),rep("MON",times=4),rep("TUE",times=4),rep("WED",times=4),rep("THU",times=4),rep("FRI",times=4),rep("SAT",times=4), "SUN"),
                               sprintf("%02d",c(rep(seq(0,18,6),times=7),0)),
                               sep="\n"))


  # two plots
  p1 <- ggplot(date, aes(1:nrow(date),pol)) +
#    geom_line(colour = "blue") + 
  theme(legend.position="bottom") + theme_bw() +
    scale_x_continuous(breaks=c(ticks$t), labels=ticks$l) +
    labs(x="time", y="average polarity", title="average polarity (blue) and number of tweets (red) along a week (smoothed)")+
    scale_fill_identity(name="", guide="legend", labels=c("Polarity")) +
  geom_smooth(colour = "blue",span=0.3)

  p2 <- ggplot(date, aes(1:nrow(date),num)) +   
#    geom_line(colour = "red") + 
    theme(panel.background = element_rect(fill = NA), legend.position="bottom") +
    scale_x_continuous(breaks=c(ticks$t), labels=ticks$l) +
    scale_fill_identity(name="", guide="legend", labels=c("Num"))+
    ylab("number of tweet")+
  geom_smooth(aes(1:nrow(date),num),colour = "red",span=0.3)

  g <- dualaxisplot(p1,p2)

  grid.draw(g)  
  dev.off()


 
  pdf("pol_len.pdf", width=18, height=9)

  threadhold <- 100
  tweetlen.re <- tweetlen[tweetlen$num > threadhold,]  

  ggplot(tweetlen.re)+geom_line(aes(x=len,y=num),color="blue") + labs(x='length', y='counts')

  ggplot(tweetlen.re)+geom_line(aes(x=len,y=pol/2-1),color="blue") + geom_line(linetype = 2, aes(x=len, y=0)) + labs(x='length',y='normalized sentiment scores')

  ggplot(tweetlen.re)+geom_line(aes(x=len,y=pol*num/2-num),color="blue") +geom_line(linetype = 2, aes(x=len, y=0)) + labs(x='length',y='normalized overall sentiment scores')
  dev.off()


  usr.data <- usr[,c("len","pol","num","at")]

library(ggplot2)
pdf("dis.pdf", width=18, height=9)

  usr.data.pos <- usr.data[usr.data$pol>=2,]
  usr.data.neg <- usr.data[usr.data$pol<2,]
  ggplot(usr.data)+geom_density(aes(x=len,y = ..density..),binwidth=1) + xlim(0.0,50) + ggtitle("tweet length overall distribution")
  ggplot(usr.data.pos)+geom_histogram(aes(x=len, y = ..density..),binwidth=1) + xlim(0.0,50) +ggtitle("tweet length (positive) distribution")
  ggplot(usr.data.neg)+geom_histogram(aes(x=len, y = ..density..),binwidth=1) + xlim(0.0,50)+ggtitle("tweet length (negative) distribution")

  ggplot(usr.data)+geom_histogram(aes(x=pol,y = ..density..),binwidth=0.3)+ggtitle("tweet polarity  distribution")
  ggplot(usr.data)+geom_histogram(aes(x=num,y = ..density..),binwidth=0.5) + xlim(0.0,20)+ggtitle("tweet number distribution")
  ggplot(usr.data)+geom_histogram(aes(x=at,y = ..density..),binwidth=0.1) + xlim(0.0,2.5)+ggtitle("tweet @ distribution")
dev.off()

pdf("relation.pdf", width=18, height=9)
  usr.data.cor <- usr.data[usr.data$len<50 & usr.data$num <10 & usr.data$at<=1.0,]  

  ggplot(usr.data.cor)+geom_point(aes(x=len,y = pol)) + ggtitle("length vs polarity")
  ggplot(usr.data.cor)+geom_point(aes(x=len,y = num)) + ggtitle("length vs number")
  ggplot(usr.data.cor)+geom_point(aes(x=len,y = at)) + ggtitle("length vs at")
  ggplot(usr.data.cor)+geom_point(aes(x=pol,y = num)) + ggtitle("polarity vs number")
  ggplot(usr.data.cor)+geom_point(aes(x=pol,y = at)) + ggtitle("polarity vs at")
  ggplot(usr.data.cor)+geom_point(aes(x=num,y = at)) + ggtitle("number vs at")

dev.off()


library(cluster)
    
  usr.data.lenpol <- usr.data.cor[,c("len","pol")]
  usr.data.lenat <- usr.data.cor[,c("len","at")]

  usr.data.lenpol <- scale(usr.data.lenpol)
  usr.data.lenat <- scale(usr.data.lenat)

  cl.lenpol <- kmeans(usr.data.lenpol, 2, iter.max=20)

  cl.lenat <- kmeans(usr.data.lenat, 2, iter.max=20)

pdf("cluster.pdf", width=18, height=9)
  plot(usr.data.lenpol, col=cl.lenpol$cluster)

  plot(usr.data.lenat, col=cl.lenat$cluster)
dev.off()

library(tm)
library(wordcloud)

creatWordCloud <- function(lords){
  lords <- tm_map(lords, stripWhitespace)
  lords <- tm_map(lords, tolower)
  lords <- tm_map(lords, removeWords, stopwords("english"))
  lords <- tm_map(lords, stemDocument)
  lords <- tm_map(lords, PlainTextDocument)
  
  wordcloud(lords, scale=c(5,0.5), max.words=100, random.order=FALSE, rot.per=0.35, use.r.layout=FALSE, colors=brewer.pal(8, "Dark2"))
}


lords <- Corpus (DirSource("./pos/"))

creatWordCloud(lords)

lords <- Corpus (DirSource("./neg/"))

creatWordCloud(lords)

