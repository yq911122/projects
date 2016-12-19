df = read.table('../input/Tweets_R.csv',header=TRUE,sep=",")

df$airline = as.factor(df$airline)
df$sentiment.fac = as.factor(df$sentiment)
df$count.fac = as.factor(df$retweet_count)

#df = df[df$retweet_count<=2,]

library(ggplot2)

#retweet count against senti scores

count <- as.data.frame(table(df$count.fac))

df$retweet_count.total <- count$Freq[match(df$count.fac,count$Var1)]
ggplot(df,aes(x=count.fac, fill=sentiment.fac))+geom_bar(position = 'fill')+geom_text(aes(y=1, label=retweet_count.total), vjust=-0.25)+scale_fill_discrete(name="sentiment", breaks=c(0, 2, 4),labels=c("negative","netural","positive"))+labs(x="retweet count", y="count (%)", title="retweet count in different senti scores (%)")
ggsave('plot/retweet_sentiscore1.png',scale=1.5)

#airline against senti scores

ggplot(df,aes(x=airline,fill=sentiment.fac))+geom_bar(position = 'dodge')+stat_bin(binwidth=1, geom="text", aes(label=..count..),position=position_dodge(width=0.9), vjust=-1) +scale_fill_discrete(name="sentiment", breaks=c(0, 2, 4),labels=c("negative","netural","positive"))+labs(x="airlines", y="count", title="airlines against senti scores")
ggsave('plot/airline_sentiscore.png',scale=1.5)

count <- as.data.frame(table(df$airline))
df$airline.total <- count$Freq[match(df$airline,count$Var1)]

ggplot(df,aes(x=airline))+geom_bar(aes(fill=sentiment.fac), position = 'fill')+geom_text(aes(y=1,label=airline.total), vjust=-1)  +scale_fill_discrete(name="sentiment", breaks=c(0, 2, 4),labels=c("negative","netural","positive"))+labs(x="airlines", y="count (%)", title="airlines against senti scores (%)")
ggsave('plot/airline_sentiscore2.png',scale=1.5)

#scores across the time (week)
df$hour <- 24*df$dayofweek+df$hour

df <- df[order(df$hour),]

ticks <- data.frame (t = seq(0,173,6), 
                     l=paste(c(rep("SUN",times=4),rep("MON",times=4),rep("TUE",times=4),rep("WED",times=4),rep("THU",times=4),rep("FRI",times=4),rep("SAT",times=4), "SUN"),
                             sprintf("%02d",c(rep(seq(0,18,6),times=7),0)),
                             sep="\n"))


ggplot(df,aes(hour,sentiment)) +  scale_x_continuous(breaks=c(ticks$t), labels=ticks$l) +  labs(x = "time", y="average sentiment score", title="average score along the week (smoothed)")+  geom_smooth(colour = "blue",span=0.3)
ggsave('plot/hour_sentiment.png',scale=1.5)

nonamerican <- df[df$airline != 'American',]
p <- ggplot(nonamerican,aes(hour,sentiment)) +
  scale_x_continuous(breaks=c(ticks$t), labels=ticks$l) +
  # labs(x="time", y="average polarity", title="average polarity (blue) and number of tweets (red) along a week (smoothed)")+
  # scale_fill_identity(name="", guide="legend", labels=c("Polarity")) +
  geom_smooth(colour = "blue",span=0.3)
ggsave('plot/hour_sentiment4.png',scale=1.5)


df$dayofweek.fac <- as.factor(df$dayofweek)
p <- ggplot(df,aes(dayofweek.fac)) + geom_bar()
  # labs(x="time", y="average polarity", title="average polarity (blue) and number of tweets (red) along a week (smoothed)")+
  # scale_fill_identity(name="", guide="legend", labels=c("Polarity")) +
  
ggsave('plot/hour_sentiment2.png',scale=1.5)

ggplot(df,aes(dayofweek.fac,fill=airline)) + geom_bar(position='stack')+ labs(x="day of week", y="count", title="# records along the week")

ggsave('plot/hour_sentiment3.png',scale=1.5)

p <- ggplot(df, aes(hour,sentiment,colour = airline)) +
  scale_x_continuous(breaks=c(ticks$t), labels=ticks$l) +
  # labs(x="time", y="average polarity", title="average polarity (blue) and number of tweets (red) along a week (smoothed)")+
  # scale_fill_identity(name="", guide="legend", labels=c("Polarity")) +
  geom_smooth(span=0.1)
ggsave('plot/hour_sentiment2.png',scale=1.5)

for(a in levels(df$airline)){
  p <- ggplot(df[df$airline == a,], aes(hour,sentiment)) +
    scale_x_continuous(breaks=c(ticks$t), labels=ticks$l) +
    # labs(x="time", y="average polarity", title="average polarity (blue) and number of tweets (red) along a week (smoothed)")+
    # scale_fill_identity(name="", guide="legend", labels=c("Polarity")) +
    geom_smooth(span=0.3)
  ggsave(paste(paste('plot/hour_sentiment',a,sep='_'),'png',sep='.'),scale=1.5)
}





data = read.csv('../input/Airline-Sentiment-2-w-AA.csv', sep=',')
library(dplyr)
data = select(data,airline_sentiment, airline, tweet_created, retweet_count)
head(data)



df$airline = as.factor(df$airline)
df$sentiment.fac = as.factor(df$sentiment)
df$count.fac = as.factor(df$retweet_count)

#df = df[df$retweet_count<=2,]

#retweet count against senti scores

count <- as.data.frame(table(df$count.fac))

df$retweet_count.total <- count$Freq[match(df$count.fac,count$Var1)]
ggplot(df,aes(x=count.fac, fill=sentiment.fac))+geom_bar(position = 'fill')+geom_text(aes(y=1, label=retweet_count.total), vjust=-0.25)+scale_fill_discrete(name="sentiment", breaks=c(0, 2, 4),labels=c("negative","netural","positive"))+labs(x="retweet count", y="count (%)", title="retweet count in different senti scores (%)")
#ggsave('plot/retweet_sentiscore1.png',scale=1.5)

#airline against senti scores

ggplot(df,aes(x=airline,fill=sentiment.fac))+geom_bar(position = 'dodge')+stat_bin(binwidth=1, geom="text", aes(label=..count..),position=position_dodge(width=0.9), vjust=-1) +scale_fill_discrete(name="sentiment", breaks=c(0, 2, 4),labels=c("negative","netural","positive"))+labs(x="airlines", y="count", title="airlines against senti scores")
#ggsave('plot/airline_sentiscore.png',scale=1.5)

count <- as.data.frame(table(df$airline))
df$airline.total <- count$Freq[match(df$airline,count$Var1)]

ggplot(df,aes(x=airline))+geom_bar(aes(fill=sentiment.fac), position = 'fill')+geom_text(aes(y=1,label=airline.total), vjust=-1)  +scale_fill_discrete(name="sentiment", breaks=c(0, 2, 4),labels=c("negative","netural","positive"))+labs(x="airlines", y="count (%)", title="airlines against senti scores (%)")
#ggsave('plot/airline_sentiscore2.png',scale=1.5)

#scores across the time (week)
df$hour <- 24*df$dayofweek+df$hour

df <- df[order(df$hour),]

ticks <- data.frame (t = seq(0,173,6), 
                     l=paste(c(rep("SUN",times=4),rep("MON",times=4),rep("TUE",times=4),rep("WED",times=4),rep("THU",times=4),rep("FRI",times=4),rep("SAT",times=4), "SUN"),
                             sprintf("%02d",c(rep(seq(0,18,6),times=7),0)),
                             sep="\n"))


ggplot(df,aes(hour,sentiment)) +  scale_x_continuous(breaks=c(ticks$t), labels=ticks$l) +  labs(x = "time", y="average sentiment score", title="average score along the week (smoothed)")+  geom_smooth(colour = "blue",span=0.3)
#ggsave('plot/hour_sentiment.png',scale=1.5)

df$dayofweek.fac <- as.factor(df$dayofweek)
ggplot(df,aes(dayofweek.fac,fill=airline)) + geom_bar(position='stack')+ labs(x="day of week", y="count", title="# records along the week")


#ggsave('plot/hour_sentiment2.png',scale=1.5)

stop_words = data.frame(
  freq = c(13573, 8630, 6605, 6032, 5288, 4669, 4455, 4342, 4134, 3982, 3882, 3760, 3708, 3642, 3270, 2904, 2523, 2372, 2100, 2072, 2023, 1924, 1912, 1727, 1723, 1718, 1688, 1619, 1511, 1484, 1478, 1476, 1375, 1372, 1334, 1252, 1229, 1199, 1194, 1158, 1140, 1133, 1074, 1048, 1047, 1027, 980, 961, 956, 952, 893, 864, 862, 821, 805, 765, 763, 762, 746, 732, 725, 701, 688, 682, 682, 670, 669, 669, 646, 643, 639, 635, 631, 625, 620, 617, 615, 607, 599, 595, 576, 561, 560, 554, 537, 536, 522, 517, 514, 513, 512, 509, 507, 506, 505, 491, 488, 487, 477, 468),
  words = c('.', 'to', 'i', 'the', '!', '?', 'a', 'you', ',', 'for', 'flight', 'on', 'and', '#', 'my', 'is', 'in', 'it', 'of', "n't", ':', '@', 'me', 'your', 'have', 'that', 'was', 'not', 'with', "'s", 'at', 'no', 'this', 'do', 'get', 'we', 'but', 'be', 'from', 'can', 'are', 'http', 'thanks', '...', 'cancelled', 'now', 'an', 'just', 'service', ';', 'so', 'been', 'help', '&', 'what', 'time', 'they', 'will', 'customer', 'up', '-', 'out', "'m", 'amp', 'our', ')', 'us', 'hours', 'when', 'how', 'hold', 'flights', 'there', 'plane', '2', 'if', 'all', 'would', 'thank', 'why', 'still', 'one', 'please', 'need', 'ca', 'delayed', 'did', '(', 'gate', 'back', 'had', 'about', 'call', 'has', 'flightled', 'or', 'bag', 'as', 'got', 'after'),
  idx = seq(1,100)
)

ticks <- data.frame (t = stop_words$idx, 
                     l = stop_words$words)

ggplot(stop_words,aes(idx,freq)) +  geom_line()
