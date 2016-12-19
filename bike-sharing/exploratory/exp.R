library(ggplot2)



df = read.table('../input/train_processed.csv',header=TRUE,sep=",")

df$season = as.factor(df$season)
df$holiday = as.factor(df$holiday)
df$workingday = as.factor(df$workingday)
df$weather = as.factor(df$weather)


savepng <- function(filename){
  ggsave(paste(paste('plot',filename, sep='/'), 'png',sep='.'),scale=1.5)
}

#by time:season
ggplot(df, aes(x=count,fill=season, color=season))+geom_density(alpha = 0.3)
savepng('season')

ggplot(df, aes(x=season, y=count))+geom_boxplot()
savepng('season_box')

#by time:month
ticks <- data.frame (t = seq(1,12), l= c('Jan','Feb','Mar','Apr','May','Jun','July','Aug','Sep','Oct','Nov','Dec'))
ggplot(df, aes(x=month))+geom_smooth(aes(y=count, color="blue"),span = 0.3)+geom_smooth(aes(y=casual,color="red"),span = 0.3)+geom_smooth(aes(y=registered,color="yellow"),span = 0.3)+scale_x_continuous(breaks=c(ticks$t), labels=ticks$l)+scale_colour_manual(name="type",values=c("blue","red","yellow"), labels = c("count","casual","registered"))
savepng('month')


#by time:week and hour
df$hour.dow <- 24*df$dayofweek+df$hour

#df <- df[order(df$hour.dow),]

ticks <- data.frame (t = seq(0,173,6),
                     l=paste(c(rep("SUN",times=4),rep("MON",times=4),rep("TUE",times=4),rep("WED",times=4),rep("THU",times=4),rep("FRI",times=4),rep("SAT",times=4), "SUN"),
                             sprintf("%02d",c(rep(seq(0,18,6),times=7),0)),
                             sep="\n"))

ggplot(df, aes(x=hour.dow))+geom_smooth(aes(y=count, color="blue"),span = 0.3)+geom_smooth(aes(y=casual,color="red"),span = 0.3)+geom_smooth(aes(y=registered,color="yellow"),span = 0.3)+scale_x_continuous(breaks=c(ticks$t), labels=ticks$l)+scale_colour_manual(name="type",values=c("blue","red","yellow"), labels = c("count","casual","registered"))
savepng('hour')


#holiday and workingday
holiday.sum <- c(length(which(df$holiday==0)), length(which(df$holiday==1)))
workingday.sum <- c(length(which(df$workingday==0)), length(which(df$workingday==1)))

temp <- data.frame(holiday = rep(df$holiday, times=2),
                   workingday = rep(df$workingday, times=2),
                   count = c(df$registered, df$casual),
                   type = as.factor(c(rep(1,times=length(df$registered)),rep(2,times=length(df$casual)))),
                   holiday.sums = ifelse(df$holiday == 0, holiday.sum[1], holiday.sum[2]),
                   workingday.sums = ifelse(df$workingday == 0, workingday.sum[1], workingday.sum[2])
                   )
ggplot(temp, aes(x=holiday,y=count,fill=type))+geom_bar(position='stack',stat = "identity")+scale_fill_discrete(name="type",breaks=c(1,2), labels = c("registered","casual"))
savepng('holiday_stacked')

#ggplot(temp, aes(x=holiday,y=count,fill=type))+geom_bar(position='fill',stat = "identity")+scale_fill_discrete(name="type",breaks=c(1,2), labels = c("registered","casual"))
#savepng('holiday_filled')

ggplot(temp, aes(x=holiday,y=count/holiday.sums,fill=type))+geom_bar(position='stack',stat = "identity")+scale_fill_discrete(name="type",breaks=c(1,2), labels = c("registered","casual"))
savepng('holiday_stacked_avg')

ggplot(temp, aes(x=workingday,y=count,fill=type))+geom_bar(position='stack',stat = "identity")+scale_fill_discrete(name="type",breaks=c(1,2), labels = c("registered","casual"))
savepng('workingday_stacked')

#ggplot(temp, aes(x=workingday,y=count,fill=type))+geom_bar(position='fill',stat = "identity")+scale_fill_discrete(name="type",breaks=c(1,2), labels = c("registered","casual"))
#savepng('workingday_filled')

ggplot(temp, aes(x=workingday,y=count/workingday.sums,fill=type))+geom_bar(position='stack',stat = "identity")+scale_fill_discrete(name="type",breaks=c(1,2), labels = c("registered","casual"))
savepng('workingday_stacked_avg')


#weather
temp <- data.frame(weather = rep(df$weather, times=2),
                   count = c(df$registered, df$casual),
                   type = as.factor(c(rep(1,times=length(df$registered)),rep(2,times=length(df$casual))))
)

weather.sum <- c('1' = length(which(df$weather==1)), '2' = length(which(df$weather==2)),'3' = length(which(df$weather==3)),'4' = length(which(df$weather==4)))
for(i in df$weather){
  temp$weather.sums <- weather.sum[[i]]
}


ggplot(temp, aes(x=weather,y=count, fill=type))+geom_bar(position='stack',stat = "identity")+scale_fill_discrete(name="type",breaks=c(1,2), labels = c("registered","casual"))
savepng('weather')

ggplot(temp, aes(x=weather,y=count/weather.sums, fill=type))+geom_bar(position='stack',stat = "identity")+scale_fill_discrete(name="type",breaks=c(1,2), labels = c("registered","casual"))
savepng('weather_avg')

#ggplot(temp, aes(x=weather,y=count, fill=type))+geom_bar(position='fill',stat = "identity")+scale_fill_discrete(name="type",breaks=c(1,2), labels = c("registered","casual"))
#savepng('weather')

#temp vs. atemp vs. humidity vs. windspeed
ggplot(df, aes(temp)) + geom_density()
savepng('temp')

ggplot(df, aes(atemp)) + geom_density()
savepng('atemp')

ggplot(df, aes(humidity)) + geom_density()
savepng('humidity')

ggplot(df, aes(windspeed)) + geom_density()
savepng('windspeed')

ggplot(df, aes(temp)) + geom_density()
savepng('temp')

ggplot(df, aes(atemp)) + geom_boxplot()
savepng('atemp')

ggplot(df, aes(humidity)) + geom_density()
savepng('humidity')

ggplot(df, aes(windspeed)) + geom_density()
savepng('windspeed')

ggplot(df, aes(count)) + geom_density()
savepng('count')

#temp vs. humidity vs. windspeed -> weather
library(scatterplot3d)
df$colors[df$weather==1] <- 'pink'
df$colors[df$weather==2] <- 'green'
df$colors[df$weather==3] <- 'yellow'
df$colors[df$weather==4] <- 'blue'

with(df,{
    s3d <- scatterplot3d(temp, humidity, windspeed,        # x y and z axis
                         color=addTrans(colors,100), pch=20,        # circle color indicates no. of cylinders
                         main="weather against temp, humidity and windspeed",
                         xlab="temperature (C)",
                         ylab="humidity (%)",
                         zlab="wind speed (m/s)")
                          

    # add the legend
    legend("bottom", inset=-.25,      # location and inset
           horiz = TRUE,cex=.5,              # suppress legend box, shrink text 50%
           title="weather",
           legend = levels(df$weather),
           col=c("pink", "green", "yellow","blue"))
})

addTrans <- function(color,trans)
{
  # This function adds transparancy to a color.
  # Define transparancy with an integer between 0 and 255
  # 0 being fully transparant and 255 being fully visable
  # Works with either color and trans a vector of equal length,
  # or one of the two of length 1.
  
  if (length(color)!=length(trans)&!any(c(length(color),length(trans))==1)) stop("Vector lengths not correct")
  if (length(color)==1 & length(trans)>1) color <- rep(color,length(trans))
  if (length(trans)==1 & length(color)>1) trans <- rep(trans,length(color))
  
  num2hex <- function(x)
  {
    hex <- unlist(strsplit("0123456789ABCDEF",split=""))
    return(paste(hex[(x-x%%16)/16+1],hex[x%%16+1],sep=""))
  }
  rgb <- rbind(col2rgb(color),trans)
  res <- paste("#",apply(apply(rgb,2,num2hex),2,paste,collapse=""),sep="")
  return(res)
}


library(rgl)

plot3d(df$temp, df$humidity, df$windspeed, type="p", col=addTrans(df$colors,100))
legend3d("topright", legend = c('spring','summer','autumn','winter'), pch = 16, col =c("pink", "green", "yellow","blue"))


ggplot(df, aes(temp)) + geom_jitter(aes(color=weather))
savepng('temp_weather')

ggplot(df, aes(humidity, color=weather)) + geom_jitter()
savepng('humidity_weather')

ggplot(df, aes(humidity, color=weather)) + geom_jitter()
savepng('windspeed_weather')

#temp vs. atemp vs. humidity vs. windspeed plot
library(corrplot)
M <-cor(cbind(df$temp, df$atemp, df$humidity, df$windspeed, df$count))

png(filename="plot/temp_atemp_humidity_windspeed_count.png")
corrplot(M, method="pie")
dev.off()

#temp vs. atemp vs. humidity vs. windspeed
ggplot(df, aes(x=temp, y=count))+geom_point()
savepng('temp_count')

ggplot(df, aes(x=atemp, y=count))+geom_point()
savepng('atemp_count')

ggplot(df, aes(x=humidity, y=count))+geom_point()
savepng('humidity_count')

ggplot(df, aes(x=windspeed, y=count))+geom_point()
savepng('windspeed_count')

ggplot(df, aes(x=count, fill=workingday)) + geom_bar()

M <- cor(cbind(df$casual, df$registered, df$count))

png(filename="plot/casual_registered_count.png")
corrplot(M, method="pie")
dev.off()

# Save average counts for each day/time in data frame

day_hour_counts <- as.data.frame(aggregate(df[,"count"], list(df$dayofweek, df$hour), mean))
day_hour_counts$Group.1 <- factor(day_hour_counts$Group.1, ordered=TRUE, levels=0:6,labels=c("Monday","Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))

day_hour_counts$hour <- as.numeric(as.character(day_hour_counts$Group.2))

# plot heat mat with ggplot

ggplot(day_hour_counts, aes(x = hour, y = Group.1)) + geom_tile(aes(fill = x)) + scale_fill_gradient(name="Average Counts", low="white", high="green") + theme(axis.title.y = element_blank())



day_hour_casual <- as.data.frame(aggregate(df[,"casual"], list(df$dayofweek, df$hour), mean))
day_hour_casual$Group.1 <- factor(day_hour_casual$Group.1, ordered=TRUE, levels=0:6,labels=c("Monday","Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))

day_hour_casual$hour <- as.numeric(as.character(day_hour_casual$Group.2))

# plot heat mat with ggplot

ggplot(day_hour_casual, aes(x = hour, y = Group.1)) + geom_tile(aes(fill = x)) + scale_fill_gradient(name="Average Counts", low="white", high="green") + theme(axis.title.y = element_blank())



day_hour_reg <- as.data.frame(aggregate(df[,"registered"], list(df$dayofweek, df$hour), mean))
day_hour_reg$Group.1 <- factor(day_hour_reg$Group.1, ordered=TRUE, levels=0:6,labels=c("Monday","Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))

day_hour_reg$hour <- as.numeric(as.character(day_hour_reg$Group.2))

# plot heat mat with ggplot

ggplot(day_hour_reg, aes(x = hour, y = Group.1)) + geom_tile(aes(fill = x)) + scale_fill_gradient(name="Average Counts", low="white", high="green") + theme(axis.title.y = element_blank())


print('holiday')
t.test(df$count~df$holiday)
print('workingday')
t.test(df$count~df$workingday)
seasons <- list('spring' = df[df$season == 1,c('season','count')],'summer' = df[df$season == 2,c('season','count')],'autumn' = df[df$season == 3,c('season','count')],'winter' = df[df$season == 4,c('season','count')])
 
for(i in seasons){
  for(j in seasons){
    i.int <- as.numeric(as.character(i$season[1]))
    j.int <- as.numeric(as.character(j$season[1]))
    if(i.int < j.int){
      print(paste(names(seasons)[i.int],names(seasons)[j.int],sep=":"))
      print(t.test(i$count,j$count))
    }
  }
}



#regression model
library(MASS)

resid.plots <- function(lm){
  res = resid(lm)
  d <- plot(density(res)) #A density plot
  n <- qqnorm(res) # A quantile normal plot - good for checking normality
  q <- qqline(res)
  return(c(d,n,q))
}


y <- 'count'

m <- lm(get(y)~weather+temp+humidity+windspeed+hour+month, data=df)
summary(m)
plot(m)
#res vs fitted: wrong equation, inconstant variances
#qqplot: not normal residuals

df[,'fitted'] = m$fitted.values
ggplot(df, aes_string(x=y,y='fitted'))+geom_point()
#m1.plots <- resid.plots(m1)

df$y.ln = log(df[,y]+1)
ggplot(df, aes(y.ln)) + geom_density()
savepng('count')

df.1 <- df[-c(5632),]
m <- lm(y.ln~weather+temp+humidity+hour+month,data=df.2)
summary(m)

plot(m)
df.1[,'fitted'] = m$fitted.values
ggplot(df.1, aes_string(x=y,y='fitted'))+geom_point()

test = read.table('../input/test_processed.csv',header=TRUE,sep=",")
test$season = as.factor(test$season)
test$holiday = as.factor(test$holiday)
test$workingday = as.factor(test$workingday)
test$weather = as.factor(test$weather)
test$weather[test$weather==4]=3

test.raw = read.table('../input/test.csv',header=TRUE,sep=",")

library(randomForest)
library(Metrics)

rf <- randomForest(y.ln~weather+atemp+humidity+hour+month+holiday+workingday+windspeed+season,data=df,mtry=3,ntree=1000)
print(rf)

pred = data.frame(datetime=test.raw$datetime,count=0)
pred$count <- round(exp(predict(rf,test[,c('weather','atemp','humidity','hour','month','holiday','workingday','windspeed','season')])))
write.table(pred,file="res.csv",sep=',',quote=FALSE,row.names=FALSE)
