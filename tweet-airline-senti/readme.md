---
# Tweet US Airline Sentiment Analysis

This is [a Kaggle
project](https://www.kaggle.com/crowdflower/twitter-airline-sentiment).
For brief process introduction with code, check [my script on
Kaggle](https://www.kaggle.com/robertq/d/crowdflower/twitter-airline-sentiment/language-model-for-sentiment-analysis/notebook).

Dataset
=======

check <https://www.kaggle.com/crowdflower/twitter-airline-sentiment>

Exploratory analysis
====================

![Sentiment score distribution of different
airlines](/exploratory/plots/airline_sentiscore2.png)


![Changes of average sentiment score in a
week](/exploratory/plots/hour_sentiment.png)


![Daily relevant tweets of different
airlines](/exploratory/plots/hour_sentiment3.png)


![Retweet
distribution](/exploratory/plots/retweet_sentiscore2.png)


Data pre-processing
===================

According to the plots in last section, we basically can treat retweet
count as a binary variable.

Another thing to do is to clean the tweets. Here are the details:

-   Reserve unicode characters only; remove airline mentioned in each
    tweet because we already know it from another data field in the
    dataset;

-   tokenize a tweet into sentences;

-   (Optional) for each sentence, identify Named Entities and remove or
    replace them;

-   Tokenize sentences into words; lowercase each word and stem it;

-   Find the stop words and remove them from each tweet.

![Words frequency
distribution](/exploratory/plots/stopwords.png)


Sentiment analysis
==================

Firstly, the whole dataset is divided by different airlines. Then [A
unigram language
model](https://github.com/yq911122/module/blob/master/lm.py) is
implemented. Then I calculate the average cross validation score.
Finally the top 3 words that help prediction most are identified and
plot.


![](/exploratory/plots/contribute1.png)

![](/exploratory/plots/contribute2.png)

![](/exploratory/plots/contribute3.png)

![](/exploratory/plots/contribute4.png)

![](/exploratory/plots/contribute5.png)

![](/exploratory/plots/contribute6.png)
