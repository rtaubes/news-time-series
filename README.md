# news-time-series

The code shows that it is possible to find news topics which have a positive correlation with a stock price within a time frame. Using the Reuters corpus from NLTK, historical news and stock prices for the second half of 2017, it was found that there is a weak positive correlation between a topic and a price. The correlation has a maximum when a delay between topic and price is 1-3 days. Only a news title is used to determine how positive or negative the news is.

See full description the LinkedIn article "".

### sklearn_lda.py
The script sklearn_lda.py is used to train a Latent Dirichlet Allocation Algorithm on a text corpus. The scikit-learn implementation is of LDA and tokenizer is used in the script.<br>
The script requires a JSON configation. See the examples in the 'configs' folder.
A configuration includes description of dataset, and where a model, tokenizer, and tokens are saved.
'Config' includes examples for:

- NLTK reuters corpus using 90 topics from the corpus,
- NLTK reuters corpus using raw data and custom topics,
- ready to use SQL set of topics from [News Corpus Builder](https://skillachie.github.io/news-corpus-builder/)
  and Ready to used corpora [finance_corpus.db](https://github.com/skillachie/binaryNLP) of the same author.
topic selection for time-series prediction

The project doesn't include datasets. Use references in the article or your own data. Corpora from NLTK can be downloaded using NLTK API.

### Reuters-2158-v3-kaggle-irish_news.ipynb

The jupyter notebook shows how to make a simple sentiment analysis when news divided into topics. Irish News are used as an example. A filter removes topics when amount of news is too small.
SentimentIntensityAnalyzer from nltk.sentiment.vader determines how positive or negative a particular news is.

### Reuters-2158-v3-kaggle-abc_news.ipynb

This is jupyter notebook shows how to make a simple sentiment analysis when news divided into topics by the LDA model prepared by 'sklearn_lda.py'. ABC News are used as an example. A filter removes topics when amount of news is too small.

### finance-yahoo-irish.ipynb, finance-yahoo-ABC.ipynb

Analysis of correlation between topics and daily time-series of Bank of America and City Group from July 2017 to January 2018.

### Howto

The article uses two types of news:
- Irish News, which are already divided into topics,
- ABC News, which are one set of texts and should be split into topics.

The algorithm for Irish News is:

1. Reuters-2158-v3-kaggle-irish_news.ipynb prepares a data frame where columns are news types, rows are estimations how news are positive.
2. finance-yahoo-irish.ipynb calculates a correlation between a news topic and prices.

ABC News requires additional steps:

1. Run 'sklearn_lda' to train the LDA model. The script saves a model, tokenizer, and vectorizer.
2. Reuters-2158-v3-kaggle-abc_news.ipynb prepares a data frame where columns are news types found by a model, rows are estimations how news are positive.
3. finance-yahoo-ABC.ipynb calculates a correlation between a news topic and prices.



