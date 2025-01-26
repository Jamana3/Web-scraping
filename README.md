# Web-scraping
I made a ml model to predict the next day stock price and gold price based on sone news keywords and existing stock data. I made three different files one is the main file for testing and training the model. 
other is news file to collect news from newsapi.org containing keywords war,election and disaster, either in their title or description. Also calculated average sentiment score for each day for each keyword and article count and stored the fetched data in pandas dataframe and use that in main.py file.
next is scraping in which i collected data for historical stock price(closing price,low,high,opening and volume with data) and Gold price from yahoo using web scraping(selenium).
