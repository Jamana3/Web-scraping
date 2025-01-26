import requests
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NewsSentimentAnalyzer:
    def __init__(self, api_key, keywords, days=30):
        self.api_key = api_key
        self.keywords = keywords
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=days)
        self.base_url = 'https://newsapi.org/v2/everything'
        self.analyzer = SentimentIntensityAnalyzer()
        self.all_data = pd.DataFrame(columns=['date', 'title', 'description', 'keyword'])

    def fetch_news_data(self):
        for keyword in self.keywords:
            start_date_str = self.start_date.strftime('%Y-%m-%d')
            end_date_str = self.end_date.strftime('%Y-%m-%d')
            params = {
                'q': keyword,
                'from': start_date_str,
                'to': end_date_str,
                'apiKey': self.api_key,
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 100  # Max articles per page
            }

            # Fetch data from API
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                for article in articles:
                    article_date = article.get('publishedAt', '').split("T")[0]
                    title = article.get('title', '')
                    description = article.get('description', '')

                    # Append data to DataFrame
                    self.all_data = pd.concat([
                        self.all_data,
                        pd.DataFrame([[article_date, title, description, keyword]],
                                     columns=['date', 'title', 'description', 'keyword'])
                    ], ignore_index=True)
            else:
                print(f"Failed to fetch data for keyword '{keyword}'")

    def calculate_sentiment(self):
        def get_sentiment(text):
            return self.analyzer.polarity_scores(text)['compound'] if pd.notnull(text) else 0
        self.all_data['title_sentiment'] = self.all_data['title'].apply(get_sentiment)
        self.all_data['description_sentiment'] = self.all_data['description'].apply(get_sentiment)
        self.all_data['average_sentiment'] = self.all_data[['title_sentiment', 'description_sentiment']].mean(axis=1)

    def get_daily_summary(self):
        self.all_data['date'] = pd.to_datetime(self.all_data['date'])
        self.all_data = self.all_data.sort_values('date').reset_index(drop=True)
        daily_sentiment = self.all_data.groupby(['date', 'keyword'])['average_sentiment'].mean().unstack(fill_value=0)
        keyword_counts = self.all_data.groupby(['date', 'keyword']).size().unstack(fill_value=0)

        return daily_sentiment, keyword_counts
