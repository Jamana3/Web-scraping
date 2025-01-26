import sqlite3
from scraping import StockScraper, GoldScraper
from news import NewsSentimentAnalyzer
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

API_KEY = "500a25c6dd7045db95bdb728a2c1eea6"
KEYWORDS = ['war', 'election', 'disaster']


amazon_scraper = StockScraper(days=30)
gold_scraper = GoldScraper(days=30)
news_analyzer = NewsSentimentAnalyzer(API_KEY, KEYWORDS)

# to fetch data
amazon_data = amazon_scraper.get_data()
gold_data = gold_scraper.get_data()
news_analyzer.fetch_news_data()
news_analyzer.calculate_sentiment()
daily_sentiment, keyword_counts = news_analyzer.get_daily_summary()

# Rename columns to ensure consistency
amazon_data.rename(columns={'close': 'amzn_close'}, inplace=True)
gold_data.rename(columns={'close': 'gold_close'}, inplace=True)

# to merge stock, gold, and news sentiment data
merged_financial_data = pd.merge(amazon_data, gold_data, on="date", how="outer")
sentiment_data = daily_sentiment.add_suffix('_sentiment').join(keyword_counts.add_suffix('_count'))
combined_data = pd.merge(merged_financial_data, sentiment_data, on="date", how="outer")

# Sort data by date to ensure forward fill works correctly
combined_data = combined_data.sort_values(by="date").reset_index(drop=True)

# Forward fill and backfill stock and gold prices for days when the market is closed
combined_data['amzn_close'] = combined_data['amzn_close'].ffill().bfill()
combined_data['gold_close'] = combined_data['gold_close'].ffill().bfill()

# Calculate technical indicators (7-day and 14-day moving averages)
combined_data['amzn_7_day_avg'] = combined_data['amzn_close'].rolling(window=7, min_periods=1).mean()
combined_data['amzn_14_day_avg'] = combined_data['amzn_close'].rolling(window=14, min_periods=1).mean()

model_data = combined_data[[
    'date', 'amzn_close', 'gold_close', 'amzn_7_day_avg', 'amzn_14_day_avg',
    'war_sentiment', 'election_sentiment', 'disaster_sentiment',
    'war_count', 'election_count', 'disaster_count'
]]

# to connect to SQLite database and store data
conn = sqlite3.connect("financial_sentiment_data.db")
c = conn.cursor()

# to create a table for model training if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS model_data (
        date TEXT PRIMARY KEY,
        amzn_close REAL,
        gold_close REAL,
        amzn_7_day_avg REAL,
        amzn_14_day_avg REAL,
        war_sentiment REAL,
        election_sentiment REAL,
        disaster_sentiment REAL,
        war_count INTEGER,
        election_count INTEGER,
        disaster_count INTEGER
    )
''')

# Insert the refined data into the new table
model_data.reset_index(drop=True, inplace=True)
model_data.to_sql('model_data', conn, if_exists='replace', index=False)

# Commit the transaction and close
conn.commit()
conn.close()

# Load data for modeling
conn = sqlite3.connect("financial_sentiment_data.db")
query = "SELECT * FROM model_data"
df = pd.read_sql(query, conn)
conn.close()

# to ensure data is sorted by date
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by='date', inplace=True)

# Feature engineering: Lagged features for next-day prediction
df['amzn_close_lag1'] = df['amzn_close'].shift(1)
df['gold_close_lag1'] = df['gold_close'].shift(1)

# Handle missing values in lagged columns without inplace=True
df['amzn_close_lag1'] = df['amzn_close_lag1'].fillna(df['amzn_close'].mean())
df['gold_close_lag1'] = df['gold_close_lag1'].fillna(df['gold_close'].mean())

# Define features (X) and target variables (y)
features = [
    'amzn_close_lag1', 'gold_close_lag1', 'amzn_7_day_avg', 'amzn_14_day_avg',
    'war_sentiment', 'election_sentiment', 'disaster_sentiment',
    'war_count', 'election_count', 'disaster_count'
]

# Ensure there are no NaNs in the features and target variables
X = df[features].fillna(0)  # Fill any remaining NaNs in features with 0
y_amzn = df['amzn_close'].fillna(df['amzn_close'].mean())  # Fill NaNs in target variable with mean
y_gold = df['gold_close'].fillna(df['gold_close'].mean())  # Fill NaNs in target variable with mean

# Split data into training and testing sets
X_train, X_test, y_amzn_train, y_amzn_test = train_test_split(X, y_amzn, test_size=0.2, shuffle=False)
_, _, y_gold_train, y_gold_test = train_test_split(X, y_gold, test_size=0.2, shuffle=False)

# Initialize XGBoost models for Amazon and gold predictions
amzn_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gold_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Train the models
amzn_model.fit(X_train, y_amzn_train)
gold_model.fit(X_train, y_gold_train)

# Make predictions for the test set
y_amzn_pred = amzn_model.predict(X_test)
y_gold_pred = gold_model.predict(X_test)

# Add predictions and actual values to the test set for better visualization
X_test = X_test.copy()  # To avoid any potential SettingWithCopyWarning
X_test['predicted_amzn_close'] = y_amzn_pred
X_test['predicted_gold_close'] = y_gold_pred
X_test['actual_amzn_close'] = y_amzn_test.values
X_test['actual_gold_close'] = y_gold_test.values

# Evaluate the models
amzn_mae = mean_absolute_error(y_amzn_test, y_amzn_pred)
gold_mae = mean_absolute_error(y_gold_test, y_gold_pred)
amzn_rmse = root_mean_squared_error(y_amzn_test, y_amzn_pred)
gold_rmse = root_mean_squared_error(y_gold_test, y_gold_pred)

print(f"Amazon Stock Prediction MAE: {amzn_mae:.2f}")
print(f"Amazon Stock Prediction RMSE: {amzn_rmse:.2f}")
print(f"Gold Price Prediction MAE: {gold_mae:.2f}")
print(f"Gold Price Prediction RMSE: {gold_rmse:.2f}")

# Make a prediction for the next day based on the last available data in X
last_row = X.iloc[[-1]]  # Use the last row in the features for the "next day" prediction
next_day_amzn_pred = amzn_model.predict(last_row)
next_day_gold_pred = gold_model.predict(last_row)

# Display the next day predictions
print("\nNext Day Predicted Stock and Gold Prices:")
print(f"Predicted Amazon Stock Price: {next_day_amzn_pred[0]:.2f}")
print(f"Predicted Gold Price: {next_day_gold_pred[0]:.2f}")

# Configure Pandas to display all columns and rows in the terminal
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.width', 1000)        # Set the width of the display

# Display features, Amazon close, and gold close prices in one table
display_table = df[['date'] + features + ['amzn_close', 'gold_close']]
print("\nFeature Values with Amazon and Gold Closing Prices:")
print(display_table)

#Display the updated test set table, including features, actual prices, and predicted prices
print("\nTest Set with Predictions:")
print(X_test)
