from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from datetime import datetime, timedelta
import time


class StockScraper:
    def __init__(self, days=30, retries=3, wait_time=10):
        self.days = days
        self.url = f"https://finance.yahoo.com/quote/AMZN/history/"
        self.data = []
        self.retries = retries
        self.wait_time = wait_time

    def start_driver(self):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_experimental_option("detach", True)
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get(self.url)

    def scrape_data(self):
        cutoff_date = datetime.now() - timedelta(days=self.days)
        attempt = 0

        while attempt < self.retries:
            try:
                self.start_driver()

                table = WebDriverWait(self.driver, self.wait_time).until(
                    EC.presence_of_element_located(
                        (By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/div[1]/div[3]/table'))
                )
                rows = table.find_elements(By.TAG_NAME, "tr")

                for row in rows:
                    columns = row.find_elements(By.TAG_NAME, "td")
                    if columns:
                        date_str = columns[0].text
                        try:
                            date = datetime.strptime(date_str, '%b %d, %Y')
                        except ValueError:
                            continue

                        if date >= cutoff_date:
                            open_price = columns[1].text
                            high_price = columns[2].text
                            low_price = columns[3].text
                            close_price = columns[4].text
                            volume = columns[6].text

                            self.data.append({
                                "date": date,
                                "open": float(open_price.replace(",", "")),
                                "high": float(high_price.replace(",", "")),
                                "low": float(low_price.replace(",", "")),
                                "close": float(close_price.replace(",", "")),
                                "volume": int(volume.replace(",", ""))
                            })
                        else:
                            break

                if self.data:
                    break

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                attempt += 1
                time.sleep(2)  # Wait before retrying

            finally:
                self.driver.quit()

    def get_data(self):
        if not self.data:
            self.scrape_data()

        stock_df = pd.DataFrame(self.data)
        stock_df = stock_df.sort_values("date").reset_index(drop=True)
        stock_df['7_day_avg'] = stock_df['close'].rolling(window=7).mean()
        stock_df['14_day_avg'] = stock_df['close'].rolling(window=14).mean()

        return stock_df


class GoldScraper:
    def __init__(self, days=30, retries=3, wait_time=10):
        self.days = days
        self.url = "https://finance.yahoo.com/quote/GC=F/history/"
        self.data = []
        self.retries = retries
        self.wait_time = wait_time

    def start_driver(self):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_experimental_option("detach", True)
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get(self.url)

    def scrape_data(self):
        cutoff_date = datetime.now() - timedelta(days=self.days)
        attempt = 0

        while attempt < self.retries:
            try:
                self.start_driver()

                table = WebDriverWait(self.driver, self.wait_time).until(
                    EC.presence_of_element_located(
                        (By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/div[1]/div[3]/table'))
                )
                rows = table.find_elements(By.TAG_NAME, "tr")
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 5:
                        date_str = cells[0].text
                        close_price = cells[4].text
                        try:
                            date = datetime.strptime(date_str, '%b %d, %Y')
                        except ValueError:
                            continue

                        if date >= cutoff_date:
                            self.data.append({
                                "date": date,
                                "gold_close": float(close_price.replace(",", ""))
                            })
                        else:
                            break
                if self.data:
                    break

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                attempt += 1
                time.sleep(2)  # Wait before retrying

            finally:
                self.driver.quit()

    def get_data(self):
        if not self.data:
            self.scrape_data()

        gold_df = pd.DataFrame(self.data)
        gold_df = gold_df.sort_values("date").reset_index(drop=True)
        return gold_df
