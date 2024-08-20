from gnews import GNews
import pandas as pd
from typing import Optional
from langchain_community.document_loaders import WebBaseLoader
from datetime import datetime
import os


def google_news_scraper(tickr):
    google_news = GNews(language='en', country='US', period='1d')
    news = google_news.get_news(tickr)
    news_scrapper = pd.DataFrame(news)
    news_scrapper.sort_values(by=['published date'], ascending=False, inplace=True)
    return news_scrapper

def get_news_content(url):
    try:
        return WebBaseLoader(WebBaseLoader(url).load()[0].page_content.split("Google NewsOpening ")[1]).load()[0].page_content
    except Exception as e:
        print(f"Error retrieving content: {e}")
        return "Dummy content"

today_date = datetime.now().strftime('%Y-%m-%d')
post_df = google_news_scraper("Large Language Models")
post_df['news_content'] = post_df['url'].apply(lambda x: get_news_content(x)) 
directory_path = f'news/{today_date}'
#os.makedirs(name=directory_path)

post_df.to_csv(directory_path+'/gnews.csv')
