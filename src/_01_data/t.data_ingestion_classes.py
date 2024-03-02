import requests
import yfinance as yf
from bs4 import BeautifulSoup

# Base class

class DataIngestor:
    def __init__(self, strategy):
        self.strategy = strategy

    def ingest_data(self, **kwargs):
        return self.strategy.ingest_data(**kwargs)

# Subclasses

class DataIngestionStrategy:
    def ingest_data(self, **kwargs):
        raise NotImplementedError

class WebsiteDataIngestionStrategy(DataIngestionStrategy):
    def ingest_data(self, urls=None):
        try:
            if urls is None:
                raise ValueError("URLs required for web scraping")
            data = {}
            for url in urls:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                # Implement logic to scrape data from the webpage
                # Example: Extracting title and links
                titles = soup.find_all('h2')
                links = [a['href'] for a in soup.find_all('a', href=True)]
                data[url] = {'titles': [title.text for title in titles], 'links': links}
            return data
        except Exception as e:
            print(f"Failed to ingest data from web scraping: {str(e)}")

class APIDataIngestionStrategy(DataIngestionStrategy):
    def ingest_data(self, endpoints=None):
        try:
            if endpoints is None:
                raise ValueError("API endpoints required for data ingestion")
            data = {}
            for endpoint in endpoints:
                response = requests.get(endpoint)
                data[endpoint] = response.json()
            return data
        except Exception as e:
            print(f"Failed to ingest data from API: {str(e)}")

class YfinanceDataIngestionStrategy(DataIngestionStrategy):
    def ingest_data(self, libraries=None):
        try:
            if libraries is None:
                raise ValueError("Libraries required for data ingestion")
            data = {}
            for library in libraries:
                # Implement logic to retrieve data using the library
                # Example: Using yfinance to fetch stock data
                data[library] = yf.download(library, period='max', interval='1d', prepost=True, group_by="ticket")
            return data
        except Exception as e:
            print(f"Failed to ingest data from library: {str(e)}")


# Factory Class

class DataIngestorFactory:
    @staticmethod
    def create_ingestor(source_type):
        if source_type == 'website':
            return DataIngestor(WebsiteDataIngestionStrategy())
        elif source_type == 'api':
            return DataIngestor(APIDataIngestionStrategy())
        elif source_type == 'yfinance_library':
            return DataIngestor(YfinanceDataIngestionStrategy())
        else:
            raise ValueError("Invalid source type")


# Execution
yfinance_ingestor = DataIngestorFactory.create_ingestor(source_type='yfinance_library')
yfinance_data = yfinance_ingestor.ingest_data(libraries=['SPY AAPL'])
print(yfinance_data)