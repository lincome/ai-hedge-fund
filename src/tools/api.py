import os
import pandas as pd
import requests
import json
import logging

logger = logging.getLogger("get_cache")
logging.basicConfig(filename='portfolio_debug.log', level=logging.ERROR,format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

from data.cache import get_cache
from data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
)

# Global cache instance
_cache = get_cache()


def save_json_to_txt(data, filename='data.txt'):
    """
    发送API请求并将JSON响应保存到TXT文件
    """
    try:
        # 发送API请求
        # response = requests.get(url)
        # response.raise_for_status()  # 检查请求是否成功
        
        # # 解析JSON数据
        # data = response.json()
        
        # 将JSON数据保存到TXT文件
        with open(filename, 'w', encoding='utf-8') as f:
            try:
                json.dump(data, f, indent=4, ensure_ascii=False)
            except TypeError as e:
                print(f"Data serialization error: {e}")
            
        print(f"数据已保存到 {filename}")
        return True

    except json.JSONDecodeError:
        print("响应内容不是有效的JSON格式")
        return False
        
def read_json_from_txt(filename='data.txt'):
    """
    从TXT文件读取JSON数据
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
        return None
    except json.JSONDecodeError:
        print(f"文件 {filename} 包含无效的JSON格式")
        return None

def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or API."""
    # Check cache first
    txt_cached_data = read_json_from_txt('data/price_'+ticker+start_date+end_date+'.txt')
    if txt_cached_data:
        # Convert cached data to Price objects
        price_response = PriceResponse(**txt_cached_data)
        prices = price_response.prices
        return prices
    
    # if cached_data := _cache.get_prices(ticker):
    #     # Filter cached data by date range and convert to Price objects
    #     filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
    #     if filtered_data:
    #         return filtered_data
    print (1222222222222)
    # If not in cache or no data in range, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    price_response = PriceResponse(**response.json())
    prices = price_response.prices

    if not prices:
        return []
    save_json_to_txt(response.json(), filename='data/price_'+ticker+start_date+end_date+'.txt')
    # Cache the results as dicts
    # _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""

    txt_cached_data = read_json_from_txt('data/financial_metrics_'+ticker+end_date+'.txt')
    if txt_cached_data:
        # Convert cached data to Price objects
        price_response = FinancialMetricsResponse(**txt_cached_data)
        prices = price_response.financial_metrics
        return prices
    
    # Check cache first
    # cached_data = _cache.get_financial_metrics(ticker)
    # logger.error("get_financial_metrics ticker: %s,end_date: %s,period: %s,cached_data: %s",ticker,end_date,period,cached_data)
    # if cached_data:
    #     # Filter cached data by date and limit
    #     filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
    #     filtered_data.sort(key=lambda x: x.report_period, reverse=True)
    #     if filtered_data:
    #         return filtered_data[:limit]

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    metrics_response = FinancialMetricsResponse(**response.json())
    # Return the FinancialMetrics objects directly instead of converting to dict
    financial_metrics = metrics_response.financial_metrics

    if not financial_metrics:
        return []
    save_json_to_txt(response.json(), filename='data/financial_metrics_'+ticker+end_date+'.txt')
    # Cache the results as dicts
    # _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    agents: str,
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items from API."""

    txt_cached_data = read_json_from_txt('data/search_line_items_'+agents+ticker+end_date+'.txt')
    if txt_cached_data:
        # Convert cached data to Price objects
        price_response = LineItemResponse(**txt_cached_data)
        prices = price_response.search_results
        return prices[:limit]
    
    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    data = response.json()
    response_model = LineItemResponse(**data)
    search_results = response_model.search_results
    if not search_results:
        return []

    save_json_to_txt(response.json(), filename='data/search_line_items_'+agents+ticker+end_date+'.txt')
    # Cache the results
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""

    txt_cached_data = read_json_from_txt('data/get_insider_trades_'+ticker+end_date+'.txt')
    if txt_cached_data:
        # print (txt_cached_data)
        return [InsiderTrade(**item) for item in txt_cached_data]
    
    # Check cache first
    # if cached_data := _cache.get_insider_trades(ticker):
    #     # Filter cached data by date range
    #     filtered_data = [InsiderTrade(**trade) for trade in cached_data 
    #                     if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
    #                     and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
    #     filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
    #     if filtered_data:
    #         return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades
        
        if not insider_trades:
            break
            
        all_trades.extend(insider_trades)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break
            
        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []
    
    data_dict = [trade.dict() for trade in all_trades]
    # print (all_trades)
    save_json_to_txt(data_dict, filename='data/get_insider_trades_'+ticker+end_date+'.txt')
    # Cache the results
    # _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""

    txt_cached_data = read_json_from_txt('data/get_company_news_'+ticker+end_date+'.txt')
    if txt_cached_data:
         return [CompanyNews(**item) for item in txt_cached_data]
    
    # Check cache first
    # if cached_data := _cache.get_company_news(ticker):
    #     # Filter cached data by date range
    #     filtered_data = [CompanyNews(**news) for news in cached_data 
    #                     if (start_date is None or news["date"] >= start_date)
    #                     and news["date"] <= end_date]
    #     filtered_data.sort(key=lambda x: x.date, reverse=True)
    #     if filtered_data:
    #         return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news
        
        if not company_news:
            break
            
        all_news.extend(company_news)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break
            
        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []
    data_dict = [trade.dict() for trade in all_news]
    save_json_to_txt(data_dict, filename='data/get_company_news_'+ticker+end_date+'.txt')
    # Cache the results
    # _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news



def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap from the API."""
    financial_metrics = get_financial_metrics(ticker, end_date)
    market_cap = financial_metrics[0].market_cap
    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
