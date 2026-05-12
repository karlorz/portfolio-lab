#!/usr/bin/env python3
"""
Fetch missing factor ETF data (QUAL, IJR) for v2.43 Dynamic Factor Timing
"""
import json
import requests
import time
from datetime import datetime

def fetch_yahoo_data(symbol: str, start_date: str = '2005-01-01'):
    """Fetch historical data from Yahoo Finance v8 API"""
    period1 = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    period2 = int(datetime.now().timestamp())
    
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={period1}&period2={period2}&interval=1d"
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        res = requests.get(url, headers=headers, timeout=30)
        res.raise_for_status()
        data = res.json()
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        adjclose = result['indicators']['adjclose'][0]['adjclose']
        
        entries = []
        for ts, price in zip(timestamps, adjclose):
            if price is not None and price > 0:
                date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                entries.append({'d': date, 'p': round(price, 2)})
        
        return entries
    except Exception as e:
        print(f"✗ {symbol}: {e}")
        return None

def main():
    # Load existing prices
    with open('public/data/prices.json', 'r') as f:
        prices = json.load(f)
    
    print(f"Current symbols: {list(prices.keys())}")
    
    # Fetch missing symbols
    missing = ['QUAL', 'IJR']
    
    for symbol in missing:
        if symbol in prices:
            print(f"✓ {symbol} already exists ({len(prices[symbol])} days)")
            continue
        
        print(f"\nFetching {symbol}...")
        data = fetch_yahoo_data(symbol)
        
        if data:
            prices[symbol] = data
            print(f"✓ {symbol}: {len(data)} days ({data[0]['d']} to {data[-1]['d']})")
        else:
            print(f"✗ Failed to fetch {symbol}")
        
        time.sleep(0.5)  # Rate limit
    
    # Save updated prices
    with open('public/data/prices.json', 'w') as f:
        json.dump(prices, f, separators=(',', ':'))
    
    print(f"\n✓ Updated prices.json with {len(prices)} symbols")
    for sym, data in sorted(prices.items()):
        print(f"  {sym}: {len(data)} days ({data[0]['d']} to {data[-1]['d']})")

if __name__ == '__main__':
    main()
