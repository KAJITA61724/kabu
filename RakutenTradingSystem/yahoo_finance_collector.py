"""
代替データ収集システム - Yahoo Finance API使用
MarketSpeedⅡが利用できない場合の代替手段
"""

import yfinance as yf
import pandas as pd
import sqlite3
import logging
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class YahooFinanceDataCollector:
    """Yahoo Finance APIを使用したデータ収集"""
    
    def __init__(self, db_path: str = "enhanced_trading.db"):
        self.db_path = db_path
        self.symbols = []
        
    def load_symbols(self) -> List[str]:
        """銘柄リスト読み込み"""
        try:
            # プライム銘柄リスト読み込み
            df = pd.read_csv('prime_symbols.csv')
            
            # 30万株以上の銘柄フィルタ
            if 'avg_volume' in df.columns:
                filtered_df = df[df['avg_volume'] >= 300000]
            else:
                filtered_df = df
            
            # Yahoo Finance形式に変換（日本株は.T付加）
            symbols = []
            for symbol in filtered_df['symbol'].astype(str):
                yahoo_symbol = f"{symbol}.T"
                symbols.append(yahoo_symbol)
            
            self.symbols = symbols
            logging.info(f"Yahoo Finance対象銘柄: {len(self.symbols)}銘柄")
            return self.symbols
            
        except FileNotFoundError:
            logging.error("prime_symbols.csvが見つかりません")
            return []
    
    def get_single_stock_data(self, symbol: str) -> Optional[Dict]:
        """単一銘柄のデータ取得"""
        try:
            # yfinanceでデータ取得
            ticker = yf.Ticker(symbol)
            
            # 基本情報取得
            info = ticker.info
            
            # 1分足データ取得（最新）
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                return None
            
            # 最新データ取得
            latest = hist.iloc[-1]
            
            # 日本株コード抽出（.T を削除）
            stock_code = symbol.replace('.T', '')
            
            data = {
                'symbol': stock_code,
                'timestamp': datetime.now().replace(
                    minute=(datetime.now().minute // 5) * 5,
                    second=0,
                    microsecond=0
                ),
                'open_price': float(latest['Open']),
                'high_price': float(latest['High']),
                'low_price': float(latest['Low']),
                'close_price': float(latest['Close']),
                'volume': int(latest['Volume']),
                'turnover': float(latest['Close'] * latest['Volume']),
                'vwap': float(latest['Close']),  # Yahoo Financeには正確なVWAPがないので現在値で代用
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0)
            }
            
            return data
            
        except Exception as e:
            logging.error(f"データ取得エラー {symbol}: {e}")
            return None
    
    def collect_batch_data(self, max_workers: int = 20) -> List[Dict]:
        """バッチデータ収集"""
        collected_data = []
        
        logging.info(f"Yahoo Finance データ収集開始: {len(self.symbols)}銘柄")
        
        # 並列処理でデータ取得
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 全銘柄のタスクを開始
            future_to_symbol = {
                executor.submit(self.get_single_stock_data, symbol): symbol 
                for symbol in self.symbols
            }
            
            # 結果を収集
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data:
                        collected_data.append(data)
                        if len(collected_data) % 10 == 0:
                            logging.info(f"進捗: {len(collected_data)}/{len(self.symbols)} 銘柄完了")
                except Exception as e:
                    logging.error(f"データ処理エラー {symbol}: {e}")
        
        logging.info(f"Yahoo Finance データ収集完了: {len(collected_data)}銘柄")
        return collected_data
    
    def save_data(self, data_list: List[Dict]):
        """データ保存"""
        if not data_list:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        
        for data in data_list:
            try:
                # 5分足データ保存
                cursor.execute('''
                    INSERT OR REPLACE INTO five_minute_data 
                    (symbol, timestamp, open_price, high_price, low_price, 
                     close_price, volume, turnover, vwap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['symbol'], data['timestamp'], data['open_price'],
                    data['high_price'], data['low_price'], data['close_price'],
                    data['volume'], data['turnover'], data['vwap']
                ))
                
                saved_count += 1
                
            except Exception as e:
                logging.error(f"データ保存エラー {data['symbol']}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        logging.info(f"Yahoo Finance データ保存完了: {saved_count}銘柄")
    
    def run_collection(self):
        """データ収集実行"""
        # 銘柄読み込み
        if not self.symbols:
            self.load_symbols()
        
        if not self.symbols:
            logging.error("銘柄データが取得できませんでした")
            return
        
        # データ収集
        collected_data = self.collect_batch_data()
        
        # データ保存
        self.save_data(collected_data)
        
        return len(collected_data)

# 使用例
if __name__ == "__main__":
    print("=== Yahoo Finance データ収集システム ===")
    print("MarketSpeedⅡの代替データソースとして使用")
    
    collector = YahooFinanceDataCollector()
    result_count = collector.run_collection()
    
    print(f"収集完了: {result_count}銘柄のデータを取得しました")
    
    # データベース確認
    conn = sqlite3.connect("enhanced_trading.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM five_minute_data")
    total_records = cursor.fetchone()[0]
    conn.close()
    
    print(f"データベース総件数: {total_records}件")
