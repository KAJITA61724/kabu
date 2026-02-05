#!/usr/bin/env python3
"""
自動データ収集システム - yfinanceベース
平日の取引時間中に5分足データを自動収集
"""

import yfinance as yf
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
import jpholiday
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AutomatedDataCollector:
    def __init__(self, db_path=None):
        # 実行場所に応じて適切なパスを選択
        if db_path is None:
            import os
            possible_paths = [
                'data/market_data.db',          # RakutenTradingSystemから実行
                '../data/market_data.db',       # data_collectionから実行
            ]
            for path in possible_paths:
                dir_path = os.path.dirname(path)
                if dir_path and not os.path.exists(dir_path):
                    continue
                db_path = path
                break
            if db_path is None:
                db_path = 'data/market_data.db'  # フォールバック
                
        self.db_path = db_path
        self.symbols = []
        self._ensure_database()
        
    def _ensure_database(self):
        """データベースとテーブルを初期化"""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chart_data_1min (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                datetime TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                collected_at TEXT,
                UNIQUE(symbol, datetime)
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol_datetime ON chart_data_1min(symbol, datetime)')
        conn.commit()
        conn.close()
        logging.info(f"データベース準備完了: {self.db_path}")
        
    def load_symbols(self):
        """prime_symbols.csvから銘柄リストを読み込み"""
        import os
        
        # 複数のパスを試す（実行場所に応じて）
        possible_paths = [
            'prime_symbols.csv',          # RakutenTradingSystemから実行
            '../prime_symbols.csv',       # data_collectionから実行
            'RakutenTradingSystem/prime_symbols.csv'  # ルートから実行
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if not csv_path:
            logging.error(f"銘柄リストが見つかりません。試したパス: {possible_paths}")
            logging.error(f"現在のディレクトリ: {os.getcwd()}")
            return False
            
        try:
            df = pd.read_csv(csv_path)
            if 'avg_volume' in df.columns:
                df = df[df['avg_volume'] >= 300000]
            self.symbols = [str(s) + '.T' for s in df['symbol'].tolist()]
            logging.info(f"対象銘柄: {len(self.symbols)}銘柄 (from {csv_path})")
            return True
        except Exception as e:
            logging.error(f"銘柄リスト読み込みエラー: {e}")
            return False
    
    def collect_single_symbol(self, symbol):
        """1銘柄のデータを収集"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='5d', interval='1m')
            
            if df.empty:
                return None
                
            df['symbol'] = symbol.replace('.T', '')
            df.reset_index(inplace=True)
            df.rename(columns={'Datetime': 'datetime'}, inplace=True)
            
            # 必要なカラムのみ
            result = df[['symbol', 'datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            result.columns = ['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']
            result['datetime'] = result['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            result['adj_close'] = result['close']
            
            return result
            
        except Exception as e:
            logging.warning(f"{symbol}: {str(e)}")
            return None
    
    def save_data(self, df):
        """データをSQLiteに保存（重複チェック付き）"""
        if df is None or df.empty:
            return 0
            
        conn = sqlite3.connect(self.db_path)
        
        # 重複を避けて挿入
        inserted = 0
        for _, row in df.iterrows():
            try:
                conn.execute('''
                    INSERT OR IGNORE INTO chart_data_1min 
                    (symbol, datetime, open, high, low, close, volume, adj_close, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['symbol'], row['datetime'], 
                    row['open'], row['high'], row['low'], row['close'], 
                    row['volume'], row['adj_close'], 
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
                inserted += 1
            except:
                pass
                
        conn.commit()
        conn.close()
        return inserted
    
    def cleanup_old_data(self, days=90):
        """古いデータを削除（90日以上前）"""
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            conn.execute('''
                DELETE FROM chart_data_5min 
                WHERE datetime < ?
            ''', (cutoff_date,))
            
            deleted = conn.total_changes
            conn.commit()
            conn.close()
            
            if deleted > 0:
                logging.info(f"古いデータを削除: {deleted}件")
                
        except Exception as e:
            logging.error(f"クリーンアップエラー: {e}")
    
    def run_collection(self):
        """データ収集を実行"""
        now = datetime.now()
        
        # 平日かつ取引時間内かチェック（JST 9:00-15:00）
        if now.weekday() >= 5:
            logging.info("週末のためスキップ")
            return
            
        if jpholiday.is_holiday(now.date()):
            logging.info("祝日のためスキップ")
            return
        
        logging.info("=" * 60)
        logging.info(f"データ収集開始: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not self.load_symbols():
            return
        
        total_inserted = 0
        success_count = 0
        
        for i, symbol in enumerate(self.symbols, 1):
            if i % 50 == 0:
                logging.info(f"進捗: {i}/{len(self.symbols)}")
                
            df = self.collect_single_symbol(symbol)
            if df is not None:
                inserted = self.save_data(df)
                total_inserted += inserted
                success_count += 1
        
        logging.info(f"✅ 収集完了: {success_count}銘柄, {total_inserted}件のデータを保存")
        
        # 古いデータのクリーンアップ
        self.cleanup_old_data()
        
        # 統計情報
        conn = sqlite3.connect(self.db_path)
        total = pd.read_sql('SELECT COUNT(*) as c FROM chart_data_5min', conn).iloc[0]['c']
        conn.close()
        
        logging.info(f"データベース総件数: {total:,}件")
        logging.info("=" * 60)

if __name__ == '__main__':
    collector = AutomatedDataCollector()
    collector.run_collection()
