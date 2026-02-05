"""
MarketSpeedⅡ MSGETCHART関数を使用した分足データ収集システム
- ExcelのMSGETCHART関数で分足データを一括取得
- PythonでExcelから結果を読み取り、SQLiteに保存
- シンプルで効率的な設計
"""

import pandas as pd
import win32com.client
import sqlite3
import logging
import time
from datetime import datetime
from typing import List, Optional
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_data_collector.log'),
        logging.StreamHandler()
    ]
)

class SimpleDataCollector:
    """
    シンプルな分足データ収集システム
    - MSGETCHART関数を使用した効率的なデータ取得
    - SQLiteデータベースに直接保存
    - 最小限の設定で最大の効果
    """
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.excel_app = None
        self.workbook = None
        self.chart_sheet = None
        self.symbols = []
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"data_collector_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # データベース初期化
        self.initialize_database()
    
    def initialize_database(self):
        """SQLiteデータベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # シンプルな分足データテーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chart_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                datetime TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, datetime, timeframe)
            )
        ''')
        
        # インデックス作成
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_datetime ON chart_data(symbol, datetime)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON chart_data(symbol, timeframe)')
        
        conn.commit()
        conn.close()
        self.logger.info("データベース初期化完了")
        
    def load_symbols(self) -> List[str]:
        """銘柄リスト読み込み"""
        try:
            df = pd.read_csv('prime_symbols.csv')
            # 出来高30万株以上の銘柄を選択
            if 'avg_volume' in df.columns:
                df = df[df['avg_volume'] >= 300000]
            
            self.symbols = df['symbol'].astype(str).tolist()[:50]  # 最初の50銘柄
            self.logger.info(f"対象銘柄数: {len(self.symbols)}銘柄")
            return self.symbols
            
        except FileNotFoundError:
            self.logger.warning("prime_symbols.csvが見つかりません。デフォルト銘柄を使用します")
            self.symbols = ["7203", "9984", "6758", "8306", "1301"]
            return self.symbols
    
    def initialize_excel(self) -> bool:
        """Excel初期化"""
        try:
            self.excel_app = win32com.client.Dispatch("Excel.Application")
            self.excel_app.Visible = False
            self.excel_app.DisplayAlerts = False
            
            # 新しいワークブックを作成
            self.workbook = self.excel_app.Workbooks.Add()
            self.chart_sheet = self.workbook.Worksheets(1)
            self.chart_sheet.Name = "ChartData"
            
            self.logger.info("Excel初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"Excel初期化エラー: {e}")
            return False
    
    def get_chart_data(self, symbol: str, timeframe: str = "5M", count: int = 1000) -> pd.DataFrame:
        """
        MSGETCHART関数を使用して分足データを取得
        """
        try:
            self.logger.info(f"データ取得開始: {symbol} ({timeframe})")
            
            # シートをクリア
            self.chart_sheet.Cells.Clear()
            
            # ヘッダー設定
            headers = ["日時", "始値", "高値", "安値", "終値", "出来高"]
            for i, header in enumerate(headers, 1):
                self.chart_sheet.Cells(1, i).Value = header
            
            # MSGETCHART関数を使用して一括でデータを取得
            # 参考: MarketSpeedⅡ RSSのMSGETCHART関数
            
            # A2セルにMSGETCHART関数を設定
            formula = f'=MSGETCHART("{symbol}","{timeframe}",{count})'
            self.chart_sheet.Cells(2, 1).Formula = formula
            
            # 計算実行
            self.excel_app.Calculate()
            time.sleep(3)  # データ取得完了まで待機
            
            # データを読み取り
            data = []
            row = 2
            
            while row <= count + 1:
                datetime_val = self.chart_sheet.Cells(row, 1).Value
                open_val = self.chart_sheet.Cells(row, 2).Value
                high_val = self.chart_sheet.Cells(row, 3).Value
                low_val = self.chart_sheet.Cells(row, 4).Value
                close_val = self.chart_sheet.Cells(row, 5).Value
                volume_val = self.chart_sheet.Cells(row, 6).Value
                
                # データが有効かチェック
                if (datetime_val and open_val and 
                    not str(datetime_val).startswith('#') and
                    not str(open_val).startswith('#')):
                    
                    data.append({
                        'datetime': datetime_val,
                        'open': float(open_val),
                        'high': float(high_val),
                        'low': float(low_val),
                        'close': float(close_val),
                        'volume': int(volume_val) if volume_val else 0
                    })
                else:
                    break
                
                row += 1
            
            # DataFrameに変換
            df = pd.DataFrame(data)
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')
            
            self.logger.info(f"データ取得完了: {symbol} - {len(df)}件")
            return df
            
        except Exception as e:
            self.logger.error(f"データ取得エラー {symbol}: {e}")
            return pd.DataFrame()
    
    def save_to_database(self, symbol: str, df: pd.DataFrame, timeframe: str = "5M"):
        """データベースに保存"""
        if df.empty:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # データを挿入
            for _, row in df.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO chart_data 
                    (symbol, datetime, timeframe, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, 
                    row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                    timeframe,
                    row['open'], 
                    row['high'], 
                    row['low'], 
                    row['close'], 
                    row['volume']
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"データベース保存完了: {symbol} - {len(df)}件")
            
        except Exception as e:
            self.logger.error(f"データベース保存エラー {symbol}: {e}")
    
    def run_collection(self, timeframe: str = "5M", count: int = 1000):
        """データ収集実行"""
        self.logger.info("=== データ収集開始 ===")
        
        # Excel初期化
        if not self.initialize_excel():
            return
        
        # 銘柄リスト読み込み
        self.load_symbols()
        
        success_count = 0
        
        try:
            for i, symbol in enumerate(self.symbols, 1):
                self.logger.info(f"進捗: {i}/{len(self.symbols)} - {symbol}")
                
                # データ取得
                df = self.get_chart_data(symbol, timeframe, count)
                
                if not df.empty:
                    # データベースに保存
                    self.save_to_database(symbol, df, timeframe)
                    success_count += 1
                
                # 銘柄間の待機
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("ユーザーによる中断")
            
        finally:
            # Excel終了
            if self.excel_app:
                try:
                    self.workbook.Close(SaveChanges=False)
                    self.excel_app.Quit()
                except:
                    pass
        
        self.logger.info(f"=== データ収集完了 ===")
        self.logger.info(f"成功: {success_count}/{len(self.symbols)}銘柄")
    
    def get_data_count(self) -> int:
        """データベース内のデータ件数を取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chart_data")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except:
            return 0

# 使用例
if __name__ == "__main__":
    # シンプルなデータ収集実行
    collector = SimpleDataCollector()
    
    # 5分足データを1000本取得
    collector.run_collection(timeframe="5M", count=1000)
    
    # データ件数確認
    count = collector.get_data_count()
    print(f"データベース内のデータ件数: {count}件")
    
    print("データ収集が完了しました。")