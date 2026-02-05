"""
デモ用サンプルデータ生成ツール
過去5営業日分の5分足データを生成してデモトレードを実行可能にする
"""

import sys
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class SampleDataGenerator:
    """サンプルデータ生成クラス"""
    
    def __init__(self, db_path="enhanced_trading.db"):
        self.db_path = db_path
        self.symbols = [
            "7203",  # トヨタ自動車
            "9984",  # ソフトバンクG
            "6758",  # ソニーG
            "8306",  # 三菱UFJ
            "6501",  # 日立製作所
            "4519",  # 中外製薬
            "9432",  # NTT
            "6861",  # キーエンス
            "7974",  # 任天堂
            "4063"   # 信越化学
        ]
        
    def generate_realistic_price_data(self, base_price, days=5):
        """リアルな価格データを生成"""
        # 1日あたり72回（5分足 × 9:00-15:00 = 6時間）
        periods_per_day = 72
        total_periods = days * periods_per_day
        
        # ランダムウォークによる価格生成
        price_changes = np.random.normal(0, 0.002, total_periods)  # 0.2%の標準偏差
        
        # 累積和で価格系列を生成
        cumulative_changes = np.cumsum(price_changes)
        prices = base_price * (1 + cumulative_changes)
        
        # OHLC生成
        data = []
        for i in range(total_periods):
            # 基準価格
            close_price = prices[i]
            
            # 高値・安値の範囲（±0.5%程度）
            high_low_range = close_price * 0.005
            high_price = close_price + random.uniform(0, high_low_range)
            low_price = close_price - random.uniform(0, high_low_range)
            
            # 始値（前回終値ベース）
            if i == 0:
                open_price = base_price
            else:
                open_price = prices[i-1] + random.uniform(-high_low_range/2, high_low_range/2)
            
            # 出来高（100の倍数）
            volume = random.randint(10000, 100000) * 100
            
            data.append({
                'open': round(open_price, 2),
                'high': round(max(open_price, high_price, close_price), 2),
                'low': round(min(open_price, low_price, close_price), 2),
                'close': round(close_price, 2),
                'volume': volume
            })
            
        return data
    
    def generate_timestamps(self, days=5):
        """5営業日分のタイムスタンプを生成（9:00-15:00, 5分間隔）"""
        timestamps = []
        
        # 今日から遡って営業日を取得
        current_date = datetime.now().date()
        business_days = []
        
        check_date = current_date
        while len(business_days) < days:
            # 平日のみ追加
            if check_date.weekday() < 5:  # 月曜=0, 金曜=4
                business_days.append(check_date)
            check_date -= timedelta(days=1)
        
        business_days.reverse()  # 古い順に並べ替え
        
        # 各営業日の5分足タイムスタンプ生成
        for day in business_days:
            # 9:00-15:00 = 6時間 = 360分 = 72回（5分間隔）
            start_time = datetime.combine(day, datetime.min.time().replace(hour=9, minute=0))
            
            for i in range(72):
                timestamp = start_time + timedelta(minutes=i * 5)
                timestamps.append(timestamp)
        
        return timestamps
    
    def populate_database(self):
        """データベースにサンプルデータを投入"""
        print("📊 サンプルデータ生成開始")
        print("=" * 50)
        
        # データベース接続
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # テーブル作成（念のため）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS five_minute_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                turnover REAL,
                vwap REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS moving_averages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                ma_5min REAL,
                ma_20min REAL,
                ma_60min REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # 既存データをクリア
        cursor.execute("DELETE FROM five_minute_data")
        cursor.execute("DELETE FROM moving_averages")
        print("🗑️ 既存データをクリア")
        
        # タイムスタンプ生成
        timestamps = self.generate_timestamps(days=5)
        print(f"📅 {len(timestamps)}個のタイムスタンプを生成")
        
        # 各銘柄のデータ生成
        total_records = 0
        for symbol in self.symbols:
            print(f"📈 {symbol} のデータ生成中...")
            
            # 銘柄ごとの基準価格設定
            base_prices = {
                "7203": 2800,   # トヨタ
                "9984": 5200,   # ソフトバンクG
                "6758": 13000,  # ソニー
                "8306": 1200,   # 三菱UFJ
                "6501": 3500,   # 日立
                "4519": 4800,   # 中外製薬
                "9432": 180,    # NTT
                "6861": 48000,  # キーエンス
                "7974": 6200,   # 任天堂
                "4063": 27000   # 信越化学
            }
            
            base_price = base_prices.get(symbol, 1000)
            price_data = self.generate_realistic_price_data(base_price, days=5)
            
            # データベースに挿入
            for i, (timestamp, ohlc) in enumerate(zip(timestamps, price_data)):
                cursor.execute("""
                    INSERT OR REPLACE INTO five_minute_data 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume, turnover, vwap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, timestamp, ohlc['open'], ohlc['high'], ohlc['low'], 
                    ohlc['close'], ohlc['volume'], 
                    ohlc['close'] * ohlc['volume'],  # turnover
                    ohlc['close']  # vwap（簡易）
                ))
                
                # 移動平均データも追加（簡易版）
                ma_5min = ohlc['close'] * random.uniform(0.995, 1.005)
                ma_20min = ohlc['close'] * random.uniform(0.990, 1.010)
                ma_60min = ohlc['close'] * random.uniform(0.985, 1.015)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO moving_averages 
                    (symbol, timestamp, ma_5min, ma_20min, ma_60min)
                    VALUES (?, ?, ?, ?, ?)
                """, (symbol, timestamp, ma_5min, ma_20min, ma_60min))
                
                total_records += 1
        
        conn.commit()
        conn.close()
        
        print(f"✅ 完了: {total_records}件のデータを生成")
        print(f"📊 銘柄数: {len(self.symbols)}")
        print(f"📅 期間: 5営業日")
        print(f"⏰ 間隔: 5分足")
        print("🎮 デモトレード実行準備完了！")

def main():
    """メイン実行"""
    generator = SampleDataGenerator()
    generator.populate_database()
    
    print("\n" + "=" * 50)
    print("🚀 次の手順:")
    print("1. python rakuten_trading_launcher.py")
    print("2. メニューから「2」を選択（デモトレード）")
    print("3. お好みの方法を選択してデモ実行")
    print("=" * 50)

if __name__ == "__main__":
    main()
