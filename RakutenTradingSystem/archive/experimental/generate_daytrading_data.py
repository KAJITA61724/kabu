"""
デイトレード用サンプルデータ生成
既存データベースからデイトレード分析用のデータを生成
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_detailed_trading_data(db_path="enhanced_trading.db", num_records=1000):
    """詳細なデイトレード用データ生成"""
    
    # 日本の主要銘柄
    symbols = [
        "7203", "9984", "6758", "8306", "6501", "4063", "9432", "8035", 
        "4519", "6861", "8411", "7267", "9983", "4755", "6954"
    ]
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # テーブル作成
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS five_minute_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open_price REAL NOT NULL,
            high_price REAL NOT NULL,
            low_price REAL NOT NULL,
            close_price REAL NOT NULL,
            volume INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 既存データクリア
    cursor.execute("DELETE FROM five_minute_data")
    
    # 基準日時設定
    base_date = datetime.now() - timedelta(days=30)
    
    # 各銘柄のデータ生成
    for symbol in symbols:
        print(f"銘柄 {symbol} のデータ生成中...")
        
        # 銘柄別基準価格設定
        base_prices = {
            "7203": 2800,   # トヨタ
            "9984": 8500,   # ソフトバンク
            "6758": 24000,  # ソニー
            "8306": 4200,   # 三菱UFJ
            "6501": 6500,   # 日立
            "4063": 3800,   # 信越化学
            "9432": 2100,   # NTT
            "8035": 3600,   # 東エレク
            "4519": 4200,   # 中外製薬
            "6861": 9500    # キーエンス
        }
        
        current_price = base_prices.get(symbol, 3000)
        
        # 5分足データ生成（約3週間分）
        for i in range(num_records // len(symbols)):
            timestamp = base_date + timedelta(minutes=i * 5)
            
            # 週末スキップ
            if timestamp.weekday() >= 5:
                continue
                
            # 市場時間外スキップ（9:00-15:00）
            if timestamp.hour < 9 or timestamp.hour >= 15:
                continue
            
            # 価格変動生成（リアルな値動き）
            volatility = 0.002  # 0.2%の標準ボラティリティ
            
            # 市場時間帯による出来高調整
            if 9 <= timestamp.hour < 10:
                volume_multiplier = 2.0  # 寄り付き
            elif 11 <= timestamp.hour < 12:
                volume_multiplier = 1.5  # 前場後半
            elif 12 <= timestamp.hour < 13:
                volume_multiplier = 0.8  # 昼休み
            elif 14 <= timestamp.hour < 15:
                volume_multiplier = 1.8  # 大引け前
            else:
                volume_multiplier = 1.0
            
            # 価格変動
            price_change = np.random.normal(0, volatility)
            
            # トレンド要素追加
            trend_factor = np.sin(i * 0.01) * 0.001  # 長期トレンド
            momentum_factor = np.random.normal(0, 0.001)  # モメンタム
            
            price_change += trend_factor + momentum_factor
            
            # 新価格計算
            new_price = current_price * (1 + price_change)
            
            # OHLC生成
            high_low_range = abs(price_change) * 2
            open_price = current_price
            close_price = new_price
            
            high_price = max(open_price, close_price) + np.random.uniform(0, high_low_range) * current_price
            low_price = min(open_price, close_price) - np.random.uniform(0, high_low_range) * current_price
            
            # 出来高生成
            base_volume = 100000
            volume = int(base_volume * volume_multiplier * np.random.lognormal(0, 0.5))
            
            # データ挿入
            cursor.execute('''
                INSERT INTO five_minute_data (timestamp, symbol, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                symbol,
                round(open_price, 2),
                round(high_price, 2),
                round(low_price, 2),
                round(close_price, 2),
                volume
            ))
            
            current_price = new_price
    
    conn.commit()
    
    # データ統計表示
    cursor.execute("SELECT COUNT(*) FROM five_minute_data")
    total_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM five_minute_data")
    unique_symbols = cursor.fetchone()[0]
    
    print(f"\\n✅ デイトレード用データ生成完了")
    print(f"📊 総レコード数: {total_records}")
    print(f"📈 銘柄数: {unique_symbols}")
    
    # サンプルデータ表示
    cursor.execute('''
        SELECT symbol, COUNT(*) as count, 
               MIN(timestamp) as start_date, 
               MAX(timestamp) as end_date,
               AVG(close_price) as avg_price,
               AVG(volume) as avg_volume
        FROM five_minute_data 
        GROUP BY symbol 
        ORDER BY symbol
    ''')
    
    results = cursor.fetchall()
    print("\\n📋 銘柄別データ統計:")
    for row in results:
        symbol, count, start_date, end_date, avg_price, avg_volume = row
        print(f"  {symbol}: {count}件, 平均価格: {avg_price:.0f}円, 平均出来高: {avg_volume:.0f}")
    
    conn.close()

def add_advanced_features_to_data(db_path="enhanced_trading.db"):
    """高度なデイトレード特徴量をデータベースに追加"""
    
    conn = sqlite3.connect(db_path)
    
    # 特徴量テーブル作成
    conn.execute('''
        CREATE TABLE IF NOT EXISTS trading_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            
            -- 価格特徴量
            price_change_1min REAL,
            price_change_5min REAL,
            price_acceleration REAL,
            
            -- モメンタム特徴量
            momentum_5min REAL,
            momentum_15min REAL,
            
            -- ボラティリティ特徴量
            volatility_5min REAL,
            volatility_15min REAL,
            
            -- 出来高特徴量
            volume_ratio REAL,
            volume_spike INTEGER,
            volume_trend REAL,
            
            -- テクニカル特徴量
            rsi REAL,
            bollinger_position REAL,
            macd_signal REAL,
            
            -- 市場マイクロ構造
            bid_ask_spread REAL,
            order_imbalance REAL,
            tick_momentum REAL,
            
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 各銘柄の特徴量計算
    symbols_query = "SELECT DISTINCT symbol FROM five_minute_data ORDER BY symbol"
    symbols = [row[0] for row in conn.execute(symbols_query).fetchall()]
    
    for symbol in symbols:
        print(f"銘柄 {symbol} の特徴量計算中...")
        
        # 価格データ取得
        price_query = '''
            SELECT timestamp, open_price, high_price, low_price, close_price, volume
            FROM five_minute_data
            WHERE symbol = ?
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(price_query, conn, params=(symbol,))
        
        if len(df) < 20:
            continue
        
        # 特徴量計算
        df['price_change_1min'] = df['close_price'].pct_change()
        df['price_change_5min'] = df['close_price'].pct_change(5)
        df['price_acceleration'] = df['price_change_1min'].diff()
        
        # モメンタム
        df['momentum_5min'] = df['close_price'] / df['close_price'].shift(5) - 1
        df['momentum_15min'] = df['close_price'] / df['close_price'].shift(15) - 1
        
        # ボラティリティ
        df['volatility_5min'] = df['price_change_1min'].rolling(5).std()
        df['volatility_15min'] = df['price_change_1min'].rolling(15).std()
        
        # 出来高
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(15).mean()
        
        # RSI
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ボリンジャーバンド
        bb_ma = df['close_price'].rolling(20).mean()
        bb_std = df['close_price'].rolling(20).std()
        bb_upper = bb_ma + (bb_std * 2)
        bb_lower = bb_ma - (bb_std * 2)
        df['bollinger_position'] = (df['close_price'] - bb_lower) / (bb_upper - bb_lower)
        
        # MACD
        exp1 = df['close_price'].ewm(span=12).mean()
        exp2 = df['close_price'].ewm(span=26).mean()
        macd = exp1 - exp2
        df['macd_signal'] = macd.ewm(span=9).mean()
        
        # 市場マイクロ構造（模擬）
        df['bid_ask_spread'] = np.random.normal(0.001, 0.0005, len(df))
        df['order_imbalance'] = np.random.normal(0, 0.5, len(df))
        df['tick_momentum'] = df['close_price'].diff().rolling(3).mean()
        
        # データベース挿入
        for index, row in df.iterrows():
            if pd.isna(row['rsi']) or pd.isna(row['momentum_5min']):
                continue
            
            conn.execute('''
                INSERT INTO trading_features (
                    timestamp, symbol, price_change_1min, price_change_5min, price_acceleration,
                    momentum_5min, momentum_15min, volatility_5min, volatility_15min,
                    volume_ratio, volume_spike, volume_trend, rsi, bollinger_position, macd_signal,
                    bid_ask_spread, order_imbalance, tick_momentum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['timestamp'], symbol,
                row['price_change_1min'], row['price_change_5min'], row['price_acceleration'],
                row['momentum_5min'], row['momentum_15min'], row['volatility_5min'], row['volatility_15min'],
                row['volume_ratio'], row['volume_spike'], row['volume_trend'],
                row['rsi'], row['bollinger_position'], row['macd_signal'],
                row['bid_ask_spread'], row['order_imbalance'], row['tick_momentum']
            ))
    
    conn.commit()
    
    # 統計表示
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trading_features")
    feature_count = cursor.fetchone()[0]
    
    print(f"\\n✅ 特徴量データ生成完了")
    print(f"📊 特徴量レコード数: {feature_count}")
    
    conn.close()

def main():
    """メイン実行"""
    print("🔄 デイトレード用データ生成開始...")
    
    # 基本データ生成
    generate_detailed_trading_data()
    
    # 特徴量計算
    add_advanced_features_to_data()
    
    print("\\n🎉 デイトレード用データ生成完了")

if __name__ == "__main__":
    main()
