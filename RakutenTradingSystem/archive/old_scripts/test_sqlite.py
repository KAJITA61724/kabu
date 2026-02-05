"""
SQLiteデータベースのテスト
"""
import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta

def test_sqlite_operations():
    """SQLiteの基本操作をテスト"""
    print("=== SQLiteデータベーステスト ===")
    
    # テスト用データベースファイル
    test_db = "test_trading.db"
    
    try:
        # データベース接続
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        print("✓ データベース接続成功")
        
        # テーブル作成
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, datetime, timeframe)
            )
        ''')
        conn.commit()
        print("✓ テーブル作成成功")
        
        # テストデータ挿入
        test_data = [
            ("7203", "2025-07-18 09:00:00", "5M", 1000.0, 1010.0, 995.0, 1005.0, 100000),
            ("7203", "2025-07-18 09:05:00", "5M", 1005.0, 1015.0, 1000.0, 1012.0, 120000),
            ("7203", "2025-07-18 09:10:00", "5M", 1012.0, 1020.0, 1008.0, 1018.0, 110000),
        ]
        
        for data in test_data:
            cursor.execute('''
                INSERT OR REPLACE INTO chart_data 
                (symbol, datetime, timeframe, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
        
        conn.commit()
        print("✓ テストデータ挿入成功")
        
        # データ読み取り
        cursor.execute("SELECT * FROM chart_data WHERE symbol = '7203' ORDER BY datetime")
        rows = cursor.fetchall()
        print(f"✓ データ読み取り成功 ({len(rows)}件)")
        
        # pandasでデータ読み込み
        df = pd.read_sql_query("SELECT * FROM chart_data WHERE symbol = '7203'", conn)
        print(f"✓ pandas読み込み成功 ({len(df)}件)")
        print(df.head())
        
        conn.close()
        print("✓ データベース接続終了")
        
        # テスト用ファイル削除
        if os.path.exists(test_db):
            os.remove(test_db)
            print("✓ テストファイル削除")
        
        print("\n=== SQLiteテスト完了 ===")
        print("SQLiteは正常に動作します")
        
    except Exception as e:
        print(f"❌ SQLiteテストエラー: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_sqlite_operations()
