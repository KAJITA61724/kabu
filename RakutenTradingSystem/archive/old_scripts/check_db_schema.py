import sqlite3
import pandas as pd

# trading_data.dbのテーブルを確認
conn = sqlite3.connect('trading_data.db')
cursor = conn.cursor()

print("=== trading_data.db のテーブル一覧 ===")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
for table in tables:
    print(f"テーブル: {table[0]}")

print("\n=== chart_data テーブルのスキーマ ===")
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='chart_data'")
result = cursor.fetchone()
if result:
    print(result[0])
else:
    print("chart_dataテーブルが見つかりません")

print("\n=== chart_data テーブルのサンプルデータ ===")
try:
    cursor.execute("SELECT * FROM chart_data LIMIT 5")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
except Exception as e:
    print(f"エラー: {e}")

print("\n=== chart_data テーブルのデータ数 ===")
try:
    cursor.execute("SELECT COUNT(*) FROM chart_data")
    count = cursor.fetchone()[0]
    print(f"総データ数: {count}")
    
    cursor.execute("SELECT symbol, COUNT(*) FROM chart_data GROUP BY symbol")
    symbol_counts = cursor.fetchall()
    for symbol, count in symbol_counts:
        print(f"{symbol}: {count}件")
except Exception as e:
    print(f"エラー: {e}")

conn.close()
