import sqlite3
from datetime import datetime

conn = sqlite3.connect('enhanced_trading.db')
cursor = conn.cursor()

# 9:20の具体的なデータを確認
cursor.execute('SELECT COUNT(*) FROM five_minute_data WHERE timestamp = "2025-07-18 09:20:00"')
count_920 = cursor.fetchone()[0]
print(f"09:20:00のデータ件数: {count_920}")

# 9:15の具体的なデータを確認
cursor.execute('SELECT COUNT(*) FROM five_minute_data WHERE timestamp = "2025-07-18 09:15:00"')
count_915 = cursor.fetchone()[0]
print(f"09:15:00のデータ件数: {count_915}")

# 今日のデータを確認
cursor.execute('SELECT COUNT(*) FROM five_minute_data WHERE timestamp >= "2025-07-18 00:00:00"')
count_today = cursor.fetchone()[0]
print(f"今日のデータ件数: {count_today}")

# テーブルの構造確認
cursor.execute("PRAGMA table_info(five_minute_data)")
table_info = cursor.fetchall()
print(f"\nテーブル構造:")
for info in table_info:
    print(f"  {info[1]} ({info[2]})")

# もしデータがあれば最新の5件を表示
cursor.execute('SELECT symbol, timestamp, close_price FROM five_minute_data ORDER BY timestamp DESC LIMIT 5')
recent_data = cursor.fetchall()
print(f"\n最新データ（5件）:")
for data in recent_data:
    print(f"  {data}")

conn.close()
