import sqlite3
from datetime import datetime

conn = sqlite3.connect('enhanced_trading.db')
cursor = conn.cursor()

# 9:25の具体的なデータを確認
cursor.execute('SELECT COUNT(*) FROM five_minute_data WHERE timestamp = "2025-07-18 09:25:00"')
count_925 = cursor.fetchone()[0]
print(f"09:25:00のデータ件数: {count_925}")

# 9:20の具体的なデータを確認
cursor.execute('SELECT COUNT(*) FROM five_minute_data WHERE timestamp = "2025-07-18 09:20:00"')
count_920 = cursor.fetchone()[0]
print(f"09:20:00のデータ件数: {count_920}")

# 今日の全データ確認
cursor.execute('SELECT COUNT(*) FROM five_minute_data WHERE timestamp >= "2025-07-18 00:00:00"')
count_today = cursor.fetchone()[0]
print(f"今日の全データ件数: {count_today}")

# 9:24以降のデータを確認
cursor.execute('SELECT COUNT(*) FROM five_minute_data WHERE timestamp >= "2025-07-18 09:24:00"')
count_924_after = cursor.fetchone()[0]
print(f"09:24以降のデータ件数: {count_924_after}")

# 全期間のデータを確認
cursor.execute('SELECT COUNT(*) FROM five_minute_data')
count_all = cursor.fetchone()[0]
print(f"全期間のデータ件数: {count_all}")

# テーブル一覧確認
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"\nテーブル一覧: {[table[0] for table in tables]}")

conn.close()
