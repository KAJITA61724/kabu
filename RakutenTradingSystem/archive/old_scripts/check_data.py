import sqlite3
from datetime import datetime

conn = sqlite3.connect('enhanced_trading.db')
cursor = conn.cursor()

# 9:15以降のデータ確認
cursor.execute('SELECT COUNT(*) FROM five_minute_data WHERE timestamp >= "2025-07-18 09:15:00"')
count_915 = cursor.fetchone()[0]
print(f"09:15以降のデータ件数: {count_915}")

# 最新データ確認
cursor.execute('SELECT symbol, timestamp FROM five_minute_data ORDER BY timestamp DESC LIMIT 10')
latest_data = cursor.fetchall()
print(f"最新データ: {latest_data}")

# 全データ件数確認
cursor.execute('SELECT COUNT(*) FROM five_minute_data')
total_count = cursor.fetchone()[0]
print(f"全データ件数: {total_count}")

conn.close()
