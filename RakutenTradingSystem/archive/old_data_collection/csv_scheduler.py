"""
CSV監視スケジュール実行システム
5分間隔でCSVファイルを監視・処理
"""

import schedule
import time
import logging
from datetime import datetime, timedelta
from csv_data_collector import CSVDataCollector

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('csv_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class CSVScheduler:
    """CSV監視スケジューラー"""
    
    def __init__(self):
        self.collector = CSVDataCollector()
        self.is_running = False
        
    def is_market_time(self) -> bool:
        """市場時間チェック"""
        now = datetime.now()
        
        # 平日のみ
        if now.weekday() >= 5:  # 土日
            return False
        
        # 9:00-15:00の間
        market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=0, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def run_collection(self):
        """データ収集実行"""
        if not self.is_market_time():
            logging.info("市場時間外のため収集をスキップ")
            return
        
        if self.is_running:
            logging.warning("既に収集処理が実行中です")
            return
        
        self.is_running = True
        
        try:
            logging.info("CSV監視データ収集開始")
            start_time = datetime.now()
            
            # データ収集実行
            collected_count = self.collector.run_collection()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logging.info(f"収集完了: {collected_count}銘柄 ({duration:.1f}秒)")
            
        except Exception as e:
            logging.error(f"収集エラー: {e}")
        finally:
            self.is_running = False
    
    def setup_schedule(self):
        """スケジュール設定"""
        # 5分間隔で実行
        schedule.every(5).minutes.do(self.run_collection)
        
        # 特定時刻での実行も可能
        schedule.every().day.at("09:00").do(self.run_collection)
        schedule.every().day.at("09:05").do(self.run_collection)
        schedule.every().day.at("09:10").do(self.run_collection)
        # ... 必要に応じて追加
        
        logging.info("スケジュール設定完了")
    
    def start(self):
        """スケジューラー開始"""
        self.setup_schedule()
        logging.info("CSV監視スケジューラー開始")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)  # 30秒間隔でチェック
                
        except KeyboardInterrupt:
            logging.info("スケジューラー停止")
        except Exception as e:
            logging.error(f"スケジューラーエラー: {e}")

# 手動実行用の関数
def run_manual_collection():
    """手動でデータ収集実行"""
    print("=== 手動データ収集 ===")
    
    collector = CSVDataCollector()
    collected_count = collector.run_collection()
    
    print(f"収集完了: {collected_count}銘柄")
    
    # データベース確認
    import sqlite3
    conn = sqlite3.connect("enhanced_trading.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM five_minute_data")
    total_records = cursor.fetchone()[0]
    conn.close()
    
    print(f"データベース総件数: {total_records}件")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        # 手動実行
        run_manual_collection()
    else:
        # スケジュール実行
        scheduler = CSVScheduler()
        scheduler.start()
