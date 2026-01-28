"""
明日の日中データ収集スケジューラー
7/18（金）の9:00-15:00の日中データを収集
"""

import sys
from pathlib import Path
import time
import logging
from datetime import datetime, timedelta
import schedule

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.enhanced_data_collector import EnhancedDataCollector

# ログ設定（月次フォルダに保存）
log_dir = Path("logs") / datetime.now().strftime("%Y%m")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'scheduled_data_collection.log'),
        logging.StreamHandler()
    ]
)

class ScheduledDataCollector:
    """スケジュール付きデータ収集システム"""
    
    def __init__(self):
        self.collector = None
        self.is_market_hours = False
        self.collection_count = 0
        
    def initialize_collector(self):
        """データコレクターを初期化"""
        try:
            self.collector = EnhancedDataCollector()
            logging.info("データコレクター初期化完了")
            
            # Excel接続を試行（失敗してもデモモードで続行）
            try:
                if self.collector.initialize_excel_connection():
                    logging.info("Excel接続成功 - リアルタイムデータ収集モード")
                else:
                    logging.warning("Excel接続失敗 - デモデータ収集モード")
            except Exception as e:
                logging.warning(f"Excel接続試行中にエラー: {e} - デモモードで続行")
                
            return True
        except Exception as e:
            logging.error(f"データコレクター初期化エラー: {e}")
            return False
    
    def is_trading_hours(self):
        """取引時間かどうかチェック（平日9:00-15:00）"""
        now = datetime.now()
        
        # 平日チェック（月曜=0, 日曜=6）
        if now.weekday() >= 5:  # 土日
            return False
            
        # 時間チェック（9:00-15:00）
        trading_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        trading_end = now.replace(hour=15, minute=0, second=0, microsecond=0)
        
        return trading_start <= now <= trading_end
    
    def collect_market_data(self):
        """市場データ収集実行"""
        try:
            if not self.is_trading_hours():
                if self.is_market_hours:
                    logging.info("取引時間終了 - データ収集を一時停止")
                    self.is_market_hours = False
                return
            
            if not self.is_market_hours:
                logging.info("取引時間開始 - データ収集を開始")
                self.is_market_hours = True
            
            # データ収集実行
            current_time = datetime.now().strftime('%H:%M:%S')
            logging.info(f"[{current_time}] データ収集実行中...")
            
            if self.collector:
                # 銘柄リストを読み込み（初回のみ）
                if not self.collector.symbols:
                    self.collector.load_nikkei225_symbols()
                
                # データ収集
                data = self.collector.collect_5min_data()
                if data:
                    # データ保存
                    self.collector.save_5min_data(data)
                    self.collection_count += 1
                    logging.info(f"  -> {len(data)}銘柄のデータ収集・保存完了 (累計: {self.collection_count}回)")
                else:
                    logging.warning("  -> データ収集結果が空です")
            else:
                logging.error("データコレクターが初期化されていません")
                
        except Exception as e:
            logging.error(f"データ収集エラー: {e}")
    
    def start_scheduled_collection(self):
        """スケジュール付きデータ収集開始"""
        logging.info("=" * 60)
        logging.info("楽天トレードシステム - スケジュール付きデータ収集")
        logging.info("=" * 60)
        logging.info("対象日: 2025年7月18日（金）")
        logging.info("収集時間: 9:00-15:00 (平日のみ)")
        logging.info("収集間隔: 5分毎")
        logging.info("=" * 60)
        
        if not self.initialize_collector():
            logging.error("システム初期化に失敗しました")
            return
        
        # 5分毎にデータ収集をスケジュール（正確なタイミング）
        # 9:00, 9:05, 9:10, 9:15, 9:20, 9:25, 9:30...
        schedule.every().minute.at(":00").do(self.collect_market_data)
        schedule.every().minute.at(":05").do(self.collect_market_data)
        schedule.every().minute.at(":10").do(self.collect_market_data)
        schedule.every().minute.at(":15").do(self.collect_market_data)
        schedule.every().minute.at(":20").do(self.collect_market_data)
        schedule.every().minute.at(":25").do(self.collect_market_data)
        schedule.every().minute.at(":30").do(self.collect_market_data)
        schedule.every().minute.at(":35").do(self.collect_market_data)
        schedule.every().minute.at(":40").do(self.collect_market_data)
        schedule.every().minute.at(":45").do(self.collect_market_data)
        schedule.every().minute.at(":50").do(self.collect_market_data)
        schedule.every().minute.at(":55").do(self.collect_market_data)
        
        # 1分毎に状況確認
        def status_check():
            now = datetime.now()
            if self.is_trading_hours():
                if not self.is_market_hours:
                    logging.info(f"[{now.strftime('%H:%M')}] 取引時間中 - 次回データ収集を待機中...")
            else:
                if now.hour < 9:
                    market_open = now.replace(hour=9, minute=0, second=0)
                    time_to_open = market_open - now
                    hours, remainder = divmod(time_to_open.seconds, 3600)
                    minutes, _ = divmod(remainder, 60)
                    logging.info(f"[{now.strftime('%H:%M')}] 市場開始まで {hours}時間{minutes}分")
                elif now.hour >= 15:
                    logging.info(f"[{now.strftime('%H:%M')}] 取引時間外 - 明日の開始を待機中...")
        
        schedule.every(1).minutes.do(status_check)
        
        logging.info("スケジューラー開始 - Ctrl+Cで停止")
        
        try:
            # 初回状況確認
            status_check()
            
            # スケジュール実行ループ
            while True:
                schedule.run_pending()
                time.sleep(30)  # 30秒毎にスケジュールチェック
                
        except KeyboardInterrupt:
            logging.info("\nデータ収集スケジューラーを停止します...")
            logging.info(f"総データ収集回数: {self.collection_count}回")

def main():
    """メイン実行関数"""
    collector = ScheduledDataCollector()
    collector.start_scheduled_collection()

if __name__ == "__main__":
    main()
