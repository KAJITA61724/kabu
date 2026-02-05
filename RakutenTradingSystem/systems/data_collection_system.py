"""
データ収集システム - 統合版
既存のenhanced_data_collectorとphase1_integrationを統合
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

# 既存モジュールをインポート
import sys
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.enhanced_data_collector import EnhancedDataCollector
from core.data_quality_monitor import DataQualityMonitor
from core.technical_indicators import TechnicalIndicators

class DataCollectionSystem:
    """データ収集システム統合クラス"""
    
    def __init__(self, config_file: str = "configs/data_collection_config.json"):
        self.logger = logging.getLogger(__name__)
        
        # ログディレクトリ設定
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        monthly_log_dir = log_dir / datetime.now().strftime("%Y%m")
        monthly_log_dir.mkdir(exist_ok=True)
        
        # ログハンドラー設定
        log_handler = logging.FileHandler(monthly_log_dir / 'data_collection.log')
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(log_handler)
        
        # 既存コンポーネント初期化
        self.data_collector = EnhancedDataCollector()
        self.quality_monitor = DataQualityMonitor()
        self.technical_indicators = TechnicalIndicators()
        
        self.running = False
        self.collection_thread = None
        
        self.logger.info("データ収集システム初期化完了")
    
    def start_collection(self):
        """データ収集開始"""
        if self.running:
            print("⚠️ データ収集は既に稼働中です")
            return
        
        print("📊 データ収集を開始します...")
        
        try:
            # Excel接続初期化
            if not self.data_collector.initialize_excel_connection():
                print("❌ Excel接続に失敗しました")
                return
            
            # 銘柄リスト読み込み
            self.data_collector.load_nikkei225_symbols()
            
            # 収集開始
            self.running = True
            self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self.collection_thread.start()
            
            print("✅ データ収集を開始しました")
            
        except Exception as e:
            print(f"❌ データ収集開始エラー: {e}")
    
    def stop_collection(self):
        """データ収集停止"""
        if not self.running:
            print("⚠️ データ収集は既に停止中です")
            return
        
        print("🛑 データ収集を停止します...")
        
        self.running = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=10)
        
        # Excel接続クリーンアップ
        try:
            if self.data_collector.excel_app:
                self.data_collector.excel_app.Quit()
        except:
            pass
        
        print("✅ データ収集を停止しました")
    
    def _collection_loop(self):
        """データ収集ループ"""
        while self.running:
            try:
                # 5分足データ収集
                collected_data = self.data_collector.collect_5min_data()
                
                if collected_data:
                    self.logger.info(f"データ収集完了: {len(collected_data)}銘柄")
                    
                    # テクニカル指標計算
                    self.technical_indicators.calculate_and_save_all_symbols()
                
                # 5分待機
                time.sleep(300)
                
            except Exception as e:
                self.logger.error(f"データ収集ループエラー: {e}")
                time.sleep(60)  # エラー時は1分待機
    
    def show_status(self):
        """収集状況確認"""
        print("\n📊 データ収集状況")
        print("="*40)
        
        status = "稼働中" if self.running else "停止中"
        print(f"状態: {status}")
        
        if self.data_collector.symbols:
            print(f"対象銘柄数: {len(self.data_collector.symbols)}")
        
        excel_status = "接続中" if (self.data_collector.connected and self.data_collector.excel_app) else "未接続"
        print(f"Excel接続: {excel_status}")
        
        # 最新データ確認
        try:
            import sqlite3
            conn = sqlite3.connect("enhanced_trading.db")
            
            latest_query = """
                SELECT MAX(timestamp) as latest_time, COUNT(DISTINCT symbol) as symbols
                FROM five_minute_data
                WHERE DATE(timestamp) = DATE('now')
            """
            
            result = conn.execute(latest_query).fetchone()
            
            if result and result[0]:
                print(f"本日最新データ: {result[0]}")
                print(f"本日収集銘柄数: {result[1]}")
            
            conn.close()
            
        except Exception as e:
            print(f"データ確認エラー: {e}")
        
        print("="*40)
    
    def check_data_quality(self):
        """データ品質チェック"""
        print("\n🔍 データ品質チェック中...")
        
        try:
            # 品質レポート生成
            report = self.quality_monitor.generate_quality_report()
            
            print("="*50)
            print("📋 データ品質レポート")
            print("="*50)
            print(f"品質スコア: {report['quality_score']}/100 ({report['status']})")
            print(f"データ完全性: {report['completeness']['avg_completeness']:.1f}%")
            print(f"価格有効性: {report['completeness']['avg_price_validity']:.1f}%")
            print(f"異常値総数: {report['anomalies']['total_anomalies']}件")
            print(f"データ新鮮度: {report['freshness']['fresh_symbols']}/{report['freshness']['total_symbols']} 銘柄")
            print("="*50)
            
            # レポート保存
            self.quality_monitor.save_quality_report(report)
            print("✅ 品質レポートを保存しました")
            
        except Exception as e:
            print(f"❌ 品質チェックエラー: {e}")
