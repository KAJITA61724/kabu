"""
デモトレードシステム - 統合版
既存のenhanced_demo_tradingを整理統合
"""

import logging
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List

# 既存モジュールをインポート
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from enhanced_demo_trading import (
    EnhancedHistoricalDataCollector, 
    EnhancedDemoTradingSimulator,
    DemoMethod
)
from core.enhanced_data_collector import EnhancedDataCollector
from core.ml_models import MLTradingModels
from core.strategy_system import TradingViewStrategies
from core.technical_indicators import TechnicalIndicators

class DemoTradingSystem:
    """デモトレードシステム統合クラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ログディレクトリ設定
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        monthly_log_dir = log_dir / datetime.now().strftime("%Y%m")
        monthly_log_dir.mkdir(exist_ok=True)
        
        # ログハンドラー設定
        log_handler = logging.FileHandler(monthly_log_dir / 'demo_trading.log')
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(log_handler)
        
        # データ収集・シミュレーター初期化
        self.data_collector = EnhancedHistoricalDataCollector()
        
        # 設定読み込み
        self.config = {
            'capital': 2000000,
            'max_positions': 3,
            'demo_mode': True
        }
        
        self.simulator = EnhancedDemoTradingSimulator(self.config, self.data_collector)
        
        self.logger.info("デモトレードシステム初期化完了")
    
    def _ensure_data_available(self) -> bool:
        """データ利用可能性確認"""
        min_date, max_date = self.data_collector.get_available_data_range()
        
        if not min_date or not max_date:
            print("❌ デモ用データがありません")
            print("💡 まずメインDBからデータをインポートしてください")
            
            choice = input("今すぐインポートしますか？ (y/n): ").strip().lower()
            if choice == 'y':
                print("📥 データインポート中...")
                self.data_collector.import_from_main_database()
                
                # 再確認
                min_date, max_date = self.data_collector.get_available_data_range()
                if min_date and max_date:
                    print(f"✅ インポート完了: {min_date} - {max_date}")
                    return True
                else:
                    print("❌ インポート後もデータが不足しています")
                    return False
            else:
                return False
        
        return True
    
    def _get_demo_period(self) -> tuple:
        """デモ実行期間取得"""
        min_date, max_date = self.data_collector.get_available_data_range()
        
        # 最新の5営業日を使用
        end_date = max_date
        start_date = max(min_date, end_date - timedelta(days=7))
        
        print(f"📅 デモ期間: {start_date} - {end_date}")
        return start_date, end_date
    
    def run_ml_demo(self):
        """方法1（ML）デモ実行"""
        print("\n🤖 方法1（ML）デモトレード開始")
        print("="*50)
        
        if not self._ensure_data_available():
            return
        
        start_date, end_date = self._get_demo_period()
        
        try:
            # MLデモ実行
            methods = [DemoMethod.METHOD_1_ML]
            self.simulator.run_enhanced_demo_simulation(start_date, end_date, methods)
            
            print("✅ 方法1デモ完了")
            
        except Exception as e:
            print(f"❌ 方法1デモエラー: {e}")
    
    def run_strategy_demo(self):
        """方法2（戦略）デモ実行"""
        print("\n📊 方法2（戦略）デモトレード開始")
        print("="*50)
        
        if not self._ensure_data_available():
            return
        
        start_date, end_date = self._get_demo_period()
        
        try:
            # 戦略デモ実行
            methods = [DemoMethod.METHOD_2_STRATEGY]
            self.simulator.run_enhanced_demo_simulation(start_date, end_date, methods)
            
            print("✅ 方法2デモ完了")
            
        except Exception as e:
            print(f"❌ 方法2デモエラー: {e}")
    
    def run_comparison_demo(self):
        """両方法比較デモ実行"""
        print("\n⚡ 両方法比較デモトレード開始")
        print("="*50)
        
        if not self._ensure_data_available():
            return
        
        start_date, end_date = self._get_demo_period()
        
        try:
            # 両方法比較実行
            methods = [DemoMethod.METHOD_1_ML, DemoMethod.METHOD_2_STRATEGY]
            self.simulator.run_enhanced_demo_simulation(start_date, end_date, methods)
            
            print("✅ 比較デモ完了")
            
        except Exception as e:
            print(f"❌ 比較デモエラー: {e}")
    
    def show_demo_results(self):
        """過去デモ結果確認"""
        print("\n📋 過去デモ結果")
        print("="*60)
        
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.data_collector.db_path)
            
            # セッション一覧取得
            query = """
                SELECT session_name, method, start_date, end_date, 
                       total_trades, winning_trades, total_pnl, avg_confidence,
                       created_at
                FROM enhanced_demo_sessions
                ORDER BY created_at DESC
                LIMIT 10
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                print("過去のデモ結果がありません")
                return
            
            for _, row in df.iterrows():
                win_rate = (row['winning_trades'] / row['total_trades'] * 100) if row['total_trades'] > 0 else 0
                
                print(f"📊 {row['session_name']}")
                print(f"   方法: {row['method']}")
                print(f"   期間: {row['start_date']} - {row['end_date']}")
                print(f"   成績: {win_rate:.1f}% ({row['winning_trades']}/{row['total_trades']})")
                print(f"   損益: {row['total_pnl']:,.0f}円")
                print(f"   信頼度: {row['avg_confidence']:.3f}")
                print(f"   実行日: {row['created_at']}")
                print()
                
        except Exception as e:
            print(f"❌ 結果表示エラー: {e}")
        
        print("="*60)
