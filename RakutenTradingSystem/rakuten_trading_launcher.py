"""
楽天トレーディングシステム統合ランチャー

全機能を整理統合:
- データ収集システム
- デモトレードシステム  
- リアルトレードシステム
- バックテストシステム
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# パス設定
sys.path.append(str(Path(__file__).parent))

from systems.data_collection_system import DataCollectionSystem
from systems.demo_trading_system import DemoTradingSystem
from systems.real_trading_system import RealTradingSystem
from systems.backtest_system import BacktestSystem

class RakutenTradingLauncher:
    """楽天トレーディングシステム統合ランチャー"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # システムコンポーネント
        self.data_collection = None
        self.demo_trading = None
        self.real_trading = None
        self.backtest = None
        
        self.logger.info("楽天トレーディングシステム初期化完了")
    
    def setup_logging(self):
        """ログ設定"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 日付別サブディレクトリ作成
        date_str = datetime.now().strftime("%Y%m")
        monthly_log_dir = log_dir / date_str
        monthly_log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(monthly_log_dir / f'rakuten_trading_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
    
    def show_main_menu(self):
        """メインメニュー表示"""
        print("\n" + "="*80)
        print("🏆 楽天トレーディングシステム統合ランチャー")
        print("="*80)
        print("📊 1. データ収集システム - MarketSpeed IIからリアルタイムデータ収集")
        print("🎮 2. デモトレードシステム - 方法1・2のリスクフリーテスト")
        print("💰 3. リアルトレードシステム - 実際の取引実行")
        print("📈 4. バックテストシステム - 過去データによる戦略検証")
        print("⚙️  5. システム設定・状況確認")
        print("📋 6. 全システム統合監視")
        print("❌ 7. 終了")
        print("="*80)
    
    def initialize_data_collection(self):
        """データ収集システム初期化"""
        if not self.data_collection:
            from systems.data_collection_system import DataCollectionSystem
            self.data_collection = DataCollectionSystem()
        return self.data_collection
    
    def initialize_demo_trading(self):
        """デモトレードシステム初期化"""
        if not self.demo_trading:
            from systems.demo_trading_system import DemoTradingSystem
            self.demo_trading = DemoTradingSystem()
        return self.demo_trading
    
    def initialize_real_trading(self):
        """リアルトレードシステム初期化"""
        if not self.real_trading:
            from systems.real_trading_system import RealTradingSystem
            self.real_trading = RealTradingSystem()
        return self.real_trading
    
    def initialize_backtest(self):
        """バックテストシステム初期化"""
        if not self.backtest:
            from systems.backtest_system import BacktestSystem
            self.backtest = BacktestSystem()
        return self.backtest
    
    def run_data_collection_menu(self):
        """データ収集メニュー"""
        system = self.initialize_data_collection()
        
        while True:
            print("\n📊 データ収集システム")
            print("1. データ収集開始")
            print("2. データ収集停止")
            print("3. 収集状況確認")
            print("4. データ品質チェック")
            print("5. 戻る")
            
            choice = input("選択してください: ").strip()
            
            if choice == '1':
                system.start_collection()
            elif choice == '2':
                system.stop_collection()
            elif choice == '3':
                system.show_status()
            elif choice == '4':
                system.check_data_quality()
            elif choice == '5':
                break
            else:
                print("❌ 無効な選択です")
    
    def run_demo_trading_menu(self):
        """デモトレードメニュー"""
        system = self.initialize_demo_trading()
        
        while True:
            print("\n🎮 デモトレードシステム")
            print("1. 方法1（ML）デモ実行")
            print("2. 方法2（戦略）デモ実行")
            print("3. 両方法比較デモ")
            print("4. 過去デモ結果確認")
            print("5. 戻る")
            
            choice = input("選択してください: ").strip()
            
            if choice == '1':
                system.run_ml_demo()
            elif choice == '2':
                system.run_strategy_demo()
            elif choice == '3':
                system.run_comparison_demo()
            elif choice == '4':
                system.show_demo_results()
            elif choice == '5':
                break
            else:
                print("❌ 無効な選択です")
    
    def run_real_trading_menu(self):
        """リアルトレードメニュー"""
        system = self.initialize_real_trading()
        
        print("\n⚠️  リアルトレード確認")
        confirm = input("実際の資金を使用した取引を開始しますか？ (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            print("取引をキャンセルしました")
            return
        
        while True:
            print("\n💰 リアルトレードシステム")
            print("1. 取引開始（方法1: ML）")
            print("2. 取引開始（方法2: 戦略）")
            print("3. 取引開始（両方法）")
            print("4. 取引停止")
            print("5. ポジション確認")
            print("6. 取引履歴確認")
            print("7. 戻る")
            
            choice = input("選択してください: ").strip()
            
            if choice == '1':
                system.start_ml_trading()
            elif choice == '2':
                system.start_strategy_trading()
            elif choice == '3':
                system.start_integrated_trading()
            elif choice == '4':
                system.stop_trading()
            elif choice == '5':
                system.show_positions()
            elif choice == '6':
                system.show_trading_history()
            elif choice == '7':
                break
            else:
                print("❌ 無効な選択です")
    
    def run_backtest_menu(self):
        """バックテストメニュー"""
        system = self.initialize_backtest()
        
        while True:
            print("\n📈 バックテストシステム")
            print("1. 方法1（ML）バックテスト")
            print("2. 方法2（戦略）バックテスト")
            print("3. 両方法比較バックテスト")
            print("4. カスタム期間バックテスト")
            print("5. バックテスト結果確認")
            print("6. 戻る")
            
            choice = input("選択してください: ").strip()
            
            if choice == '1':
                system.run_ml_backtest()
            elif choice == '2':
                system.run_strategy_backtest()
            elif choice == '3':
                system.run_comparison_backtest()
            elif choice == '4':
                system.run_custom_backtest()
            elif choice == '5':
                system.show_backtest_results()
            elif choice == '6':
                break
            else:
                print("❌ 無効な選択です")
    
    def run_system_config_menu(self):
        """システム設定メニュー"""
        while True:
            print("\n⚙️ システム設定・状況確認")
            print("1. 全システム状況確認")
            print("2. データベース状況確認")
            print("3. 設定ファイル確認")
            print("4. ログファイル確認")
            print("5. システム初期化")
            print("6. 戻る")
            
            choice = input("選択してください: ").strip()
            
            if choice == '1':
                self.show_all_system_status()
            elif choice == '2':
                self.show_database_status()
            elif choice == '3':
                self.show_config_files()
            elif choice == '4':
                self.show_log_files()
            elif choice == '5':
                self.initialize_all_systems()
            elif choice == '6':
                break
            else:
                print("❌ 無効な選択です")
    
    def run_integrated_monitoring(self):
        """統合監視画面"""
        print("\n📋 全システム統合監視")
        print("="*60)
        
        # データ収集状況
        if self.data_collection:
            print("📊 データ収集: 稼働中")
        else:
            print("📊 データ収集: 停止中")
        
        # デモトレード状況
        if self.demo_trading:
            print("🎮 デモトレード: 利用可能")
        else:
            print("🎮 デモトレード: 未初期化")
        
        # リアルトレード状況
        if self.real_trading:
            print("💰 リアルトレード: 稼働中")
        else:
            print("💰 リアルトレード: 停止中")
        
        # バックテスト状況
        if self.backtest:
            print("📈 バックテスト: 利用可能")
        else:
            print("📈 バックテスト: 未初期化")
        
        print("="*60)
        input("Enterキーで戻る...")
    
    def show_all_system_status(self):
        """全システム状況表示"""
        print("\n🔍 全システム状況確認")
        print("="*50)
        
        # ファイル存在確認
        core_files = [
            "ml_models.py", "strategy_system.py", "technical_indicators.py",
            "enhanced_data_collector.py", "data_quality_monitor.py"
        ]
        
        print("📁 コアファイル状況:")
        for file in core_files:
            exists = "✅" if os.path.exists(file) else "❌"
            print(f"  {exists} {file}")
        
        # データベース確認
        db_files = ["enhanced_trading.db", "enhanced_demo_trading.db"]
        print("\n💾 データベース状況:")
        for db in db_files:
            exists = "✅" if os.path.exists(db) else "❌"
            size = os.path.getsize(db) / 1024 / 1024 if os.path.exists(db) else 0
            print(f"  {exists} {db} ({size:.1f}MB)")
        
        print("="*50)
    
    def show_database_status(self):
        """データベース状況表示"""
        try:
            import sqlite3
            
            print("\n💾 データベース詳細状況")
            print("="*50)
            
            # メインDB確認
            if os.path.exists("enhanced_trading.db"):
                conn = sqlite3.connect("enhanced_trading.db")
                
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                print("📊 メインDB (enhanced_trading.db):")
                for table in tables:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
                    print(f"  - {table[0]}: {count:,}レコード")
                
                conn.close()
            
            # デモDB確認
            if os.path.exists("enhanced_demo_trading.db"):
                conn = sqlite3.connect("enhanced_demo_trading.db")
                
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                print("\n🎮 デモDB (enhanced_demo_trading.db):")
                for table in tables:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
                    print(f"  - {table[0]}: {count:,}レコード")
                
                conn.close()
            
            print("="*50)
            
        except Exception as e:
            print(f"❌ データベース確認エラー: {e}")
    
    def show_config_files(self):
        """設定ファイル表示"""
        config_files = [
            "config_daytrading.json", "enhanced_demo_config.json",
            "prime_symbols.csv", "nikkei225_symbols.csv"
        ]
        
        print("\n⚙️ 設定ファイル状況")
        print("="*40)
        
        for config in config_files:
            if os.path.exists(config):
                size = os.path.getsize(config) / 1024
                print(f"✅ {config} ({size:.1f}KB)")
            else:
                print(f"❌ {config} (未作成)")
        
        print("="*40)
    
    def show_log_files(self):
        """ログファイル表示"""
        log_dir = Path("logs")
        
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            
            print("\n📋 ログファイル")
            print("="*40)
            
            for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                size = log_file.stat().st_size / 1024
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                print(f"📄 {log_file.name} ({size:.1f}KB) - {mtime.strftime('%Y/%m/%d %H:%M')}")
            
            print("="*40)
        else:
            print("📋 ログディレクトリが存在しません")
    
    def initialize_all_systems(self):
        """全システム初期化"""
        print("\n🔄 全システム初期化中...")
        
        confirm = input("全てのシステムを初期化しますか？ (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("初期化をキャンセルしました")
            return
        
        try:
            # 各システム初期化
            self.initialize_data_collection()
            self.initialize_demo_trading()
            self.initialize_backtest()
            
            print("✅ 全システム初期化完了")
            
        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
    
    def run(self):
        """メインループ実行"""
        try:
            while True:
                self.show_main_menu()
                choice = input("\n選択してください (1-7): ").strip()
                
                if choice == '1':
                    self.run_data_collection_menu()
                elif choice == '2':
                    self.run_demo_trading_menu()
                elif choice == '3':
                    self.run_real_trading_menu()
                elif choice == '4':
                    self.run_backtest_menu()
                elif choice == '5':
                    self.run_system_config_menu()
                elif choice == '6':
                    self.run_integrated_monitoring()
                elif choice == '7':
                    print("👋 システムを終了します")
                    break
                else:
                    print("❌ 無効な選択です")
        
        except KeyboardInterrupt:
            print("\n👋 システムを終了します")
        except Exception as e:
            self.logger.error(f"システムエラー: {e}")
            print(f"❌ システムエラー: {e}")

def main():
    """メイン関数"""
    launcher = RakutenTradingLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
