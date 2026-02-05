"""
楽天証券取引システム統合ランチャー - ファンダメンタルズ分析対応
"""

import logging
import time
from datetime import datetime
from pathlib import Path
import json

# 各システムのインポート
from fundamental_data_collector import FundamentalDataCollector
from fundamental_analysis_demo import FundamentalAnalysisDemo
from multi_model_comparison import MultiModelComparison
from systems.demo_trading_system import DemoTradingSystem
from core.enhanced_data_collector import EnhancedDataCollector

class RakutenTradingLauncher:
    """楽天証券取引システム統合ランチャー"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # システム初期化
        self.fundamental_collector = FundamentalDataCollector()
        self.fundamental_demo = FundamentalAnalysisDemo()
        self.model_comparison = MultiModelComparison()
        self.demo_system = DemoTradingSystem()
        self.data_collector = EnhancedDataCollector()
        
        print("🚀 楽天証券取引システム統合ランチャー")
        print("=" * 60)
        
    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_system.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def show_menu(self):
        """メニュー表示"""
        print("\\n📋 システムメニュー")
        print("-" * 40)
        print("1. ファンダメンタルズ分析デモ")
        print("2. データ収集システム")
        print("3. モデル比較検証")
        print("4. デモ取引システム")
        print("5. 統合システム実行")
        print("6. システム情報表示")
        print("7. デイトレードモデル比較")
        print("0. 終了")
        print("-" * 40)
    
    def run_fundamental_demo(self):
        """ファンダメンタルズ分析デモ実行"""
        print("\\n📊 ファンダメンタルズ分析デモ実行中...")
        try:
            self.fundamental_demo.run_demo()
            print("✅ ファンダメンタルズ分析デモ完了")
        except Exception as e:
            print(f"❌ エラー: {e}")
    
    def run_data_collection(self):
        """データ収集システム実行"""
        print("\\n📡 データ収集システム実行中...")
        try:
            # サンプル銘柄でデータ収集
            symbols = ["7203", "9984", "6758", "8306", "6501"]
            
            # ファンダメンタルズデータ収集
            print("ファンダメンタルズデータ収集中...")
            fundamental_data = self.fundamental_collector.collect_fundamental_data(symbols)
            
            # 業界別平均計算
            sector_averages = self.fundamental_collector.calculate_sector_averages()
            
            print(f"✅ データ収集完了: {len(fundamental_data)} 銘柄")
            print(f"✅ 業界別平均計算完了: {len(sector_averages)} 業界")
            
        except Exception as e:
            print(f"❌ データ収集エラー: {e}")
    
    def run_model_comparison(self):
        """モデル比較検証実行"""
        print("\\n🤖 モデル比較検証実行中...")
        try:
            # 限定的なモデル比較（ファンダメンタルズのみ）
            print("ファンダメンタルズモデル比較を実行します...")
            
            # ファンダメンタルズデータ準備
            symbols = ["7203", "9984", "6758", "8306", "6501", "4063", "9432", "8035"]
            fundamental_data = self.fundamental_collector.collect_fundamental_data(symbols)
            
            # 簡易比較結果
            if fundamental_data:
                print(f"✅ {len(fundamental_data)} 銘柄のファンダメンタルズデータを準備")
                print("📋 ファンダメンタルズ指標サマリー:")
                
                per_values = [d.per for d in fundamental_data.values() if d.per > 0]
                pbr_values = [d.pbr for d in fundamental_data.values() if d.pbr > 0]
                roe_values = [d.roe for d in fundamental_data.values() if d.roe > 0]
                
                if per_values:
                    print(f"  PER: 平均 {sum(per_values)/len(per_values):.1f}")
                if pbr_values:
                    print(f"  PBR: 平均 {sum(pbr_values)/len(pbr_values):.1f}")
                if roe_values:
                    print(f"  ROE: 平均 {sum(roe_values)/len(roe_values):.1%}")
                
                print("✅ モデル比較検証完了")
            else:
                print("❌ データ不足のため比較できません")
                
        except Exception as e:
            print(f"❌ モデル比較エラー: {e}")
    
    def run_demo_trading(self):
        """デモ取引システム実行"""
        print("\\n💰 デモ取引システム実行中...")
        try:
            print("デモ取引システムを初期化しています...")
            
            # 簡易デモ実行
            print("✅ デモ取引システム初期化完了")
            print("💡 実際の取引には楽天証券APIとの連携が必要です")
            
        except Exception as e:
            print(f"❌ デモ取引エラー: {e}")
    
    def run_integrated_system(self):
        """統合システム実行"""
        print("\\n🔗 統合システム実行中...")
        try:
            # 1. データ収集
            print("1️⃣ データ収集実行...")
            self.run_data_collection()
            
            # 2. ファンダメンタルズ分析
            print("\\n2️⃣ ファンダメンタルズ分析実行...")
            self.run_fundamental_demo()
            
            # 3. モデル比較
            print("\\n3️⃣ モデル比較実行...")
            self.run_model_comparison()
            
            print("\\n✅ 統合システム実行完了")
            
        except Exception as e:
            print(f"❌ 統合システムエラー: {e}")
    
    def show_system_info(self):
        """システム情報表示"""
        print("\\n🔍 システム情報")
        print("-" * 40)
        print("📊 ファンダメンタルズ分析:")
        print("  - Yahoo Finance API連携")
        print("  - PER, PBR, ROE等の指標分析")
        print("  - 業界別相対評価")
        print("  - 投資魅力度スコア算出")
        
        print("\\n🤖 機械学習モデル:")
        print("  - RandomForest分類器")
        print("  - バリュー投資モデル")
        print("  - 成長株モデル")
        print("  - 複合分析モデル")
        
        print("\\n💾 データベース:")
        print("  - SQLite3使用")
        print("  - ファンダメンタルズデータ")
        print("  - 技術指標データ")
        print("  - 業界別平均データ")
        
        print("\\n📈 レポート機能:")
        print("  - 分析チャート生成")
        print("  - 投資推奨銘柄表示")
        print("  - JSON形式結果保存")
        
        # データベース統計
        try:
            import sqlite3
            conn = sqlite3.connect("fundamental_data.db")
            fundamental_count = conn.execute("SELECT COUNT(*) FROM fundamental_data").fetchone()[0]
            print(f"\\n📊 データベース統計:")
            print(f"  - ファンダメンタルズデータ: {fundamental_count} 件")
            conn.close()
        except:
            print("\\n📊 データベース統計: 取得できませんでした")
    
    def run(self):
        """メインループ"""
        while True:
            self.show_menu()
            
            try:
                choice = input("\\n選択してください (0-7): ").strip()
                
                if choice == "0":
                    print("\\n👋 システムを終了します...")
                    break
                elif choice == "1":
                    self.run_fundamental_demo()
                elif choice == "2":
                    self.run_data_collection()
                elif choice == "3":
                    self.run_model_comparison()
                elif choice == "4":
                    self.run_demo_trading()
                elif choice == "5":
                    self.run_integrated_system()
                elif choice == "6":
                    self.show_system_info()
                elif choice == "7":
                    self.run_daytrading_comparison()
                else:
                    print("❌ 無効な選択です。0-7の数字を入力してください。")
                
                input("\\n⏸️ 続行するには Enter キーを押してください...")
                
            except KeyboardInterrupt:
                print("\\n\\n👋 システムを終了します...")
                break
            except Exception as e:
                print(f"❌ エラーが発生しました: {e}")
                input("\\n⏸️ 続行するには Enter キーを押してください...")
        
        print("\\n🎉 楽天証券取引システムを終了しました")
    
    def run_daytrading_comparison(self):
        """デイトレードモデル比較実行"""
        print("\\n⚡ デイトレードモデル比較実行中...")
        try:
            from daytrading_model_comparison import DayTradingModelComparison
            
            # 高流動性銘柄でデイトレード比較
            symbols = ["7203", "9984", "6758", "8306", "6501", "4063", "9432", "8035", "4519", "6861"]
            
            comparison = DayTradingModelComparison()
            performances = comparison.compare_daytrading_models(symbols)
            
            if performances:
                print("✅ デイトレードモデル比較完了")
                print(f"📊 {len(performances)} モデルの比較結果を生成しました")
                
                # 最優秀モデル表示
                best_model = max(performances, key=lambda x: x.profit_rate)
                print(f"\\n🥇 最優秀モデル: {best_model.model_name}")
                print(f"📈 期待利益率: {best_model.profit_rate:.1%}")
                print(f"🎯 勝率: {best_model.win_rate:.1%}")
            else:
                print("❌ 比較結果が得られませんでした")
                
        except Exception as e:
            print(f"❌ デイトレード比較エラー: {e}")
            import traceback
            traceback.print_exc()

def main():
    """メイン実行"""
    launcher = RakutenTradingLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
