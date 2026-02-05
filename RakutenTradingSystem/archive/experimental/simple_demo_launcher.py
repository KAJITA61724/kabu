"""
シンプルデモトレード実行ツール
サンプルデータを使ってすぐにデモトレードを試す
"""

import sys
from pathlib import Path
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from enhanced_demo_trading import EnhancedDemoTradingSimulator, DemoMethod, EnhancedHistoricalDataCollector
from datetime import datetime, timedelta

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """シンプルデモトレード実行"""
    print("🎮 楽天トレードシステム - デモトレード")
    print("=" * 50)
    
    try:
        # データ収集器初期化
        data_collector = EnhancedHistoricalDataCollector()
        
        # 利用可能データの確認
        min_date, max_date = data_collector.get_available_data_range()
        
        if not min_date or not max_date:
            print("❌ デモ用データがありません")
            print("� メインデータベースからデータをインポートします...")
            
            # メインデータベースからインポート
            data_collector.import_from_main_database()
            
            # 再確認
            min_date, max_date = data_collector.get_available_data_range()
            if not min_date or not max_date:
                print("❌ インポート後もデータがありません")
                print("�💡 先に generate_sample_data.py を実行してください")
                return
            else:
                print(f"✅ インポート完了: {min_date} - {max_date}")
        
        print(f"📅 利用可能データ: {min_date} - {max_date}")
        
        # シミュレーター設定
        config = {
            'capital': 2000000,      # 初期資本200万円
            'max_positions': 3,      # 最大3ポジション
            'demo_mode': True
        }
        
        simulator = EnhancedDemoTradingSimulator(config, data_collector)
        
        # デモ期間設定（最新の3営業日）
        end_date = max_date
        start_date = max(min_date, end_date - timedelta(days=5))
        
        print(f"📊 デモ期間: {start_date} - {end_date}")
        print()
        
        # デモ方法選択
        print("デモ方法を選択してください：")
        print("1. 方法1（ML）デモ")
        print("2. 方法2（戦略）デモ")
        print("3. 両方法比較デモ")
        print()
        
        choice = input("選択 (1-3): ").strip()
        
        if choice == "1":
            print("\n🤖 方法1（ML）デモ開始")
            methods = [DemoMethod.METHOD_1_ML]
        elif choice == "2":
            print("\n📊 方法2（戦略）デモ開始")
            methods = [DemoMethod.METHOD_2_STRATEGY]
        elif choice == "3":
            print("\n⚡ 両方法比較デモ開始")
            methods = [DemoMethod.METHOD_1_ML, DemoMethod.METHOD_2_STRATEGY]
        else:
            print("❌ 無効な選択です")
            return
        
        print("=" * 50)
        
        # デモ実行
        simulator.run_enhanced_demo_simulation(start_date, end_date, methods)
        
        print("\n✅ デモトレード完了")
        print("📊 レポートは reports/demo/ フォルダに保存されました")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        logging.error(f"デモトレードエラー: {e}")

if __name__ == "__main__":
    main()
