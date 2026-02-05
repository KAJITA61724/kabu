"""
戦略比較デモ実行ツール
ML戦略 vs テクニカル戦略の比較検証
"""

import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from enhanced_demo_trading import EnhancedDemoTradingSimulator, DemoMethod, EnhancedHistoricalDataCollector

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """戦略比較デモ実行"""
    print("⚡ 楽天トレードシステム - 戦略比較デモ")
    print("=" * 60)
    
    try:
        # データ収集器初期化
        print("📊 データ収集器初期化中...")
        data_collector = EnhancedHistoricalDataCollector()
        
        # データ範囲確認
        min_date, max_date = data_collector.get_available_data_range()
        
        if not min_date or not max_date:
            print("📥 メインデータベースからデータをインポート中...")
            data_collector.import_from_main_database()
            min_date, max_date = data_collector.get_available_data_range()
        
        if not min_date or not max_date:
            print("❌ データが利用できません")
            return
        
        print(f"📅 利用可能データ: {min_date} - {max_date}")
        
        # シミュレーター設定
        config = {
            'capital': 2000000,      # 初期資本200万円
            'max_positions': 3,      # 最大3ポジション
            'demo_mode': True
        }
        
        print("🎮 シミュレーター初期化中...")
        simulator = EnhancedDemoTradingSimulator(config, data_collector)
        
        # デモ期間設定（最新の3営業日）
        end_date = max_date
        start_date = max(min_date, end_date - timedelta(days=3))
        
        print(f"📈 デモ期間: {start_date} - {end_date}")
        print("=" * 60)
        
        # 戦略比較デモ実行
        print("🚀 戦略比較デモ開始...")
        print("   - 方法1: ML（機械学習）戦略")
        print("   - 方法2: テクニカル戦略")
        print("=" * 60)
        
        # 両方の方法で比較
        methods = [DemoMethod.METHOD_1_ML, DemoMethod.METHOD_2_STRATEGY]
        
        simulator.run_enhanced_demo_simulation(start_date, end_date, methods)
        
        print("=" * 60)
        print("✅ 戦略比較デモ完了")
        print("📊 詳細レポートは reports/demo/ フォルダに保存されました")
        print("📈 各戦略の損益とパフォーマンス比較をご確認ください")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        logging.error(f"戦略比較デモエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
