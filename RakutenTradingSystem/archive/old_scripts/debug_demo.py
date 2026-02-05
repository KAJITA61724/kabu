"""
デバッグ用シンプルデモテスト
どこで止まっているか特定する
"""

import sys
from pathlib import Path
import logging
import sqlite3

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# ログ設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_database():
    """データベースの内容を確認"""
    print("🔍 データベース内容確認")
    print("=" * 50)
    
    try:
        # メインデータベース確認
        conn = sqlite3.connect("enhanced_trading.db")
        cursor = conn.cursor()
        
        # five_minute_dataテーブルの件数確認
        cursor.execute("SELECT COUNT(*) FROM five_minute_data")
        count = cursor.fetchone()[0]
        print(f"📊 five_minute_data: {count}件")
        
        # 銘柄一覧確認
        cursor.execute("SELECT DISTINCT symbol FROM five_minute_data LIMIT 5")
        symbols = [row[0] for row in cursor.fetchall()]
        print(f"📈 銘柄例: {symbols}")
        
        # 日付範囲確認
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM five_minute_data")
        min_date, max_date = cursor.fetchone()
        print(f"📅 期間: {min_date} - {max_date}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ データベースエラー: {e}")

def test_demo_import():
    """デモデータインポートをテスト"""
    print("\n🔄 デモデータインポートテスト")
    print("=" * 50)
    
    try:
        from enhanced_demo_trading import EnhancedHistoricalDataCollector
        
        # データ収集器作成
        print("1. データ収集器作成中...")
        data_collector = EnhancedHistoricalDataCollector()
        
        # 利用可能データ確認
        print("2. 利用可能データ確認中...")
        min_date, max_date = data_collector.get_available_data_range()
        print(f"   初期データ範囲: {min_date} - {max_date}")
        
        if not min_date or not max_date:
            print("3. メインデータベースからインポート中...")
            data_collector.import_from_main_database()
            
            # 再確認
            print("4. インポート後データ確認中...")
            min_date, max_date = data_collector.get_available_data_range()
            print(f"   インポート後範囲: {min_date} - {max_date}")
        
        if min_date and max_date:
            print("✅ データインポート成功")
            return True
        else:
            print("❌ データインポート失敗")
            return False
            
    except Exception as e:
        print(f"❌ インポートエラー: {e}")
        return False

def test_demo_simulation():
    """デモシミュレーションをテスト"""
    print("\n🎮 デモシミュレーションテスト")
    print("=" * 50)
    
    try:
        from enhanced_demo_trading import EnhancedDemoTradingSimulator, DemoMethod, EnhancedHistoricalDataCollector
        from datetime import datetime, timedelta
        
        # データ収集器作成
        print("1. データ収集器作成中...")
        data_collector = EnhancedHistoricalDataCollector()
        
        # データ範囲取得
        print("2. データ範囲取得中...")
        min_date, max_date = data_collector.get_available_data_range()
        
        if not min_date or not max_date:
            print("   データインポート中...")
            data_collector.import_from_main_database()
            min_date, max_date = data_collector.get_available_data_range()
        
        if not min_date or not max_date:
            print("❌ データが利用できません")
            return False
        
        print(f"   データ範囲: {min_date} - {max_date}")
        
        # シミュレーター作成
        print("3. シミュレーター作成中...")
        config = {
            'capital': 1000000,
            'max_positions': 2,
            'demo_mode': True
        }
        
        simulator = EnhancedDemoTradingSimulator(config, data_collector)
        
        # 短期間でテスト（1日のみ）
        print("4. 短期間テスト実行中...")
        end_date = max_date
        start_date = end_date - timedelta(days=1)
        
        print(f"   テスト期間: {start_date} - {end_date}")
        
        # 1つの方法のみテスト
        methods = [DemoMethod.METHOD_2_STRATEGY]  # 戦略のみ
        
        print("5. シミュレーション実行中...")
        simulator.run_enhanced_demo_simulation(start_date, end_date, methods)
        
        print("✅ シミュレーション完了")
        return True
        
    except Exception as e:
        print(f"❌ シミュレーションエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    print("🔧 デモシステムデバッグ")
    print("=" * 50)
    
    # ステップ1: データベース確認
    test_database()
    
    # ステップ2: インポートテスト
    if test_demo_import():
        # ステップ3: シミュレーションテスト
        test_demo_simulation()
    
    print("\n" + "=" * 50)
    print("🔧 デバッグ完了")

if __name__ == "__main__":
    main()
