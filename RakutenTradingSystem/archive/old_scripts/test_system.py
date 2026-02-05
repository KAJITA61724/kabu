"""
データ収集システムの動作確認
"""

import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

try:
    from enhanced_data_collector import SimpleDataCollector
    
    print("=== データ収集システム動作確認 ===")
    print("✓ SimpleDataCollectorのインポート成功")
    
    # インスタンス作成
    collector = SimpleDataCollector()
    print("✓ SimpleDataCollectorのインスタンス作成成功")
    
    # データベース初期化確認
    count = collector.get_data_count()
    print(f"✓ データベース接続成功 (現在のデータ件数: {count}件)")
    
    # 銘柄リスト読み込み確認
    symbols = collector.load_symbols()
    print(f"✓ 銘柄リスト読み込み成功 ({len(symbols)}銘柄)")
    
    print("\n=== 動作確認完了 ===")
    print("システムは正常に動作します。")
    print("実行するには: python run_data_collection.py")
    
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    print("core/enhanced_data_collector.pyを確認してください")
    
except Exception as e:
    print(f"❌ エラー: {e}")
    print("システムの設定を確認してください")
