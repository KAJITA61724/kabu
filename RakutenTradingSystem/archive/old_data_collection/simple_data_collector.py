"""
シンプルなデータ収集スクリプト
まずはデータ収集から開始
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.enhanced_data_collector import EnhancedDataCollector
import time
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)

def main():
    """データ収集を開始"""
    try:
        print("楽天トレードシステム - データ収集開始")
        print("=" * 50)
        
        # データ収集インスタンスを作成
        collector = EnhancedDataCollector()
        
        # Excel接続を初期化
        print("Excel接続を初期化中...")
        if not collector.initialize_excel_connection():
            print("Excel接続に失敗しました。デモモードで続行します。")
        else:
            print("Excel接続が正常に確立されました。")
        
        print("データ収集を開始します...")
        print("Ctrl+Cで停止できます")
        
        # 継続的なデータ収集
        while True:
            try:
                # 5分ごとにデータ収集
                print(f"データ収集中... {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # ここで実際のデータ収集処理
                data = collector.collect_5min_data()
                if data:
                    print(f"  -> {len(data)} 銘柄のデータを収集しました")
                
                time.sleep(300)  # 5分待機
                
            except KeyboardInterrupt:
                print("\nデータ収集を停止します...")
                break
            except Exception as e:
                logging.error(f"データ収集エラー: {e}")
                time.sleep(60)  # エラー時は1分待機して再試行
                
    except Exception as e:
        logging.error(f"システム初期化エラー: {e}")
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
