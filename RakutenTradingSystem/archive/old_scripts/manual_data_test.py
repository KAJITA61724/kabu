"""
手動データ収集テスト - MarketSpeedⅡ関数の動作確認
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.enhanced_data_collector import EnhancedDataCollector
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_data_collection():
    """データ収集テスト"""
    print("=== 手動データ収集テスト ===")
    
    # データコレクター初期化
    collector = EnhancedDataCollector()
    
    # Excel接続
    if not collector.initialize_excel_connection():
        print("Excel接続失敗")
        return False
    
    # 銘柄読み込み
    symbols = collector.load_nikkei225_symbols()
    print(f"対象銘柄数: {len(symbols)}")
    
    # 最初の5銘柄のみでテスト
    collector.symbols = symbols[:5]
    print(f"テスト銘柄: {collector.symbols}")
    
    # データ収集実行
    print("データ収集開始...")
    collected_data = collector.collect_5min_data()
    
    print(f"収集結果: {len(collected_data)}件")
    
    # 結果詳細表示
    for data in collected_data:
        print(f"銘柄: {data['symbol']}")
        print(f"  現在値: {data['close_price']}")
        print(f"  出来高: {data['volume']}")
        print(f"  VWAP: {data['vwap']}")
        print("---")
    
    # データ保存
    if collected_data:
        collector.save_5min_data(collected_data)
        print("データ保存完了")
    else:
        print("保存するデータがありません")
    
    # クリーンアップ
    try:
        collector.workbook.Close(False)
        collector.excel_app.Quit()
        print("Excel終了")
    except:
        pass
    
    return len(collected_data) > 0

if __name__ == "__main__":
    success = test_data_collection()
    print(f"テスト結果: {'成功' if success else '失敗'}")
