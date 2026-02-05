"""
データ収集システムのテスト（Excel無しモード）
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from enhanced_data_collector import SimpleDataCollector
import pandas as pd
from datetime import datetime, timedelta

def test_data_collection_system():
    """データ収集システムのテスト"""
    print("=== データ収集システムテスト ===")
    
    try:
        # インスタンス作成
        collector = SimpleDataCollector()
        print("✓ SimpleDataCollectorインスタンス作成成功")
        
        # データベース初期化確認
        count = collector.get_data_count()
        print(f"✓ データベース接続成功 (現在のデータ件数: {count}件)")
        
        # 銘柄リスト読み込み
        symbols = collector.load_symbols()
        print(f"✓ 銘柄リスト読み込み成功 ({len(symbols)}銘柄)")
        print(f"  対象銘柄: {symbols[:5]}...")
        
        # テストデータをデータベースに直接挿入
        test_symbol = "7203"
        test_data = []
        
        # 5分足データを3件作成
        base_time = datetime.now().replace(second=0, microsecond=0)
        for i in range(3):
            test_data.append({
                'datetime': base_time - timedelta(minutes=5*i),
                'open': 1000 + i,
                'high': 1010 + i,
                'low': 995 + i,
                'close': 1005 + i,
                'volume': 100000 + i*1000
            })
        
        # DataFrameに変換
        test_df = pd.DataFrame(test_data)
        print(f"✓ テストデータ作成成功 ({len(test_df)}件)")
        
        # データベースに保存
        collector.save_to_database(test_symbol, test_df, "5M")
        print("✓ データベース保存成功")
        
        # 保存されたデータの確認
        new_count = collector.get_data_count()
        print(f"✓ データ保存確認 (データ件数: {count} → {new_count}件)")
        
        # データベース内容確認
        import sqlite3
        conn = sqlite3.connect(collector.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chart_data WHERE symbol = ? ORDER BY datetime DESC", (test_symbol,))
        rows = cursor.fetchall()
        conn.close()
        
        print(f"✓ データ取得確認 ({len(rows)}件)")
        for row in rows:
            print(f"  {row[1]} {row[2]} {row[3]} - O:{row[4]} H:{row[5]} L:{row[6]} C:{row[7]} V:{row[8]}")
        
        print("\n=== データ収集システムテスト完了 ===")
        print("システムは正常に動作します")
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_data_collection_system()
