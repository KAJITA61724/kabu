"""
MSGETCHART関数を使用したシンプルなデータ収集実行スクリプト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from enhanced_data_collector import SimpleDataCollector
import logging

def main():
    """メイン実行関数"""
    print("=== MarketSpeedⅡ MSGETCHART分足データ収集 ===")
    print("- ExcelのMSGETCHART関数を使用")
    print("- PythonでExcelから結果を読み取り")
    print("- SQLiteデータベースに保存")
    print("- シンプルで効率的な設計")
    print()
    
    # 設定入力
    timeframe = input("足種を選択してください (1M/5M/15M/30M/1H) [5M]: ").strip()
    if not timeframe:
        timeframe = "5M"
    
    count = input("取得本数を入力してください [1000]: ").strip()
    if not count:
        count = 1000
    else:
        try:
            count = int(count)
        except ValueError:
            count = 1000
    
    print(f"\n設定:")
    print(f"  足種: {timeframe}")
    print(f"  取得本数: {count}")
    print(f"  保存先: SQLiteデータベース (trading_data.db)")
    print()
    
    # 実行確認
    confirm = input("実行してよろしいですか? (y/N): ")
    if confirm.lower() != 'y':
        print("中止しました。")
        return
    
    # データ収集実行
    try:
        collector = SimpleDataCollector()
        collector.run_collection(timeframe=timeframe, count=count)
        
        # 結果確認
        total_count = collector.get_data_count()
        print(f"\n=== 収集完了 ===")
        print(f"データベース内の総データ件数: {total_count}件")
        print(f"データベースファイル: trading_data.db")
        
    except KeyboardInterrupt:
        print("\nユーザーによる中断")
    except Exception as e:
        logging.error(f"実行エラー: {e}")
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
