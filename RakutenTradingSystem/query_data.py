"""
データベースからデータを取得するためのユーザーインターフェース
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.data_reader import DataReader
import pandas as pd
from datetime import datetime, timedelta

def interactive_data_query():
    """インタラクティブなデータクエリツール"""
    
    reader = DataReader('trading_data.db')
    
    print("=== データベースクエリツール ===")
    print()
    
    # 利用可能な銘柄を表示
    symbols = reader.get_available_symbols()
    if not symbols:
        print("❌ データベースにデータがありません")
        print("まずは 'python run_data_collection.py' でデータを収集してください")
        return
    
    print(f"利用可能な銘柄: {', '.join(symbols)}")
    print()
    
    while True:
        print("--- 操作メニュー ---")
        print("1. 最新データ取得")
        print("2. 期間指定データ取得")
        print("3. 複数銘柄データ取得")
        print("4. 統計情報表示")
        print("5. データベースサマリー")
        print("6. 終了")
        print()
        
        choice = input("選択してください (1-6): ").strip()
        
        if choice == '1':
            # 最新データ取得
            symbol = input(f"銘柄コード ({', '.join(symbols)}): ").strip()
            if symbol not in symbols:
                print("❌ 無効な銘柄コードです")
                continue
                
            limit = input("取得件数 (デフォルト: 10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            
            print(f"\n{symbol}の最新データ ({limit}件):")
            df = reader.get_latest_data(symbol, limit=limit)
            if not df.empty:
                print(df.to_string(index=False))
            else:
                print("データなし")
            print()
            
        elif choice == '2':
            # 期間指定データ取得
            symbol = input(f"銘柄コード ({', '.join(symbols)}): ").strip()
            if symbol not in symbols:
                print("❌ 無効な銘柄コードです")
                continue
                
            start_date = input("開始日時 (YYYY-MM-DD HH:MM:SS): ").strip()
            end_date = input("終了日時 (YYYY-MM-DD HH:MM:SS): ").strip()
            
            try:
                df = reader.get_data_by_date_range(symbol, start_date, end_date)
                if not df.empty:
                    print(f"\n{symbol}の期間データ ({len(df)}件):")
                    print(df.to_string(index=False))
                else:
                    print("指定期間にデータがありません")
            except Exception as e:
                print(f"❌ エラー: {e}")
            print()
            
        elif choice == '3':
            # 複数銘柄データ取得
            selected_symbols = input(f"銘柄コード (カンマ区切り、全て: all): ").strip()
            if selected_symbols.lower() == 'all':
                selected_symbols = symbols
            else:
                selected_symbols = [s.strip() for s in selected_symbols.split(',')]
                selected_symbols = [s for s in selected_symbols if s in symbols]
            
            if not selected_symbols:
                print("❌ 有効な銘柄コードがありません")
                continue
            
            limit = input("各銘柄の取得件数 (デフォルト: 5): ").strip()
            limit = int(limit) if limit.isdigit() else 5
            
            print(f"\n複数銘柄データ (各{limit}件):")
            results = reader.get_multiple_symbols_data(selected_symbols, limit=limit)
            
            for symbol, df in results.items():
                print(f"\n【{symbol}】")
                if not df.empty:
                    print(df.to_string(index=False))
                else:
                    print("データなし")
            print()
            
        elif choice == '4':
            # 統計情報表示
            symbol = input(f"銘柄コード ({', '.join(symbols)}): ").strip()
            if symbol not in symbols:
                print("❌ 無効な銘柄コードです")
                continue
                
            days = input("統計期間 (日数、デフォルト: 30): ").strip()
            days = int(days) if days.isdigit() else 30
            
            print(f"\n{symbol}の統計情報 (過去{days}日):")
            stats = reader.get_data_statistics(symbol, days=days)
            
            if stats['data_count'] > 0:
                print(f"  データ件数: {stats['data_count']}")
                print(f"  平均価格: {stats['avg_price']}")
                print(f"  最高価格: {stats['max_price']}")
                print(f"  最低価格: {stats['min_price']}")
                print(f"  平均出来高: {stats['avg_volume']:,}")
                print(f"  最大出来高: {stats['max_volume']:,}")
            else:
                print("  データなし")
            print()
            
        elif choice == '5':
            # データベースサマリー
            print("\nデータベースサマリー:")
            summary = reader.get_data_summary()
            print(f"  総銘柄数: {summary['total_symbols']}")
            print(f"  総レコード数: {summary['total_records']}")
            print()
            
            for detail in summary['symbol_details']:
                print(f"  {detail['symbol']} ({detail['timeframe']}): {detail['data_count']}件")
                print(f"    期間: {detail['first_date']} ～ {detail['last_date']}")
            print()
            
        elif choice == '6':
            print("終了します")
            break
            
        else:
            print("❌ 無効な選択です")
            print()

if __name__ == "__main__":
    interactive_data_query()
