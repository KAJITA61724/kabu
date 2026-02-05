"""
全データベースの内容確認
"""
import sqlite3
import pandas as pd
import os

def check_all_databases():
    """すべてのデータベースファイルの内容を確認"""
    
    db_files = [
        'demo_trading.db',
        'enhanced_demo_trading.db', 
        'enhanced_trading.db',
        'fundamental_data.db',
        'trading_data.db'
    ]
    
    print("=== 全データベースファイル確認 ===")
    print()
    
    for db_file in db_files:
        if os.path.exists(db_file):
            print(f"【{db_file}】")
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # テーブル一覧を取得
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                if tables:
                    print(f"  テーブル数: {len(tables)}")
                    for table in tables:
                        table_name = table[0]
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        print(f"    {table_name}: {count}件")
                        
                        # データがある場合は最新の数件を表示
                        if count > 0:
                            try:
                                # まずは列名を取得
                                cursor.execute(f"PRAGMA table_info({table_name})")
                                columns = cursor.fetchall()
                                col_names = [col[1] for col in columns]
                                
                                # データを取得
                                cursor.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 3")
                                rows = cursor.fetchall()
                                
                                print(f"      最新データ (最大3件):")
                                for row in rows:
                                    row_data = dict(zip(col_names, row))
                                    print(f"        {row_data}")
                                    
                            except Exception as e:
                                print(f"        データ取得エラー: {e}")
                else:
                    print("  テーブルなし")
                    
                conn.close()
                
            except Exception as e:
                print(f"  エラー: {e}")
            
            print()
        else:
            print(f"【{db_file}】: ファイルが存在しません")
            print()

if __name__ == "__main__":
    check_all_databases()
