"""
MarketSpeedⅡ CSV出力ベースのデータ収集システム
RSS機能が使用できない場合の代替手段
"""

import pandas as pd
import sqlite3
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import glob
import shutil

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CSVDataCollector:
    """CSVファイルベースのデータ収集システム"""
    
    def __init__(self, db_path: str = "enhanced_trading.db"):
        self.db_path = db_path
        self.symbols = []
        
        # CSV監視フォルダ
        self.csv_folder = r"C:\Users\akane\RakutenTradingSystem\csv_data"
        self.ensure_csv_folder()
        
    def ensure_csv_folder(self):
        """CSV監視フォルダの確保"""
        if not os.path.exists(self.csv_folder):
            os.makedirs(self.csv_folder)
            logging.info(f"CSV監視フォルダを作成: {self.csv_folder}")
    
    def load_symbols(self) -> List[str]:
        """銘柄リスト読み込み"""
        try:
            # プライム銘柄リスト読み込み
            df = pd.read_csv('prime_symbols.csv')
            
            # 30万株以上の銘柄フィルタ
            if 'avg_volume' in df.columns:
                filtered_df = df[df['avg_volume'] >= 300000]
            else:
                filtered_df = df
            
            self.symbols = filtered_df['symbol'].astype(str).tolist()
            logging.info(f"対象銘柄: {len(self.symbols)}銘柄")
            return self.symbols
            
        except FileNotFoundError:
            logging.error("prime_symbols.csvが見つかりません")
            return []
    
    def create_sample_csv(self):
        """サンプルCSVファイルを作成（MarketSpeedⅡの出力形式に合わせる）"""
        sample_data = {
            '銘柄コード': ['1301', '7203', '6758', '9984', '8306'],
            '銘柄名': ['極洋', 'トヨタ', 'ソニー', 'SBG', '三菱UFJ'],
            '現在値': [2850, 2750, 13500, 12800, 1250],
            '始値': [2820, 2740, 13450, 12750, 1240],
            '高値': [2870, 2760, 13600, 12900, 1260],
            '安値': [2810, 2730, 13400, 12700, 1235],
            '出来高': [450000, 8500000, 2100000, 3200000, 15000000],
            '売買代金': [1282500000, 23375000000, 28350000000, 40960000000, 18750000000],
            '前日比': [30, 10, 50, 50, 10],
            '前日比率': [1.07, 0.36, 0.37, 0.39, 0.81],
            '更新時刻': ['09:25:00', '09:25:00', '09:25:00', '09:25:00', '09:25:00']
        }
        
        df = pd.DataFrame(sample_data)
        
        # 現在時刻でファイル名作成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"market_data_{timestamp}.csv"
        filepath = os.path.join(self.csv_folder, filename)
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logging.info(f"サンプルCSVファイル作成: {filename}")
        
        return filepath
    
    def scan_csv_files(self) -> List[str]:
        """CSVファイルをスキャン"""
        pattern = os.path.join(self.csv_folder, "*.csv")
        csv_files = glob.glob(pattern)
        
        # 最新のファイルのみ処理
        if csv_files:
            # 更新時刻順にソート
            csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_file = csv_files[0]
            
            # 5分以内のファイルのみ処理
            file_time = os.path.getmtime(latest_file)
            current_time = time.time()
            
            if current_time - file_time <= 300:  # 5分以内
                return [latest_file]
        
        return []
    
    def process_csv_file(self, csv_file: str) -> List[Dict]:
        """CSVファイルを処理してデータ変換"""
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            collected_data = []
            current_time = datetime.now()
            
            # 5分単位に丸める
            rounded_time = current_time.replace(
                minute=(current_time.minute // 5) * 5,
                second=0,
                microsecond=0
            )
            
            for _, row in df.iterrows():
                try:
                    # 必要な列の存在確認
                    required_columns = ['銘柄コード', '現在値', '始値', '高値', '安値', '出来高']
                    if not all(col in row for col in required_columns):
                        continue
                    
                    # データ変換
                    data_point = {
                        'symbol': str(row['銘柄コード']),
                        'timestamp': rounded_time,
                        'open_price': float(row['始値']) if pd.notna(row['始値']) else float(row['現在値']),
                        'high_price': float(row['高値']) if pd.notna(row['高値']) else float(row['現在値']),
                        'low_price': float(row['安値']) if pd.notna(row['安値']) else float(row['現在値']),
                        'close_price': float(row['現在値']),
                        'volume': int(row['出来高']) if pd.notna(row['出来高']) else 0,
                        'turnover': float(row['売買代金']) if '売買代金' in row and pd.notna(row['売買代金']) else 0.0,
                        'vwap': float(row['現在値'])  # VWAPがない場合は現在値で代用
                    }
                    
                    collected_data.append(data_point)
                    
                except Exception as e:
                    logging.error(f"行処理エラー {row.get('銘柄コード', 'N/A')}: {e}")
                    continue
            
            logging.info(f"CSV処理完了: {len(collected_data)}銘柄")
            return collected_data
            
        except Exception as e:
            logging.error(f"CSVファイル処理エラー {csv_file}: {e}")
            return []
    
    def save_data(self, data_list: List[Dict]):
        """データをデータベースに保存"""
        if not data_list:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        
        for data in data_list:
            try:
                # 5分足データ保存
                cursor.execute('''
                    INSERT OR REPLACE INTO five_minute_data 
                    (symbol, timestamp, open_price, high_price, low_price, 
                     close_price, volume, turnover, vwap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['symbol'], data['timestamp'], data['open_price'],
                    data['high_price'], data['low_price'], data['close_price'],
                    data['volume'], data['turnover'], data['vwap']
                ))
                
                saved_count += 1
                
            except Exception as e:
                logging.error(f"データ保存エラー {data['symbol']}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        logging.info(f"データ保存完了: {saved_count}銘柄")
    
    def archive_processed_file(self, csv_file: str):
        """処理済みファイルをアーカイブ"""
        try:
            archive_folder = os.path.join(self.csv_folder, "processed")
            if not os.path.exists(archive_folder):
                os.makedirs(archive_folder)
            
            filename = os.path.basename(csv_file)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"processed_{timestamp}_{filename}"
            archive_path = os.path.join(archive_folder, archive_name)
            
            shutil.move(csv_file, archive_path)
            logging.info(f"ファイルアーカイブ: {archive_name}")
            
        except Exception as e:
            logging.error(f"アーカイブエラー: {e}")
    
    def run_collection(self):
        """データ収集実行"""
        logging.info("CSV監視データ収集開始")
        
        # CSVファイルスキャン
        csv_files = self.scan_csv_files()
        
        if not csv_files:
            logging.info("処理対象のCSVファイルが見つかりません")
            return 0
        
        total_collected = 0
        
        for csv_file in csv_files:
            logging.info(f"処理開始: {os.path.basename(csv_file)}")
            
            # CSVファイル処理
            collected_data = self.process_csv_file(csv_file)
            
            # データ保存
            self.save_data(collected_data)
            
            # ファイルアーカイブ
            self.archive_processed_file(csv_file)
            
            total_collected += len(collected_data)
        
        logging.info(f"CSV監視データ収集完了: {total_collected}銘柄")
        return total_collected
    
    def create_instruction_file(self):
        """MarketSpeedⅡでのCSV出力手順を作成"""
        instructions = """
MarketSpeedⅡでのCSV出力手順

1. MarketSpeedⅡを起動してログイン

2. 「銘柄リスト」画面を開く

3. 必要な銘柄を登録
   - prime_symbols.csvの銘柄を参考に登録
   - 216銘柄程度を登録

4. 表示項目を設定
   - 銘柄コード
   - 銘柄名
   - 現在値
   - 始値
   - 高値
   - 安値
   - 出来高
   - 売買代金
   - 前日比
   - 前日比率
   - 更新時刻

5. CSV出力実行
   - 「ファイル」→「CSV出力」を選択
   - 保存先を以下に設定：
     C:\\Users\\akane\\RakutenTradingSystem\\csv_data\\
   - ファイル名：market_data_YYYYMMDD_HHMMSS.csv
   - エンコーディング：UTF-8 BOM付き

6. 5分間隔で手動実行
   - 9:25, 9:30, 9:35, 9:40... の5分間隔で実行
   - または自動化スクリプトを使用

注意事項：
- MarketSpeedⅡにログインしていることを確認
- データ配信が有効になっていることを確認
- CSV出力は手動実行が必要
"""
        
        instruction_file = os.path.join(self.csv_folder, "CSV出力手順.txt")
        with open(instruction_file, 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        logging.info(f"手順書作成: {instruction_file}")

# 使用例
if __name__ == "__main__":
    print("=== MarketSpeedⅡ CSV収集システム ===")
    
    collector = CSVDataCollector()
    
    # 手順書作成
    collector.create_instruction_file()
    
    # サンプルCSVファイル作成
    collector.create_sample_csv()
    
    # データ収集実行
    result_count = collector.run_collection()
    
    print(f"収集完了: {result_count}銘柄のデータを取得しました")
    
    # データベース確認
    conn = sqlite3.connect("enhanced_trading.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM five_minute_data")
    total_records = cursor.fetchone()[0]
    conn.close()
    
    print(f"データベース総件数: {total_records}件")
    print(f"CSV監視フォルダ: {collector.csv_folder}")
    print("CSV出力手順.txtを確認してMarketSpeedⅡでCSV出力を実行してください")
