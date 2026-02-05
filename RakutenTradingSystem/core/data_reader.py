"""
データベースからデータを取り出すユーティリティ
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging

class DataReader:
    """データベースからデータを読み込むクラス"""
    
    def __init__(self, db_path='../trading_data.db'):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
    def get_latest_data(self, symbol: str, timeframe: str = "5M", limit: int = 100) -> pd.DataFrame:
        """最新のデータを取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT datetime, open_price, high_price, low_price, close_price, volume
                FROM chart_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY datetime DESC
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=[symbol, timeframe, limit])
            conn.close()
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')  # 時系列順に並び替え
                
            return df
            
        except Exception as e:
            self.logger.error(f"データ取得エラー: {e}")
            return pd.DataFrame()
    
    def get_data_by_date_range(self, symbol: str, start_date: str, end_date: str, timeframe: str = "5M") -> pd.DataFrame:
        """指定期間のデータを取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT datetime, open_price, high_price, low_price, close_price, volume
                FROM chart_data 
                WHERE symbol = ? AND timeframe = ? 
                AND datetime >= ? AND datetime <= ?
                ORDER BY datetime ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=[symbol, timeframe, start_date, end_date])
            conn.close()
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                
            return df
            
        except Exception as e:
            self.logger.error(f"期間指定データ取得エラー: {e}")
            return pd.DataFrame()
    
    def get_multiple_symbols_data(self, symbols: list, timeframe: str = "5M", limit: int = 100) -> dict:
        """複数銘柄のデータを一括取得"""
        results = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            for symbol in symbols:
                query = '''
                    SELECT datetime, open_price, high_price, low_price, close_price, volume
                    FROM chart_data 
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY datetime DESC
                    LIMIT ?
                '''
                
                df = pd.read_sql_query(query, conn, params=[symbol, timeframe, limit])
                
                if not df.empty:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.sort_values('datetime')
                    results[symbol] = df
                else:
                    results[symbol] = pd.DataFrame()
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"複数銘柄データ取得エラー: {e}")
            return {}
    
    def get_data_statistics(self, symbol: str, timeframe: str = "5M", days: int = 30) -> dict:
        """データ統計情報を取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 過去N日間のデータを取得
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            query = '''
                SELECT 
                    COUNT(*) as data_count,
                    AVG(close_price) as avg_price,
                    MAX(high_price) as max_price,
                    MIN(low_price) as min_price,
                    AVG(volume) as avg_volume,
                    MAX(volume) as max_volume
                FROM chart_data 
                WHERE symbol = ? AND timeframe = ?
                AND datetime >= ?
            '''
            
            cursor = conn.cursor()
            cursor.execute(query, [symbol, timeframe, start_date.strftime('%Y-%m-%d %H:%M:%S')])
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] > 0:
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'data_count': result[0],
                    'avg_price': round(result[1], 2) if result[1] else 0,
                    'max_price': result[2] if result[2] else 0,
                    'min_price': result[3] if result[3] else 0,
                    'avg_volume': int(result[4]) if result[4] else 0,
                    'max_volume': result[5] if result[5] else 0,
                    'period_days': days
                }
            else:
                return {'symbol': symbol, 'data_count': 0}
                
        except Exception as e:
            self.logger.error(f"統計情報取得エラー: {e}")
            return {'symbol': symbol, 'data_count': 0}
    
    def get_available_symbols(self) -> list:
        """利用可能な銘柄リストを取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT DISTINCT symbol 
                FROM chart_data 
                ORDER BY symbol
            '''
            
            cursor = conn.cursor()
            cursor.execute(query)
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return symbols
            
        except Exception as e:
            self.logger.error(f"銘柄リスト取得エラー: {e}")
            return []
    
    def get_data_summary(self) -> dict:
        """データベース全体のサマリー情報を取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    symbol,
                    timeframe,
                    COUNT(*) as data_count,
                    MIN(datetime) as first_date,
                    MAX(datetime) as last_date
                FROM chart_data 
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            '''
            
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            
            summary = {
                'total_symbols': len(set([row[0] for row in results])),
                'total_records': sum([row[2] for row in results]),
                'symbol_details': []
            }
            
            for row in results:
                summary['symbol_details'].append({
                    'symbol': row[0],
                    'timeframe': row[1],
                    'data_count': row[2],
                    'first_date': row[3],
                    'last_date': row[4]
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"サマリー取得エラー: {e}")
            return {'total_symbols': 0, 'total_records': 0}


def main():
    """メイン実行関数"""
    # ログ設定
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    reader = DataReader()
    
    print("=== データベースからのデータ取得デモ ===")
    print()
    
    # 利用可能な銘柄を表示
    symbols = reader.get_available_symbols()
    print(f"利用可能な銘柄: {symbols}")
    print()
    
    # データベースサマリー
    summary = reader.get_data_summary()
    print("データベースサマリー:")
    print(f"  総銘柄数: {summary['total_symbols']}")
    print(f"  総レコード数: {summary['total_records']}")
    print()
    
    for detail in summary['symbol_details']:
        print(f"  {detail['symbol']} ({detail['timeframe']}): {detail['data_count']}件")
        print(f"    期間: {detail['first_date']} ～ {detail['last_date']}")
    print()
    
    # 個別銘柄データ取得例
    if symbols:
        symbol = symbols[0]
        print(f"{symbol}の最新データ（5件）:")
        df = reader.get_latest_data(symbol, limit=5)
        if not df.empty:
            print(df)
        else:
            print("  データなし")
        print()
        
        # 統計情報
        stats = reader.get_data_statistics(symbol)
        if stats['data_count'] > 0:
            print(f"{symbol}の統計情報:")
            print(f"  データ件数: {stats['data_count']}")
            print(f"  平均価格: {stats['avg_price']}")
            print(f"  最高価格: {stats['max_price']}")
            print(f"  最低価格: {stats['min_price']}")
            print(f"  平均出来高: {stats['avg_volume']:,}")


if __name__ == "__main__":
    main()
