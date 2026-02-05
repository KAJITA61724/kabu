"""
yfinanceを使った5分足データ取得テスト
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import logging

class YFinanceDataCollector:
    """yfinanceデータ収集クラス"""
    
    def __init__(self, db_path='trading_data.db'):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def get_5min_data(self, symbol: str, days_back: int = 1) -> pd.DataFrame:
        """5分足データを取得"""
        try:
            # 日本株のシンボル形式に変換
            yahoo_symbol = f"{symbol}.T"
            
            # 期間設定
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            self.logger.info(f"データ取得開始: {yahoo_symbol} ({start_date.date()} - {end_date.date()})")
            
            # yfinanceでデータ取得
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval="5m"
            )
            
            if data.empty:
                self.logger.warning(f"データが見つかりません: {yahoo_symbol}")
                return pd.DataFrame()
            
            # データフレーム整形
            df = data.reset_index()
            df['symbol'] = symbol
            df['datetime'] = df['Datetime']
            df['open_price'] = df['Open']
            df['high_price'] = df['High']
            df['low_price'] = df['Low']
            df['close_price'] = df['Close']
            df['volume'] = df['Volume']
            
            # 必要な列のみ選択
            df = df[['symbol', 'datetime', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']]
            
            self.logger.info(f"データ取得完了: {len(df)}件")
            return df
            
        except Exception as e:
            self.logger.error(f"データ取得エラー {symbol}: {e}")
            return pd.DataFrame()
    
    def save_to_database(self, df: pd.DataFrame):
        """データベースに保存"""
        if df.empty:
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # テーブル作成
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chart_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, datetime, timeframe)
                )
            ''')
            
            # データ保存
            for _, row in df.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO chart_data 
                    (symbol, datetime, timeframe, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['symbol'],
                    row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                    '5M',
                    row['open_price'],
                    row['high_price'],
                    row['low_price'],
                    row['close_price'],
                    int(row['volume'])
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"データベース保存完了: {len(df)}件")
            return True
            
        except Exception as e:
            self.logger.error(f"データベース保存エラー: {e}")
            return False
    
    def test_multiple_symbols(self, symbols: list, days_back: int = 1) -> dict:
        """複数銘柄のテスト"""
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"テスト開始: {symbol}")
            self.logger.info('='*50)
            
            df = self.get_5min_data(symbol, days_back)
            
            if not df.empty:
                # 基本統計
                stats = {
                    'data_count': len(df),
                    'date_range': f"{df['datetime'].min()} - {df['datetime'].max()}",
                    'price_range': f"{df['low_price'].min():.2f} - {df['high_price'].max():.2f}",
                    'avg_volume': int(df['volume'].mean()),
                    'total_volume': int(df['volume'].sum())
                }
                
                # データベース保存
                if self.save_to_database(df):
                    stats['database_saved'] = True
                else:
                    stats['database_saved'] = False
                
                results[symbol] = stats
                
                # サンプルデータ表示
                self.logger.info(f"データ件数: {len(df)}")
                self.logger.info(f"期間: {stats['date_range']}")
                self.logger.info(f"価格帯: {stats['price_range']}")
                self.logger.info(f"平均出来高: {stats['avg_volume']:,}")
                self.logger.info("\n最新5件:")
                self.logger.info(df.tail().to_string(index=False))
                
            else:
                results[symbol] = {'error': 'データ取得失敗'}
        
        return results


def test_yfinance_data():
    """yfinanceデータ取得テスト"""
    print("=== yfinance 5分足データ取得テスト ===")
    print()
    
    collector = YFinanceDataCollector()
    
    # テスト銘柄（日本の主要株）
    test_symbols = [
        '7203',  # トヨタ自動車
        '6758',  # ソニーグループ
        '8306',  # 三菱UFJ
        '9984',  # ソフトバンクグループ
        '6861'   # キーエンス
    ]
    
    print(f"テスト銘柄: {', '.join(test_symbols)}")
    print("昨日一日分の5分足データを取得します...")
    print()
    
    # データ取得テスト
    results = collector.test_multiple_symbols(test_symbols, days_back=2)
    
    print("\n" + "="*80)
    print("テスト結果サマリー")
    print("="*80)
    
    total_data = 0
    successful_symbols = []
    
    for symbol, result in results.items():
        if 'error' in result:
            print(f"❌ {symbol}: {result['error']}")
        else:
            print(f"✅ {symbol}: {result['data_count']}件")
            print(f"   期間: {result['date_range']}")
            print(f"   価格帯: {result['price_range']}")
            print(f"   平均出来高: {result['avg_volume']:,}")
            print(f"   DB保存: {'成功' if result['database_saved'] else '失敗'}")
            print()
            
            total_data += result['data_count']
            successful_symbols.append(symbol)
    
    print(f"成功銘柄数: {len(successful_symbols)}/{len(test_symbols)}")
    print(f"総データ件数: {total_data}")
    
    # データベースの状態確認
    print("\n" + "="*80)
    print("データベース確認")
    print("="*80)
    
    try:
        conn = sqlite3.connect(collector.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, COUNT(*) as count, MIN(datetime) as start_date, MAX(datetime) as end_date
            FROM chart_data 
            GROUP BY symbol 
            ORDER BY symbol
        ''')
        
        db_results = cursor.fetchall()
        
        for row in db_results:
            symbol, count, start_date, end_date = row
            print(f"{symbol}: {count}件 ({start_date} - {end_date})")
        
        conn.close()
        
    except Exception as e:
        print(f"データベース確認エラー: {e}")
    
    return results


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('yfinance_test.log', encoding='utf-8')
        ]
    )
    
    test_yfinance_data()
