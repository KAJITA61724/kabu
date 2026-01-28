#!/usr/bin/env python3
"""
データ収集状況を詳細にチェックするスクリプト
"""
import sqlite3
import os
from datetime import datetime
from collections import defaultdict
import json

def check_database_status(db_path):
    """データベースの状態を詳細チェック"""
    if not os.path.exists(db_path):
        return None
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # ファイルサイズ
    file_size = os.path.getsize(db_path) / 1024 / 1024  # MB
    
    # テーブル一覧
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    result = {
        'path': db_path,
        'size_mb': round(file_size, 2),
        'tables': tables,
        'table_info': {}
    }
    
    # 各テーブルの情報
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            result['table_info'][table] = {'count': count}
        except:
            pass
    
    conn.close()
    return result

def analyze_chart_data(db_path='trading_data.db'):
    """chart_dataの詳細分析"""
    if not os.path.exists(db_path):
        print(f"❌ {db_path} が見つかりません")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=" * 80)
    print("📊 chart_data 詳細分析")
    print("=" * 80)
    
    # 総レコード数
    cursor.execute("SELECT COUNT(*) FROM chart_data")
    total_records = cursor.fetchone()[0]
    print(f"\n総レコード数: {total_records:,}")
    
    # 銘柄一覧
    cursor.execute("SELECT DISTINCT symbol FROM chart_data ORDER BY symbol")
    symbols = [row[0] for row in cursor.fetchall()]
    print(f"収集銘柄数: {len(symbols)}")
    print(f"銘柄コード: {', '.join(symbols)}")
    
    # タイムフレーム
    cursor.execute("SELECT DISTINCT timeframe FROM chart_data")
    timeframes = [row[0] for row in cursor.fetchall()]
    print(f"タイムフレーム: {', '.join(timeframes)}")
    
    # データ期間
    cursor.execute("SELECT MIN(datetime), MAX(datetime) FROM chart_data")
    min_date, max_date = cursor.fetchone()
    print(f"データ期間: {min_date} ～ {max_date}")
    
    # 銘柄別詳細
    print("\n" + "-" * 80)
    print("銘柄別データ統計")
    print("-" * 80)
    
    query = """
    SELECT 
        symbol,
        timeframe,
        COUNT(*) as records,
        MIN(datetime) as first_datetime,
        MAX(datetime) as last_datetime,
        COUNT(DISTINCT DATE(datetime)) as trading_days,
        AVG(volume) as avg_volume,
        MAX(high_price) as max_price,
        MIN(low_price) as min_price
    FROM chart_data
    GROUP BY symbol, timeframe
    ORDER BY symbol, timeframe
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    symbol_stats = defaultdict(lambda: {'timeframes': {}, 'total_records': 0})
    
    for row in results:
        symbol, tf, records, first, last, days, avg_vol, max_p, min_p = row
        symbol_stats[symbol]['total_records'] += records
        symbol_stats[symbol]['timeframes'][tf] = {
            'records': records,
            'first': first,
            'last': last,
            'trading_days': days,
            'avg_volume': avg_vol,
            'price_range': (min_p, max_p)
        }
    
    for symbol in sorted(symbol_stats.keys()):
        stats = symbol_stats[symbol]
        print(f"\n【{symbol}】")
        print(f"  総レコード数: {stats['total_records']:,}")
        
        for tf, tf_stats in stats['timeframes'].items():
            print(f"  - タイムフレーム: {tf}")
            print(f"    レコード数: {tf_stats['records']:,}")
            print(f"    取引日数: {tf_stats['trading_days']}日")
            print(f"    期間: {tf_stats['first']} ～ {tf_stats['last']}")
            if tf_stats['avg_volume']:
                print(f"    平均出来高: {tf_stats['avg_volume']:,.0f}")
            if tf_stats['price_range'][0]:
                print(f"    価格レンジ: {tf_stats['price_range'][0]:,.2f} ～ {tf_stats['price_range'][1]:,.2f}")
    
    # 日次統計
    print("\n" + "-" * 80)
    print("日次データ統計")
    print("-" * 80)
    
    query = """
    SELECT 
        DATE(datetime) as date,
        COUNT(*) as records,
        COUNT(DISTINCT symbol) as symbols,
        COUNT(DISTINCT timeframe) as timeframes
    FROM chart_data
    GROUP BY DATE(datetime)
    ORDER BY date
    """
    
    cursor.execute(query)
    daily_stats = cursor.fetchall()
    
    print(f"\n{'日付':<12} {'レコード数':>10} {'銘柄数':>8} {'TF数':>6}")
    print("-" * 40)
    for date, records, symbols, tfs in daily_stats:
        print(f"{date:<12} {records:>10,} {symbols:>8} {tfs:>6}")
    
    # データ品質チェック
    print("\n" + "-" * 80)
    print("データ品質チェック")
    print("-" * 80)
    
    # NULL値チェック
    null_checks = [
        ('symbol', 'symbol'),
        ('datetime', 'datetime'),
        ('open_price', 'open_price'),
        ('high_price', 'high_price'),
        ('low_price', 'low_price'),
        ('close_price', 'close_price'),
        ('volume', 'volume')
    ]
    
    print("\nNULL値チェック:")
    has_nulls = False
    for name, col in null_checks:
        cursor.execute(f"SELECT COUNT(*) FROM chart_data WHERE {col} IS NULL")
        null_count = cursor.fetchone()[0]
        if null_count > 0:
            print(f"  ⚠️  {name}: {null_count} 件のNULL")
            has_nulls = True
    
    if not has_nulls:
        print("  ✅ NULL値なし")
    
    # 価格の整合性チェック
    cursor.execute("""
        SELECT COUNT(*) FROM chart_data 
        WHERE high_price < low_price 
           OR high_price < open_price 
           OR high_price < close_price
           OR low_price > open_price
           OR low_price > close_price
    """)
    invalid_prices = cursor.fetchone()[0]
    
    if invalid_prices > 0:
        print(f"  ⚠️  価格の不整合: {invalid_prices} 件")
    else:
        print("  ✅ 価格データの整合性: 正常")
    
    # 重複チェック
    cursor.execute("""
        SELECT symbol, datetime, timeframe, COUNT(*) as cnt
        FROM chart_data
        GROUP BY symbol, datetime, timeframe
        HAVING cnt > 1
    """)
    duplicates = cursor.fetchall()
    
    if duplicates:
        print(f"  ⚠️  重複レコード: {len(duplicates)} 件")
        for dup in duplicates[:5]:
            print(f"     {dup[0]} {dup[1]} {dup[2]}: {dup[3]} 件")
    else:
        print("  ✅ 重複レコード: なし")
    
    conn.close()

def analyze_fundamental_data(db_path='fundamental_data.db'):
    """ファンダメンタルデータの分析"""
    if not os.path.exists(db_path):
        print(f"\n❌ {db_path} が見つかりません")
        return
    
    print("\n" + "=" * 80)
    print("📈 fundamental_data 分析")
    print("=" * 80)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # テーブル一覧
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\nテーブル: {', '.join(tables)}")
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count:,} レコード")
        
        if count > 0 and table == 'fundamental_data':
            cursor.execute(f"SELECT * FROM {table} LIMIT 3")
            print(f"\n  サンプルデータ:")
            for row in cursor.fetchall():
                print(f"    {row}")
    
    conn.close()

def check_csv_data(csv_base_path='csv_data'):
    """CSVデータの存在確認"""
    print("\n" + "=" * 80)
    print("📁 CSV データチェック")
    print("=" * 80)
    
    if not os.path.exists(csv_base_path):
        print(f"\n❌ {csv_base_path} ディレクトリが存在しません")
        return
    
    csv_files = []
    for root, dirs, files in os.walk(csv_base_path):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                size = os.path.getsize(full_path)
                csv_files.append((full_path, size))
    
    if not csv_files:
        print("\n❌ CSVファイルが見つかりません")
    else:
        print(f"\n見つかったCSVファイル: {len(csv_files)} 件")
        for path, size in csv_files[:10]:
            print(f"  {path}: {size:,} bytes")
        if len(csv_files) > 10:
            print(f"  ... 他 {len(csv_files) - 10} ファイル")

def check_config():
    """設定ファイルの確認"""
    print("\n" + "=" * 80)
    print("⚙️  設定ファイル確認")
    print("=" * 80)
    
    config_file = 'config.json'
    if not os.path.exists(config_file):
        print(f"\n❌ {config_file} が見つかりません")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"\nシステム設定:")
    for key, value in config.get('system_settings', {}).items():
        print(f"  {key}: {value}")
    
    print(f"\n収集設定:")
    for key, value in config.get('collection_settings', {}).items():
        print(f"  {key}: {value}")

def check_prime_symbols():
    """監視対象銘柄の確認"""
    print("\n" + "=" * 80)
    print("📋 監視対象銘柄")
    print("=" * 80)
    
    symbols_file = 'prime_symbols.csv'
    if not os.path.exists(symbols_file):
        print(f"\n❌ {symbols_file} が見つかりません")
        return
    
    with open(symbols_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # ヘッダーを除く
    data_lines = [l for l in lines if l.strip() and not l.startswith('symbol,')]
    
    print(f"\n監視対象銘柄数: {len(data_lines)}")
    
    # デイトレード適合銘柄
    suitable_count = sum(1 for line in data_lines if 'true' in line.lower())
    print(f"デイトレード適合: {suitable_count} 銘柄")
    
    # 実際に収集されている銘柄との比較
    if os.path.exists('trading_data.db'):
        conn = sqlite3.connect('trading_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM chart_data")
        collected_symbols = set(row[0] for row in cursor.fetchall())
        conn.close()
        
        print(f"実際に収集済み: {len(collected_symbols)} 銘柄")
        print(f"収集率: {len(collected_symbols) / len(data_lines) * 100:.1f}%")
        
        if collected_symbols:
            print(f"\n収集済み銘柄: {', '.join(sorted(collected_symbols))}")

def main():
    """メイン処理"""
    os.chdir('/workspaces/kabu/RakutenTradingSystem')
    
    print("=" * 80)
    print("🔍 データ収集状況チェックツール")
    print("=" * 80)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"作業ディレクトリ: {os.getcwd()}")
    
    # 設定確認
    check_config()
    
    # 監視対象銘柄
    check_prime_symbols()
    
    # データベース状態
    print("\n" + "=" * 80)
    print("💾 データベースファイル一覧")
    print("=" * 80)
    
    db_files = [
        'trading_data.db',
        'fundamental_data.db',
        'daily_trading_data.db',
        'core/trading_data.db',
        'core/fundamental_data.db'
    ]
    
    for db_file in db_files:
        if os.path.exists(db_file):
            status = check_database_status(db_file)
            if status:
                print(f"\n{db_file}:")
                print(f"  サイズ: {status['size_mb']} MB")
                print(f"  テーブル: {', '.join(status['tables'])}")
                for table, info in status['table_info'].items():
                    print(f"    {table}: {info['count']:,} レコード")
    
    # 詳細分析
    analyze_chart_data('trading_data.db')
    analyze_fundamental_data('fundamental_data.db')
    check_csv_data('csv_data')
    
    # サマリー
    print("\n" + "=" * 80)
    print("📝 チェック完了")
    print("=" * 80)
    print("\n詳細レポートは data_collection_check_report.md を参照してください")

if __name__ == '__main__':
    main()
