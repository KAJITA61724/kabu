# データベースからのデータ取得・分析ガイド

## 概要
リアルタイムデータ収集でメモリを使用せずに、データベースに保存されたデータを効率的に取得・分析するためのツール群です。

## 利用可能なツール

### 1. データ取得ツール (`query_data.py`)
データベースから様々な条件でデータを取得できるインタラクティブツールです。

```bash
python query_data.py
```

**機能:**
- 最新データ取得
- 期間指定データ取得
- 複数銘柄データ取得
- 統計情報表示
- データベースサマリー

### 2. データ分析ツール (`analyze_data.py`)
保存されたデータに対してテクニカル分析を実行します。

```bash
python analyze_data.py
```

**機能:**
- テクニカル指標計算（移動平均、RSI、MACD、ボリンジャーバンド）
- チャートパターン検出
- 複数銘柄比較
- 分析レポート生成

### 3. DataReaderクラス (`core/data_reader.py`)
プログラムから直接データを取得するためのクラスです。

```python
from core.data_reader import DataReader

reader = DataReader()

# 最新データ取得
df = reader.get_latest_data('7203', limit=100)

# 期間指定データ取得
df = reader.get_data_by_date_range('7203', '2025-07-18 09:00:00', '2025-07-18 15:00:00')

# 複数銘柄データ取得
data = reader.get_multiple_symbols_data(['7203', '9984'])

# 統計情報取得
stats = reader.get_data_statistics('7203')
```

## 主要メソッド

### DataReader
- `get_latest_data(symbol, timeframe, limit)`: 最新データを指定件数取得
- `get_data_by_date_range(symbol, start_date, end_date, timeframe)`: 期間指定でデータ取得
- `get_multiple_symbols_data(symbols, timeframe, limit)`: 複数銘柄のデータを一括取得
- `get_data_statistics(symbol, timeframe, days)`: 統計情報取得
- `get_available_symbols()`: 利用可能な銘柄リスト取得
- `get_data_summary()`: データベース全体のサマリー取得

### DataAnalyzer
- `calculate_technical_indicators(symbol, period)`: テクニカル指標計算
- `detect_patterns(symbol, period)`: チャートパターン検出
- `compare_symbols(symbols, period)`: 複数銘柄比較
- `generate_report(symbol)`: 分析レポート生成

## 使用例

### 1. 特定銘柄の最新データ取得
```python
from core.data_reader import DataReader

reader = DataReader()
df = reader.get_latest_data('7203', limit=50)
print(df.head())
```

### 2. 期間指定でのデータ分析
```python
from core.data_reader import DataReader

reader = DataReader()
df = reader.get_data_by_date_range('7203', '2025-07-18 09:00:00', '2025-07-18 15:00:00')

# 価格変動分析
price_change = df['close_price'].pct_change()
volatility = price_change.std()
print(f"ボラティリティ: {volatility:.4f}")
```

### 3. テクニカル分析
```python
from analyze_data import DataAnalyzer

analyzer = DataAnalyzer()
df = analyzer.calculate_technical_indicators('7203')

# 最新のテクニカル指標
latest = df.iloc[-1]
print(f"RSI: {latest['rsi']:.2f}")
print(f"MACD: {latest['macd']:.2f}")
```

### 4. 複数銘柄比較
```python
from analyze_data import DataAnalyzer

analyzer = DataAnalyzer()
comparison = analyzer.compare_symbols(['7203', '9984', '8306'])
print(comparison)
```

## メモリ効率の利点

1. **リアルタイム処理不要**: データベースから必要な分だけ取得
2. **メモリ使用量削減**: 大量データを一度にメモリに読み込まない
3. **高速クエリ**: SQLiteによる効率的なデータアクセス
4. **スケーラビリティ**: データ量が増加してもパフォーマンス維持

## データベース構造

### chart_data テーブル
```sql
CREATE TABLE chart_data (
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
);
```

## 注意事項

1. データベースファイル（`trading_data.db`）が存在する必要があります
2. 大量データを取得する際は、`limit`パラメータを適切に設定してください
3. 期間指定クエリでは、日時フォーマットを正確に指定してください（`YYYY-MM-DD HH:MM:SS`）
4. テクニカル分析には最低限のデータ件数が必要です（RSIは14件、移動平均は指定期間分）

## パフォーマンスTips

1. **インデックス活用**: symbol, datetime, timeframe でのクエリが最適化されています
2. **適切なlimit設定**: 必要以上に大量のデータを取得しない
3. **期間指定の活用**: 不要な過去データの取得を避ける
4. **バッチ処理**: 複数銘柄の処理は `get_multiple_symbols_data` を使用
