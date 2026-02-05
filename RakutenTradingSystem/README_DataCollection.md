# MarketSpeedⅡ分足データ収集システム

参考: https://note.com/rcat999/n/n22952acffbf9

## 概要

MarketSpeedⅡのRSS機能を使用して分足データを自動収集するシステムです。
ExcelのMSGETDATA関数を使用してチャートデータを取得し、CSVファイルまたはSQLiteデータベースに保存します。

## 前提条件

1. **楽天証券MarketSpeedⅡ**がインストールされ、ログイン済みであること
2. **MarketSpeedⅡ RSSアドイン**がインストールされていること
3. **Microsoft Excel**がインストールされていること
4. **Python 3.8以上**がインストールされていること

## 必要なPythonライブラリ

```bash
pip install pandas numpy pywin32 matplotlib seaborn scikit-learn
```

## ファイル構成

```
RakutenTradingSystem/
├── core/
│   └── enhanced_data_collector.py    # メインデータ収集システム
├── run_data_collection.py            # 実行スクリプト
├── config.json                       # 設定ファイル
├── prime_symbols.csv                 # 銘柄リスト
├── csv_data/                         # CSVデータ保存先
│   └── [銘柄コード]/
│       └── [足種]_[日付].csv
├── backups/                          # バックアップ保存先
└── trading_data.db                   # SQLiteデータベース
```

## 使用方法

### 1. 基本的な使用方法

```bash
python run_data_collection.py
```

対話形式で以下を設定します：
- モード選択（CSV / データベース）
- 足種選択（1M / 5M / 15M / 30M / 1H）
- 取得本数（最大3000本）

### 2. プログラムからの使用

```python
from core.enhanced_data_collector import MarketSpeedDataCollector

# CSVモードで実行
collector = MarketSpeedDataCollector(csv_mode=True)
collector.run_collection(timeframe="5M", count=1000)

# データベースモードで実行
collector = MarketSpeedDataCollector(csv_mode=False)
collector.run_collection(timeframe="5M", count=1000)
```

## 機能

### データ収集機能
- **分足データ取得**: 1分足～1時間足まで対応
- **複数銘柄の一括処理**: prime_symbols.csvの銘柄を自動処理
- **エラーハンドリング**: 個別銘柄でエラーが発生しても継続処理

### 保存機能
- **CSVモード**: 銘柄ごと・日付ごとにCSVファイルを作成
- **データベースモード**: SQLiteデータベースに一元管理
- **自動バックアップ**: CSVファイルのZIP圧縮バックアップ

### 管理機能
- **データ新しさチェック**: 最新データからの経過日数を確認
- **詳細ログ**: 処理状況をログファイルに記録
- **進捗表示**: 現在の処理銘柄と進捗を表示

## 設定

### config.json の主要設定項目

```json
{
    "system_settings": {
        "csv_mode": true,              # CSVモード（true/false）
        "max_symbols": 100,            # 最大処理銘柄数
        "excel_visible": false         # Excel表示（true/false）
    },
    "collection_settings": {
        "default_timeframe": "5M",     # デフォルト足種
        "default_count": 1000,         # デフォルト取得本数
        "symbol_delay": 1.0            # 銘柄間の待機時間（秒）
    }
}
```

### prime_symbols.csv の形式

```csv
symbol,name,avg_volume
7203,トヨタ自動車,8500000
9984,ソフトバンクグループ,3200000
6758,ソニーグループ,2100000
```

## 注意事項

1. **取引時間外での実行を推奨**
   - 取引時間中はサーバー負荷が高く、データ取得に時間がかかる場合があります

2. **MarketSpeedⅡのログイン状態を確認**
   - データ収集前に必ずMarketSpeedⅡにログインしてください

3. **Excelの他作業を避ける**
   - データ収集中は他のExcelファイルの操作を避けてください

4. **大量データの処理時間**
   - 100銘柄×1000本の場合、約30分～1時間程度かかります

## トラブルシューティング

### よくあるエラーと対処法

1. **"Excel接続に失敗しました"**
   - Excelが正しくインストールされているか確認
   - 管理者権限で実行してみる

2. **"#NAME?"エラーが表示される**
   - MarketSpeedⅡにログインしているか確認
   - RSSアドインが正しくインストールされているか確認

3. **"データ取得失敗"が多発する**
   - 取引時間外に実行してみる
   - 取得本数を減らしてみる（1000→500など）

4. **処理が途中で止まる**
   - ログファイルでエラー内容を確認
   - MarketSpeedⅡの接続状態を確認

## ログファイルの確認

```bash
# 本日のログファイル
data_collector_20250718.log
```

ログファイルには以下の情報が記録されます：
- 処理開始・終了時刻
- 各銘柄の処理状況
- エラーメッセージ
- 取得データ件数

## バックアップ

CSVモードの場合、処理完了後に自動でバックアップが作成されます：
```
backups/backup_20250718_143022.zip
```

## 参考資料

- 元記事: https://note.com/rcat999/n/n22952acffbf9
- MarketSpeedⅡ RSS機能ヘルプ
- 楽天証券サポートページ
