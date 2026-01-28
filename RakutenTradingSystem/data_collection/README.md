# データ収集システム

このフォルダには、yfinanceを使った自動データ収集システムが含まれています。

## 稼働中のファイル（✅ 使用中）

- **automated_data_collection.py**: メインのデータ収集スクリプト
- **run_continuous_collection.sh**: 連続実行用シェルスクリプト
- **autostart.sh**: Codespace再起動時の自動起動スクリプト

## データ保存先

- **../data/market_data.db**: SQLiteデータベース（5分足データ）
- **../logs/continuous.log**: 収集ログ

## 実行方法

### バックグラウンド実行（推奨）
```bash
cd /workspaces/kabu/RakutenTradingSystem/data_collection
nohup bash run_continuous_collection.sh > ../logs/continuous.log 2>&1 &
```

### 手動実行（テスト用）
```bash
cd /workspaces/kabu/RakutenTradingSystem/data_collection
python3 automated_data_collection.py
```

### 自動起動
```bash
bash autostart.sh
```

## 動作確認

```bash
# プロセス確認
ps aux | grep "run_continuous_collection"

# ログ確認
tail -f ../logs/continuous.log

# データ確認
python3 -c "
import sqlite3
import pandas as pd
conn = sqlite3.connect('../data/market_data.db')
print('総件数:', pd.read_sql('SELECT COUNT(*) FROM chart_data_5min', conn).iloc[0][0])
print('最新:', pd.read_sql('SELECT MAX(datetime) FROM chart_data_5min', conn).iloc[0][0])
conn.close()
"
```
