# データ収集システム - 最新状況レポート

**日時**: 2026-02-05 16:24 JST  
**確認者**: システム管理

---

## 📊 現在のデータ状況

### データベース統計 (market_data.db)

```
総データ件数: 62,936件
収集銘柄数: 208銘柄
データ期間: 2026-01-22 ~ 2026-01-28
最新データ: 2026-01-28 13:25:00
```

### データ件数トップ10銘柄

| 銘柄コード | データ件数 |
|-----------|----------|
| 6136 | 306件 |
| 1801 | 305件 |
| 2269 | 305件 |
| 2502 | 305件 |
| 5232 | 305件 |
| 5301 | 305件 |
| 5333 | 305件 |
| 5707 | 305件 |
| 5802 | 305件 |
| 6448 | 305件 |

---

## ⚠️ 発見された問題

### 1. データ収集が停止している

- ✅ **良い点**: 208銘柄、62,936件のデータが蓄積済み
- ❌ **問題**: 最新データが1月28日で止まっている（約1週間前）
- ❌ **問題**: 本日（2月5日）のデータが存在しない
- ❌ **問題**: 過去7日間のデータが0件

### 2. GitHub Actions が全て失敗

**最新の実行状況**:
```
最新実行: 2026-02-05 07:05 UTC（16:05 JST）
ステータス: failure (失敗)
エラー: sqlite3.OperationalError: unable to open database file
```

**原因**:
- GitHub Actions環境で毎回クリーンな環境から始まる
- データベーステーブルが初期化されていない
- データベースディレクトリが存在しない

---

## 🔧 実施した修正

### 修正1: データベース自動初期化機能の追加

**ファイル**: `data_collection/automated_data_collection.py`

**追加内容**:
```python
def _ensure_database(self):
    """データベースとテーブルを初期化"""
    import os
    os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    conn = sqlite3.connect(self.db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chart_data_5min (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            datetime TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adj_close REAL,
            collected_at TEXT,
            UNIQUE(symbol, datetime)
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol_datetime 
                  ON chart_data_5min(symbol, datetime)')
    conn.commit()
    conn.close()
    logging.info(f"データベース準備完了: {self.db_path}")
```

**機能**:
- コンストラクタで自動的に呼び出される
- データベースファイルの親ディレクトリを自動作成
- テーブルとインデックスを自動作成（存在しない場合のみ）
- GitHub Actions環境でも初回から動作する

### 修正2: ワークフローのエラーハンドリング改善

**ファイル**: `.github/workflows/data-collection.yml`

**変更点**:
```yaml
- name: 統計表示
  run: |
    python3 -c "
    import sqlite3, pandas as pd
    import os
    db_path = 'RakutenTradingSystem/data/market_data.db'
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        total = pd.read_sql('SELECT COUNT(*) as c FROM chart_data_5min', conn).iloc[0]['c']
        latest = pd.read_sql('SELECT MAX(datetime) as dt FROM chart_data_5min', conn).iloc[0]['dt']
        print(f'✅ 総件数: {total:,}件')
        print(f'⏰ 最新: {latest}')
        conn.close()
    else:
        print('⚠️ データベースファイルが見つかりません')
    "
```

**改善内容**:
- ファイル存在チェックを追加
- ファイルが無い場合でもエラーにならない
- 適切なメッセージを表示

---

## ✅ 動作確認

### ローカル環境でのテスト結果

```
✅ データベース初期化: 成功
✅ テーブル作成: 成功  
✅ 既存データ読み込み: 62,936件
✅ インデックス作成: 成功
```

---

## 📋 次のステップ

### 1. 即座に実施すべき対応

- [ ] 修正をGitHubにプッシュ
- [ ] GitHub Actionsのワークフローを手動実行してテスト
- [ ] データ収集が正常に動作することを確認

### 2. 今後のモニタリング

- [ ] 今後数日間のデータ収集状況を監視
- [ ] 5分おきのcron実行が正常に動作しているか確認
- [ ] エラーが発生した場合の通知設定を検討

### 3. システム改善の提案

#### 優先度: 高
- [ ] データ収集の成功/失敗を通知する仕組み（Slack, Email等）
- [ ] データギャップの自動検出と補完
- [ ] ヘルスチェックダッシュボードの作成

#### 優先度: 中
- [ ] 収集データの品質チェック（異常値検出）
- [ ] バックアップとリストア機能
- [ ] 複数のデータソースからのフェイルオーバー

---

## 📝 技術的詳細

### データベーススキーマ

```sql
CREATE TABLE chart_data_5min (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,              -- 銘柄コード
    datetime TEXT NOT NULL,            -- 日時（YYYY-MM-DD HH:MM:SS）
    open REAL,                         -- 始値
    high REAL,                         -- 高値
    low REAL,                          -- 安値
    close REAL,                        -- 終値
    volume INTEGER,                    -- 出来高
    adj_close REAL,                    -- 調整後終値
    collected_at TEXT,                 -- 収集日時
    UNIQUE(symbol, datetime)           -- 重複防止
);

CREATE INDEX idx_symbol_datetime ON chart_data_5min(symbol, datetime);
```

### GitHub Actions スケジュール

```yaml
schedule:
  # 平日9:00-15:00（JST）に5分おきに実行
  # UTC時刻で指定（JST - 9時間）
  - cron: '0-59/5 0-6 * * 1-5'  # 月-金 9:00-15:00 JST
```

**実行タイミング**:
- 月曜日～金曜日
- JST 9:00～15:00（市場時間）
- 5分間隔
- UTC 0:00～6:00に相当

---

## 🎯 期待される効果

### 修正後の期待動作

1. **自動データ収集の再開**
   - 平日の市場時間中に5分おきにデータ取得
   - 208銘柄 × 1日あたり約72回 = 約15,000件/日

2. **安定した運用**
   - GitHub Actions環境での確実な動作
   - エラー時にも適切なログ出力
   - データの重複防止

3. **データの継続性**
   - 1月28日以降のギャップを埋める
   - 今後は継続的にデータ蓄積

---

## 📞 サポート情報

**問題が続く場合**:
1. GitHub Actionsのログを確認
2. ローカル環境で `automated_data_collection.py` を直接実行してテスト
3. prime_symbols.csv の内容を確認

**確認コマンド**:
```bash
# ローカルでのテスト実行
cd RakutenTradingSystem/data_collection
python3 automated_data_collection.py

# データベース内容確認
sqlite3 ../data/market_data.db "SELECT COUNT(*) FROM chart_data_5min;"
```

---

**レポート終了**
