# プロジェクト構造 - AWS移行対応版

## 📂 整理済みフォルダ構造

```
RakutenTradingSystem/
├── aws_deployment/              # ★ AWS移行用（重要）
│   ├── README.md                # AWS移行の全手順
│   ├── MIGRATION_CHECKLIST.md  # 移行チェックリスト
│   ├── setup_aws.sh             # 自動セットアップスクリプト
│   ├── check_status.py          # ステータス確認ツール
│   └── requirements.txt         # Python依存関係
│
├── data_collection/             # ★ 現在稼働中のシステム
│   ├── README.md                # 使い方
│   ├── automated_data_collection.py  # メインスクリプト
│   ├── run_continuous_collection.sh  # 連続実行
│   └── autostart.sh             # 自動起動
│
├── archive/                     # 古いファイル（参考用）
│   └── old_data_collection/     # 旧データ収集システム
│
├── data/                        # データベース保存先
│   └── market_data.db           # SQLiteデータベース
│
├── logs/                        # ログファイル
│   └── continuous.log           # 収集ログ
│
└── prime_symbols.csv            # 対象銘柄リスト（216銘柄）
```

## 🎯 AWS移行に必要なファイル（最小構成）

### 必須ファイル
1. **data_collection/** フォルダ全体
2. **prime_symbols.csv**
3. **aws_deployment/requirements.txt**

### 転送不要（AWS上で自動作成）
- data/ - 空フォルダ作成
- logs/ - 空フォルダ作成

### オプション（既存データがある場合）
- data/market_data.db

## 📋 AWS移行の流れ

### 1. 準備（5分）
- [aws_deployment/README.md](aws_deployment/README.md) を読む
- [aws_deployment/MIGRATION_CHECKLIST.md](aws_deployment/MIGRATION_CHECKLIST.md) を印刷

### 2. AWS環境構築（15分）
- Lightsailインスタンス作成
- SSH接続確認

### 3. 自動セットアップ（10分）
```bash
# AWSで1コマンド実行
curl -o setup.sh https://raw.githubusercontent.com/KAJITA61724/kabu/main/RakutenTradingSystem/aws_deployment/setup_aws.sh
chmod +x setup.sh
./setup.sh
```

### 4. サービス起動（1分）
```bash
sudo systemctl start kabu-data-collection
```

### 5. 動作確認（5分）
```bash
python3 ~/kabu_trading/kabu/RakutenTradingSystem/aws_deployment/check_status.py
```

**合計所要時間: 約40分**

## 💡 Codespace vs AWS 比較

| 項目 | Codespace（現在） | AWS（移行後） |
|------|------------------|--------------|
| **稼働時間** | 4時間制限 | 24時間連続 |
| **手動操作** | 朝・昼2回 | 完全自動 |
| **データ損失** | リスク中 | リスク低 |
| **月額料金** | 無料 | $5程度 |
| **安定性** | 中 | 高 |
| **メンテナンス** | 毎日必要 | 週1回確認 |

## 🔄 現在の状態

### ✅ 完了
- データ収集システム整理
- AWS移行ファイル作成
- GitHub Actions設定（データ引き継ぎなし版）
- Codespaceで稼働中

### ⏳ 次のステップ
1. AWS環境作成（ユーザー判断）
2. [aws_deployment/MIGRATION_CHECKLIST.md](aws_deployment/MIGRATION_CHECKLIST.md) に従って移行
3. 1週間テスト
4. 完全移行

## 📞 サポート

- AWS移行: [aws_deployment/README.md](aws_deployment/README.md)
- データ収集: [data_collection/README.md](data_collection/README.md)
- GitHub Actions: [GITHUB_ACTIONS_SETUP.md](../GITHUB_ACTIONS_SETUP.md)

---

**推奨**: まずはCodespaceで数日運用して、データ収集が安定したらAWSに移行
