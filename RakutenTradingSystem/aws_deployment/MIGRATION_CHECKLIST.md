# AWS移行チェックリスト

## 📋 移行前の準備

### 1. AWS環境の準備
- [ ] AWSアカウント作成済み
- [ ] クレジットカード登録済み
- [ ] SSH鍵ペア生成済み（`ssh-keygen`で作成）

### 2. インスタンス作成
- [ ] Lightsail または EC2 インスタンス作成
- [ ] OS: Ubuntu 22.04 LTS選択
- [ ] ストレージ: 20GB以上確保
- [ ] SSH接続確認完了

### 3. ローカルの準備
- [ ] 現在のデータベースをバックアップ
  ```bash
  cd /workspaces/kabu/RakutenTradingSystem
  cp data/market_data.db data/market_data_backup_$(date +%Y%m%d).db
  ```

## 🚀 移行作業

### ステップ1: AWS接続確認
```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<AWS_PUBLIC_IP>
```
- [ ] SSH接続成功

### ステップ2: 自動セットアップ実行
```bash
# AWSインスタンスで実行
curl -o setup_aws.sh https://raw.githubusercontent.com/KAJITA61724/kabu/main/RakutenTradingSystem/aws_deployment/setup_aws.sh
chmod +x setup_aws.sh
./setup_aws.sh
```
- [ ] セットアップスクリプト実行完了
- [ ] エラーなく完了

### ステップ3: データベース転送（既存データがある場合）
```bash
# ローカルマシンから実行
cd /workspaces/kabu/RakutenTradingSystem
scp -i ~/.ssh/your-key.pem data/market_data.db ubuntu@<AWS_IP>:~/kabu_trading/kabu/RakutenTradingSystem/data/
```
- [ ] データベースファイル転送完了
- [ ] ファイルサイズ確認済み

### ステップ4: サービス起動
```bash
# AWSで実行
sudo systemctl start kabu-data-collection
sudo systemctl status kabu-data-collection
```
- [ ] サービス起動成功（緑の●active (running)）
- [ ] エラーログなし

### ステップ5: 動作確認
```bash
# ログ確認（数分待ってから）
tail -f ~/kabu_trading/kabu/RakutenTradingSystem/logs/continuous.log
```
- [ ] データ収集ログが表示される
- [ ] "✅ 収集完了"メッセージ確認
- [ ] データベース件数増加確認

```bash
# データ確認
python3 -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('~/kabu_trading/kabu/RakutenTradingSystem/data/market_data.db')
print('総件数:', pd.read_sql('SELECT COUNT(*) FROM chart_data_5min', conn).iloc[0][0])
print('最新:', pd.read_sql('SELECT MAX(datetime) FROM chart_data_5min', conn).iloc[0][0])
conn.close()
"
```
- [ ] データ件数確認
- [ ] 最新時刻が当日

## 🔍 1週間テスト期間

### 毎日確認すること
- [ ] Day 1: サービス稼働中か確認
- [ ] Day 2: データが増えているか確認
- [ ] Day 3: ログにエラーがないか確認
- [ ] Day 4: ディスク容量チェック
- [ ] Day 5: データベース整合性確認
- [ ] Day 6: 取引時間外の動作確認
- [ ] Day 7: 週末の動作確認（停止しているか）

### 確認コマンド
```bash
# サービス状態
sudo systemctl status kabu-data-collection

# データ統計
python3 ~/kabu_trading/kabu/RakutenTradingSystem/aws_deployment/check_status.py

# ディスク容量
df -h

# プロセス確認
ps aux | grep python
```

## ✅ 完全移行判断

以下すべてクリアで移行完了：
- [ ] 1週間連続で正常動作
- [ ] データ欠損なし
- [ ] エラーログなし
- [ ] ディスク容量問題なし
- [ ] Codespaceより安定

## 🔄 移行完了後

### Codespace側の処理
- [ ] Codespaceのデータ収集プロセスを停止
  ```bash
  pkill -f run_continuous_collection
  ```
- [ ] GitHub Actionsを無効化（Settings → Actions → Disable）
- [ ] Codespaceは開発用のみ使用

### AWS側の設定
- [ ] 自動バックアップ設定（週1回スナップショット）
- [ ] CloudWatch監視設定（オプション）
- [ ] 料金アラート設定（月$10で設定）

## 📊 移行前後の比較

| 項目 | Codespace | AWS |
|------|-----------|-----|
| 稼働時間 | 4時間制限 | 24時間 |
| 手動操作 | 1日2回必要 | 不要 |
| データ損失リスク | 中 | 低 |
| 月額料金 | 無料 | $5程度 |
| 信頼性 | 中 | 高 |

## ⚠️ トラブル時の対応

### サービスが停止した場合
```bash
sudo systemctl restart kabu-data-collection
```

### ログが出ない場合
```bash
# 手動実行でテスト
cd ~/kabu_trading/kabu/RakutenTradingSystem/data_collection
python3 automated_data_collection.py
```

### 完全リセット
```bash
sudo systemctl stop kabu-data-collection
rm -rf ~/kabu_trading
# setup_aws.shを再実行
```

---

**完了したらチェック**: 全項目✅になったら移行成功です！
