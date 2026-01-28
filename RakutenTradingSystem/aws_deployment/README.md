# AWS移行ガイド

このフォルダには、AWS EC2/Lightsailへの移行に必要な情報とファイルがすべて含まれています。

## 📦 必要なファイル一覧

### 1. データ収集システム（必須）
```
../data_collection/
├── automated_data_collection.py  # メインスクリプト
├── run_continuous_collection.sh  # 連続実行スクリプト
└── autostart.sh                  # 自動起動スクリプト
```

### 2. 設定ファイル（必須）
```
../prime_symbols.csv              # 銘柄リスト
requirements.txt                  # Python依存関係（このフォルダ内）
```

### 3. データベース（初回は不要）
```
../data/market_data.db           # 既存データ（オプション）
```

## 🚀 AWS移行手順

### ステップ1: AWSインスタンス作成

#### 推奨スペック
- **サービス**: EC2 t3.micro または Lightsail 最小プラン
- **OS**: Ubuntu 22.04 LTS
- **ストレージ**: 20GB（データ増加を考慮）
- **料金**: 月$5-10程度

#### セキュリティ設定
- SSH(22番ポート)のみ許可
- HTTPSは不要（データ収集のみ）

### ステップ2: 初期セットアップ

```bash
# SSHでAWSに接続後

# システム更新
sudo apt update && sudo apt upgrade -y

# Python3とpipのインストール
sudo apt install -y python3 python3-pip git

# 作業ディレクトリ作成
mkdir -p ~/kabu_trading
cd ~/kabu_trading
```

### ステップ3: ファイル転送

#### 方法A: Gitから直接クローン（推奨）
```bash
cd ~/kabu_trading
git clone https://github.com/KAJITA61724/kabu.git
cd kabu/RakutenTradingSystem
```

#### 方法B: 必要ファイルのみSCPで転送
```bash
# ローカルマシンから実行
scp -r data_collection/ ubuntu@<AWS_IP>:~/kabu_trading/
scp prime_symbols.csv ubuntu@<AWS_IP>:~/kabu_trading/
scp aws_deployment/requirements.txt ubuntu@<AWS_IP>:~/kabu_trading/
```

### ステップ4: Python環境構築

```bash
cd ~/kabu_trading

# 依存関係インストール
pip3 install -r requirements.txt

# または個別インストール
pip3 install yfinance pandas jpholiday
```

### ステップ5: ディレクトリ構造作成

```bash
cd ~/kabu_trading
mkdir -p data logs

# 既存DBがある場合は転送
# scp data/market_data.db ubuntu@<AWS_IP>:~/kabu_trading/data/
```

### ステップ6: 自動起動設定（systemd）

```bash
# サービスファイル作成
sudo nano /etc/systemd/system/kabu-data-collection.service
```

以下の内容を貼り付け：
```ini
[Unit]
Description=Kabu Data Collection Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/kabu_trading/data_collection
ExecStart=/usr/bin/bash /home/ubuntu/kabu_trading/data_collection/run_continuous_collection.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

保存して有効化：
```bash
sudo systemctl daemon-reload
sudo systemctl enable kabu-data-collection
sudo systemctl start kabu-data-collection
```

### ステップ7: 動作確認

```bash
# サービス状態確認
sudo systemctl status kabu-data-collection

# ログ確認
tail -f ~/kabu_trading/logs/continuous.log

# データベース確認
python3 -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('/home/ubuntu/kabu_trading/data/market_data.db')
print('件数:', pd.read_sql('SELECT COUNT(*) FROM chart_data_5min', conn).iloc[0][0])
conn.close()
"
```

### ステップ8: 監視設定（オプション）

```bash
# crontabで定期的にステータスメール送信
crontab -e

# 毎日16時に統計をメール送信（要設定）
0 16 * * 1-5 python3 ~/kabu_trading/data_collection/send_daily_report.py
```

## 🔧 トラブルシューティング

### サービスが起動しない
```bash
# ログ確認
sudo journalctl -u kabu-data-collection -n 50

# 手動実行でエラー確認
cd ~/kabu_trading/data_collection
bash run_continuous_collection.sh
```

### タイムゾーンが合わない
```bash
# 日本時間に設定
sudo timedatectl set-timezone Asia/Tokyo
date  # 確認
```

### ディスク容量不足
```bash
# 容量確認
df -h

# 古いデータ削除（90日以上前）は自動実行されています
```

## 💰 コスト見積もり

### AWS Lightsail（最も簡単）
- $3.5/月プラン: メモリ512MB、ストレージ20GB
- $5/月プラン: メモリ1GB、ストレージ40GB（推奨）

### AWS EC2 t3.micro
- オンデマンド: 約$7.5/月
- 1年リザーブド: 約$4.5/月

### データ転送
- 受信: 無料
- 送信: 月100GBまで無料（十分）

## 📊 移行後のメリット

✅ **24時間365日稼働**（Codespaceの制限なし）  
✅ **自動起動**（systemd管理）  
✅ **データ損失リスク低減**  
✅ **メンテナンス容易**  
✅ **料金固定**（月$5程度）

## 🎯 次のステップ

1. AWSアカウント作成（未作成の場合）
2. Lightsailインスタンス作成
3. SSH鍵設定
4. 上記手順を順番に実行
5. 1週間動作確認
6. Codespaceから完全移行

---

**質問がある場合**: このREADMEを参照しながら進めてください。
