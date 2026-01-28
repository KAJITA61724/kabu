#!/bin/bash
# AWS環境セットアップスクリプト
# 新しいAWSインスタンスで実行してください

set -e  # エラーで停止

echo "=========================================="
echo "AWS環境セットアップ開始"
echo "=========================================="

# 現在のディレクトリ確認
WORK_DIR="$HOME/kabu_trading"

echo ""
echo "作業ディレクトリ: $WORK_DIR"
echo ""

# システム更新
echo "1. システム更新中..."
sudo apt update && sudo apt upgrade -y

# 必要なパッケージインストール
echo ""
echo "2. 必要なパッケージをインストール中..."
sudo apt install -y python3 python3-pip git curl

# 作業ディレクトリ作成
echo ""
echo "3. 作業ディレクトリ作成..."
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Gitリポジトリクローン
echo ""
echo "4. GitHubリポジトリをクローン中..."
if [ -d "kabu" ]; then
    echo "既存のリポジトリを更新します..."
    cd kabu
    git pull
    cd ..
else
    git clone https://github.com/KAJITA61724/kabu.git
fi

# Python依存関係インストール
echo ""
echo "5. Python依存関係をインストール中..."
pip3 install --upgrade pip
pip3 install yfinance pandas jpholiday

# ディレクトリ構造作成
echo ""
echo "6. ディレクトリ構造を作成中..."
cd "$WORK_DIR/kabu/RakutenTradingSystem"
mkdir -p data logs

# タイムゾーン設定
echo ""
echo "7. タイムゾーンを日本時間に設定..."
sudo timedatectl set-timezone Asia/Tokyo

# systemdサービスファイル作成
echo ""
echo "8. systemdサービスを設定中..."
sudo tee /etc/systemd/system/kabu-data-collection.service > /dev/null <<EOF
[Unit]
Description=Kabu Data Collection Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR/kabu/RakutenTradingSystem/data_collection
ExecStart=/usr/bin/bash $WORK_DIR/kabu/RakutenTradingSystem/data_collection/run_continuous_collection.sh
Restart=always
RestartSec=10
StandardOutput=append:$WORK_DIR/kabu/RakutenTradingSystem/logs/service.log
StandardError=append:$WORK_DIR/kabu/RakutenTradingSystem/logs/service_error.log

[Install]
WantedBy=multi-user.target
EOF

# サービス有効化
sudo systemctl daemon-reload
sudo systemctl enable kabu-data-collection

echo ""
echo "=========================================="
echo "セットアップ完了！"
echo "=========================================="
echo ""
echo "次のコマンドでデータ収集を開始してください:"
echo "  sudo systemctl start kabu-data-collection"
echo ""
echo "ログ確認:"
echo "  tail -f $WORK_DIR/kabu/RakutenTradingSystem/logs/continuous.log"
echo ""
echo "サービス状態確認:"
echo "  sudo systemctl status kabu-data-collection"
echo ""
