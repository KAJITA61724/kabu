#!/bin/bash
# Codespace起動時に自動実行されるスクリプト

cd /workspaces/kabu/RakutenTradingSystem/data_collection

# データ収集プロセスが動いていなければ起動
if ! pgrep -f "run_continuous_collection" > /dev/null; then
    nohup bash run_continuous_collection.sh > ../logs/continuous.log 2>&1 &
    echo "データ収集を自動起動しました (PID: $!)"
fi

