#!/bin/bash
# 連続データ収集スクリプト（平日の取引時間中のみ）

cd "$(dirname "$0")"

LOG_FILE="../logs/continuous.log"
mkdir -p ../logs

echo "連続データ収集を開始します..." | tee -a "$LOG_FILE"

while true; do
    CURRENT_TIME=$(TZ='Asia/Tokyo' date '+%Y-%m-%d %H:%M:%S')
    DAY_OF_WEEK=$(TZ='Asia/Tokyo' date '+%u')  # 1=月曜, 7=日曜
    HOUR=$(TZ='Asia/Tokyo' date '+%H')
    
    # 週末チェック
    if [ "$DAY_OF_WEEK" -ge 6 ]; then
        echo "[$CURRENT_TIME] 週末のため待機中..." | tee -a "$LOG_FILE"
        sleep 3600  # 1時間待機
        continue
    fi
    
    # 取引時間チェック（9:00-15:00）
    if [ "$HOUR" -lt 9 ] || [ "$HOUR" -ge 15 ]; then
        echo "[$CURRENT_TIME] 取引時間外のため待機中..." | tee -a "$LOG_FILE"
        sleep 600  # 10分待機
        continue
    fi
    
    # データ収集実行
    echo "" | tee -a "$LOG_FILE"
    echo "[$CURRENT_TIME] データ収集実行..." | tee -a "$LOG_FILE"
    
    START=$(date +%s)
    python3 automated_data_collection.py 2>&1 | tee -a "$LOG_FILE"
    END=$(date +%s)
    
    DURATION=$((END - START))
    echo "実行時間: ${DURATION}秒" | tee -a "$LOG_FILE"
    echo "✅ 収集完了 - 次回は5分後" | tee -a "$LOG_FILE"
    
    # 5分待機
    sleep 300
done
