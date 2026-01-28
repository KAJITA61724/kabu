#!/usr/bin/env python3
"""
AWS環境のステータス確認スクリプト
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

DB_PATH = os.path.expanduser('~/kabu_trading/kabu/RakutenTradingSystem/data/market_data.db')

def check_status():
    print("=" * 60)
    print("データ収集システム - ステータスレポート")
    print(f"確認日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if not os.path.exists(DB_PATH):
        print("❌ エラー: データベースが見つかりません")
        print(f"   パス: {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    
    # 総レコード数
    total = pd.read_sql('SELECT COUNT(*) as c FROM chart_data_5min', conn).iloc[0]['c']
    print(f"\n📊 総レコード数: {total:,}件")
    
    # 銘柄数
    symbols = pd.read_sql('SELECT COUNT(DISTINCT symbol) as c FROM chart_data_5min', conn).iloc[0]['c']
    print(f"📈 銘柄数: {symbols}銘柄")
    
    # 最新データ
    latest = pd.read_sql('SELECT MAX(datetime) as dt FROM chart_data_5min', conn).iloc[0]['dt']
    print(f"⏰ 最新データ: {latest}")
    
    # 今日のデータ
    today = datetime.now().strftime('%Y-%m-%d')
    today_count = pd.read_sql(
        f"SELECT COUNT(*) as c FROM chart_data_5min WHERE datetime LIKE '{today}%'", 
        conn
    ).iloc[0]['c']
    print(f"📅 今日のデータ: {today_count:,}件")
    
    # 直近1時間のデータ
    recent = pd.read_sql('''
        SELECT datetime, COUNT(*) as count 
        FROM chart_data_5min 
        WHERE datetime >= datetime('now', '-1 hour')
        GROUP BY datetime
        ORDER BY datetime DESC
        LIMIT 5
    ''', conn)
    
    print(f"\n🕐 直近1時間の収集状況:")
    if len(recent) > 0:
        for _, row in recent.iterrows():
            print(f"  {row['datetime']}: {row['count']}銘柄")
    else:
        print("  データなし（取引時間外の可能性）")
    
    # データ期間
    date_range = pd.read_sql('''
        SELECT 
            MIN(datetime) as first,
            MAX(datetime) as last,
            JULIANDAY(MAX(datetime)) - JULIANDAY(MIN(datetime)) as days
        FROM chart_data_5min
    ''', conn).iloc[0]
    
    print(f"\n📆 データ期間:")
    print(f"  開始: {date_range['first']}")
    print(f"  終了: {date_range['last']}")
    print(f"  期間: {date_range['days']:.1f}日")
    
    # データベースサイズ
    db_size = os.path.getsize(DB_PATH) / (1024*1024)
    print(f"\n💾 データベースサイズ: {db_size:.2f} MB")
    
    # 健全性チェック
    print(f"\n✅ 健全性チェック:")
    
    # 最新データが古すぎないか
    if latest:
        latest_dt = datetime.strptime(latest, '%Y-%m-%d %H:%M:%S')
        hours_old = (datetime.now() - latest_dt).total_seconds() / 3600
        
        if hours_old < 2:
            print("  ✅ 最新データは2時間以内")
        elif hours_old < 24:
            print("  ⚠️  最新データが少し古い（取引時間外？）")
        else:
            print("  ❌ 最新データが24時間以上前（要確認）")
    
    # 今日のデータがあるか（平日の場合）
    weekday = datetime.now().weekday()
    if weekday < 5:  # 月〜金
        if today_count > 0:
            print("  ✅ 今日のデータあり")
        else:
            print("  ⚠️  今日のデータなし（要確認）")
    else:
        print("  ℹ️  週末のため今日のデータなしは正常")
    
    conn.close()
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    try:
        check_status()
    except Exception as e:
        print(f"❌ エラー: {e}")
