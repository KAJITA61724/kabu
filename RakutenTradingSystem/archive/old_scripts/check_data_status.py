#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データの確認と要約スクリプト
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def check_database_status():
    """データベースの状況を確認"""
    try:
        conn = sqlite3.connect('trading_data.db')
        cursor = conn.cursor()
        
        # テーブル一覧を取得
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print("=== データベース内のテーブル ===")
        print(tables)
        
        if ('chart_data',) in tables:
            # テーブル構造を確認
            cursor.execute("PRAGMA table_info(chart_data)")
            columns = cursor.fetchall()
            print("\n=== テーブル構造 ===")
            column_names = []
            for col in columns:
                print(f"{col[1]} ({col[2]})")
                column_names.append(col[1])
            
            # レコード数を確認
            cursor.execute("SELECT COUNT(*) FROM chart_data")
            count = cursor.fetchone()[0]
            print(f"\nchart_dataテーブルのレコード数: {count}")
            
            # データの期間を確認（timestampカラムを使用）
            if 'timestamp' in column_names:
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM chart_data")
                date_range = cursor.fetchone()
                print(f"データの期間: {date_range[0]} から {date_range[1]}")
            
            # 銘柄別データ数を確認
            cursor.execute("SELECT symbol, COUNT(*) FROM chart_data GROUP BY symbol")
            symbol_counts = cursor.fetchall()
            print("\n=== 銘柄別データ数 ===")
            for symbol, count in symbol_counts:
                print(f"{symbol}: {count}件")
            
            # 最新データの確認
            if 'datetime' in column_names:
                cursor.execute("SELECT symbol, datetime, close_price FROM chart_data ORDER BY datetime DESC, symbol LIMIT 10")
                latest_data = cursor.fetchall()
                print("\n=== 最新データ (上位10件) ===")
                for symbol, datetime, close_price in latest_data:
                    print(f"{symbol}: {datetime} - 終値: {close_price}")
            else:
                cursor.execute("SELECT symbol, close_price FROM chart_data ORDER BY id DESC LIMIT 10")
                latest_data = cursor.fetchall()
                print("\n=== 最新データ (上位10件) ===")
                for symbol, close_price in latest_data:
                    print(f"{symbol}: 終値: {close_price}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"データベース確認エラー: {e}")
        return False

def get_data_summary():
    """データの要約統計を取得"""
    try:
        conn = sqlite3.connect('trading_data.db')
        df = pd.read_sql_query("SELECT * FROM chart_data", conn)
        
        print("\n=== データ要約統計 ===")
        print(f"総レコード数: {len(df)}")
        print(f"銘柄数: {df['symbol'].nunique()}")
        
        # datetimeカラムがあるかチェック
        if 'datetime' in df.columns:
            print(f"期間: {df['datetime'].min()} ～ {df['datetime'].max()}")
        
        # 銘柄別統計（正しいカラム名を使用）
        print("\n=== 銘柄別統計 ===")
        summary = df.groupby('symbol').agg({
            'close_price': ['count', 'mean', 'std', 'min', 'max'],
            'volume': ['mean', 'std']
        }).round(2)
        print(summary)
        
        # 最新の日付データを取得
        if 'datetime' in df.columns:
            df['date'] = pd.to_datetime(df['datetime']).dt.date
            latest_dates = df['date'].value_counts().head(10)
            print("\n=== 最新の日付別データ数 ===")
            print(latest_dates)
        
        conn.close()
        return df
        
    except Exception as e:
        print(f"データ要約エラー: {e}")
        return None

if __name__ == "__main__":
    print("=== データ確認と要約 ===")
    
    # データベース状況確認
    if check_database_status():
        # データ要約
        df = get_data_summary()
        
        if df is not None:
            print("\n✅ データ確認完了")
        else:
            print("\n❌ データ要約に失敗しました")
    else:
        print("\n❌ データベース確認に失敗しました")
