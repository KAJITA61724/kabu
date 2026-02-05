#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yfinanceの分足データ取得期間テスト
"""
import yfinance as yf
from datetime import datetime, timedelta

def test_yfinance_data_range():
    """yfinanceの分足データ取得可能期間をテスト"""
    ticker = yf.Ticker('7203.T')
    
    print('yfinanceの分足データ取得期間テスト:')
    print('=' * 50)
    
    test_periods = [7, 30, 60, 90, 120, 180, 365, 730]  # 日数
    
    for days in test_periods:
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
        
        try:
            data = ticker.history(start=start_date, end=end_date, interval='5m')
            
            if not data.empty:
                data_start = data.index[0].strftime('%Y-%m-%d')
                data_end = data.index[-1].strftime('%Y-%m-%d')
                
                print(f'{days:3d}日前から: {len(data):5d}件 ({data_start} ~ {data_end})')
            else:
                print(f'{days:3d}日前から: データなし')
                
        except Exception as e:
            print(f'{days:3d}日前から: エラー - {e}')
    
    # 最適な期間を特定
    print('\n' + '=' * 50)
    print('最大取得可能期間の特定:')
    
    max_days = 60  # yfinanceの5分足は通常60日程度が上限
    start_date = datetime.now() - timedelta(days=max_days)
    
    try:
        data = ticker.history(start=start_date, interval='5m')
        
        if not data.empty:
            actual_days = (data.index[-1] - data.index[0]).days
            print(f'実際の取得期間: {actual_days}日')
            print(f'データ数: {len(data)}件')
            print(f'開始日: {data.index[0].strftime("%Y-%m-%d %H:%M")}')
            print(f'終了日: {data.index[-1].strftime("%Y-%m-%d %H:%M")}')
            
            return max_days, len(data)
        else:
            print('データが取得できませんでした')
            return 0, 0
            
    except Exception as e:
        print(f'エラー: {e}')
        return 0, 0

if __name__ == "__main__":
    max_days, data_count = test_yfinance_data_range()
    print(f'\n推奨設定: {max_days}日間, 約{data_count}件のデータ')
