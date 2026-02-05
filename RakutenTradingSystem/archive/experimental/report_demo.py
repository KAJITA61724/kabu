#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日次・月次レポート機能のデモ
thursday_friday_demo.pyの結果を使用
"""
from datetime import datetime, timedelta
import pandas as pd

def generate_demo_trades():
    """デモ用の取引データを生成"""
    # thursday_friday_demo.pyの実際の結果を再現
    demo_trades = [
        {
            'symbol': '7203',
            'entry_time': datetime(2025, 7, 18, 9, 30),
            'exit_time': datetime(2025, 7, 18, 14, 15),
            'entry_price': 3245.0,
            'exit_price': 3312.0,
            'return': 0.0206,
            'exit_reason': '利確',
            'ensemble_pred': 0.018,
            'individual_preds': {'RandomForest': 0.020, 'LinearRegression': 0.015, 'LightGBM': 0.019}
        },
        {
            'symbol': '6758',
            'entry_time': datetime(2025, 7, 18, 10, 0),
            'exit_time': datetime(2025, 7, 18, 13, 45),
            'entry_price': 24890.0,
            'exit_price': 25657.0,
            'return': 0.0308,
            'exit_reason': '利確',
            'ensemble_pred': 0.025,
            'individual_preds': {'RandomForest': 0.028, 'LinearRegression': 0.022, 'LightGBM': 0.026}
        },
        {
            'symbol': '8306',
            'entry_time': datetime(2025, 7, 18, 10, 30),
            'exit_time': datetime(2025, 7, 18, 12, 20),
            'entry_price': 956.5,
            'exit_price': 937.8,
            'return': -0.0195,
            'exit_reason': '損切',
            'ensemble_pred': 0.012,
            'individual_preds': {'RandomForest': 0.015, 'LinearRegression': 0.008, 'LightGBM': 0.013}
        },
        {
            'symbol': '9984',
            'entry_time': datetime(2025, 7, 18, 11, 15),
            'exit_time': datetime(2025, 7, 18, 15, 0),
            'entry_price': 12350.0,
            'exit_price': 12726.0,
            'return': 0.0304,
            'exit_reason': '時間切れ',
            'ensemble_pred': 0.022,
            'individual_preds': {'RandomForest': 0.024, 'LinearRegression': 0.018, 'LightGBM': 0.025}
        },
        # 7月17日の取引
        {
            'symbol': '7203',
            'entry_time': datetime(2025, 7, 17, 9, 45),
            'exit_time': datetime(2025, 7, 17, 11, 30),
            'entry_price': 3198.0,
            'exit_price': 3245.0,
            'return': 0.0147,
            'exit_reason': '利確',
            'ensemble_pred': 0.016,
            'individual_preds': {'RandomForest': 0.018, 'LinearRegression': 0.012, 'LightGBM': 0.017}
        },
        {
            'symbol': '6758',
            'entry_time': datetime(2025, 7, 17, 10, 15),
            'exit_time': datetime(2025, 7, 17, 14, 45),
            'entry_price': 24456.0,
            'exit_price': 24890.0,
            'return': 0.0177,
            'exit_reason': '利確',
            'ensemble_pred': 0.019,
            'individual_preds': {'RandomForest': 0.021, 'LinearRegression': 0.016, 'LightGBM': 0.020}
        },
        # 7月16日の取引  
        {
            'symbol': '8306',
            'entry_time': datetime(2025, 7, 16, 10, 0),
            'exit_time': datetime(2025, 7, 16, 12, 0),
            'entry_price': 945.2,
            'exit_price': 956.5,
            'return': 0.0120,
            'exit_reason': '利確',
            'ensemble_pred': 0.014,
            'individual_preds': {'RandomForest': 0.016, 'LinearRegression': 0.011, 'LightGBM': 0.015}
        },
        {
            'symbol': '9984',
            'entry_time': datetime(2025, 7, 16, 11, 30),
            'exit_time': datetime(2025, 7, 16, 13, 15),
            'entry_price': 12145.0,
            'exit_price': 12350.0,
            'return': 0.0169,
            'exit_reason': '利確',
            'ensemble_pred': 0.018,
            'individual_preds': {'RandomForest': 0.020, 'LinearRegression': 0.015, 'LightGBM': 0.019}
        }
    ]
    
    return demo_trades

def generate_daily_report(trades, target_date):
    """1日単位のレポート生成"""
    if not trades:
        return f"\\n{target_date.date()} - 取引なし"
    
    df = pd.DataFrame(trades)
    
    # 指定日の取引のみフィルタ
    df['entry_date'] = pd.to_datetime(df['entry_time']).dt.date
    daily_trades = df[df['entry_date'] == target_date.date()]
    
    if daily_trades.empty:
        return f"\\n{target_date.date()} - 取引なし"
    
    report = f"\\n{'='*50}\\n"
    report += f"{target_date.date()} 日次取引レポート\\n"
    report += f"{'='*50}\\n"
    
    # 基本統計
    total_trades = len(daily_trades)
    winning_trades = len(daily_trades[daily_trades['return'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_return = daily_trades['return'].sum()
    
    report += f"取引数: {total_trades}\\n"
    report += f"勝率: {win_rate:.1%} ({winning_trades}/{total_trades})\\n"
    report += f"日次収益率: {total_return:.2%}\\n"
    
    if total_trades > 0:
        report += f"平均収益率: {daily_trades['return'].mean():.3%}\\n"
        report += f"最大利益: {daily_trades['return'].max():.3%}\\n"
        report += f"最大損失: {daily_trades['return'].min():.3%}\\n"
    
    # 取引詳細
    report += f"\\n取引詳細:\\n"
    for _, trade in daily_trades.iterrows():
        entry_time = pd.to_datetime(trade['entry_time']).strftime('%H:%M')
        exit_time = pd.to_datetime(trade['exit_time']).strftime('%H:%M')
        report += f"  {trade['symbol']}: {entry_time}-{exit_time} "
        report += f"¥{trade['entry_price']:.0f}→¥{trade['exit_price']:.0f} "
        report += f"({trade['return']:+.2%}) [{trade['exit_reason']}]\\n"
        
        # 予測詳細も追加
        if 'ensemble_pred' in trade:
            report += f"    予測: {trade['ensemble_pred']:.3f} "
            if 'individual_preds' in trade and trade['individual_preds']:
                preds = trade['individual_preds']
                report += f"(RF:{preds.get('RandomForest', 0):.3f}, "
                report += f"LR:{preds.get('LinearRegression', 0):.3f}, "
                report += f"LGB:{preds.get('LightGBM', 0):.3f})\\n"
            else:
                report += "\\n"
    
    # 銘柄別統計
    if len(daily_trades) > 1:
        symbol_stats = daily_trades.groupby('symbol').agg({
            'return': ['count', 'sum', 'mean']
        }).round(3)
        
        report += f"\\n銘柄別成績:\\n"
        for symbol in symbol_stats.index:
            count = symbol_stats.loc[symbol, ('return', 'count')]
            total_ret = symbol_stats.loc[symbol, ('return', 'sum')]
            avg_ret = symbol_stats.loc[symbol, ('return', 'mean')]
            report += f"  {symbol}: {count}回, 収益率{total_ret:+.2%} (平均{avg_ret:+.3%})\\n"
    
    return report

def generate_monthly_report(trades, start_date, end_date):
    """1か月単位のレポート生成"""
    if not trades:
        return f"\\n{start_date.date()} ～ {end_date.date()} - 取引なし"
    
    df = pd.DataFrame(trades)
    df['entry_date'] = pd.to_datetime(df['entry_time']).dt.date
    
    # 期間内の取引をフィルタ
    period_trades = df[
        (df['entry_date'] >= start_date.date()) & 
        (df['entry_date'] <= end_date.date())
    ]
    
    if period_trades.empty:
        return f"\\n{start_date.date()} ～ {end_date.date()} - 取引なし"
    
    report = f"\\n{'='*60}\\n"
    report += f"{start_date.date()} ～ {end_date.date()} 月次取引レポート\\n"
    report += f"{'='*60}\\n"
    
    # 基本統計
    total_trades = len(period_trades)
    winning_trades = len(period_trades[period_trades['return'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_return = period_trades['return'].sum()
    
    report += f"総取引数: {total_trades}\\n"
    report += f"勝率: {win_rate:.1%} ({winning_trades}/{total_trades})\\n"
    report += f"総収益率: {total_return:.2%}\\n"
    
    if total_trades > 0:
        report += f"平均取引収益率: {period_trades['return'].mean():.3%}\\n"
        report += f"最大利益: {period_trades['return'].max():.3%}\\n"
        report += f"最大損失: {period_trades['return'].min():.3%}\\n"
        report += f"収益標準偏差: {period_trades['return'].std():.3%}\\n"
        
        # シャープレシオ計算
        risk_free_rate = 0.001
        excess_return = period_trades['return'].mean() - risk_free_rate
        sharpe_ratio = excess_return / period_trades['return'].std() if period_trades['return'].std() > 0 else 0
        report += f"シャープレシオ: {sharpe_ratio:.3f}\\n"
    
    # 日次統計
    daily_stats = period_trades.groupby('entry_date').agg({
        'return': ['count', 'sum', 'mean']
    }).round(3)
    
    report += f"\\n日次成績サマリー:\\n"
    trading_days = len(daily_stats)
    profitable_days = len(daily_stats[daily_stats[('return', 'sum')] > 0])
    report += f"取引日数: {trading_days}日\\n"
    report += f"利益日数: {profitable_days}日 ({profitable_days/trading_days:.1%})\\n"
    report += f"日次平均収益率: {daily_stats[('return', 'sum')].mean():.3%}\\n"
    report += f"最高日次収益率: {daily_stats[('return', 'sum')].max():.3%}\\n"
    report += f"最低日次収益率: {daily_stats[('return', 'sum')].min():.3%}\\n"
    
    # 銘柄別統計
    symbol_stats = period_trades.groupby('symbol').agg({
        'return': ['count', 'sum', 'mean'],
        'exit_reason': lambda x: (x == '利確').sum()
    }).round(3)
    
    symbol_stats.columns = ['取引数', '総収益率', '平均収益率', '利確回数']
    symbol_stats['勝率'] = symbol_stats['利確回数'] / symbol_stats['取引数']
    symbol_stats = symbol_stats.sort_values('総収益率', ascending=False)
    
    report += f"\\n銘柄別成績 (上位10銘柄):\\n"
    for symbol, stats in symbol_stats.head(10).iterrows():
        report += f"  {symbol}: {stats['取引数']}回, 収益率{stats['総収益率']:+.2%} "
        report += f"(平均{stats['平均収益率']:+.3%}, 勝率{stats['勝率']:.1%})\\n"
    
    # 時間別統計
    period_trades['hour'] = pd.to_datetime(period_trades['entry_time']).dt.hour
    hourly_stats = period_trades.groupby('hour').agg({
        'return': ['count', 'mean', 'sum']
    }).round(3)
    
    report += f"\\n時間別成績:\\n"
    for hour, stats in hourly_stats.iterrows():
        count = stats[('return', 'count')]
        avg_ret = stats[('return', 'mean')]
        total_ret = stats[('return', 'sum')]
        report += f"  {hour:2d}時台: {count}回, 平均{avg_ret:+.3%}, 合計{total_ret:+.2%}\\n"
    
    # 曜日別統計
    period_trades['weekday'] = pd.to_datetime(period_trades['entry_time']).dt.day_name()
    weekday_stats = period_trades.groupby('weekday').agg({
        'return': ['count', 'mean', 'sum']
    }).round(3)
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    report += f"\\n曜日別成績:\\n"
    for weekday in weekday_order:
        if weekday in weekday_stats.index:
            stats = weekday_stats.loc[weekday]
            count = stats[('return', 'count')]
            avg_ret = stats[('return', 'mean')]
            total_ret = stats[('return', 'sum')]
            report += f"  {weekday[:3]}: {count}回, 平均{avg_ret:+.3%}, 合計{total_ret:+.2%}\\n"
    
    return report

def save_reports_to_file(trades, report_date):
    """レポートをファイルに保存"""
    # 1日レポート
    daily_report = generate_daily_report(trades, report_date)
    daily_filename = f"daily_report_{report_date.strftime('%Y%m%d')}.txt"
    
    with open(daily_filename, 'w', encoding='utf-8') as f:
        f.write(daily_report)
    
    print(f"日次レポート保存: {daily_filename}")
    
    # 月次レポート（過去30日間）
    month_start = report_date - timedelta(days=30)
    monthly_report = generate_monthly_report(trades, month_start, report_date)
    monthly_filename = f"monthly_report_{report_date.strftime('%Y%m%d')}.txt"
    
    with open(monthly_filename, 'w', encoding='utf-8') as f:
        f.write(monthly_report)
    
    print(f"月次レポート保存: {monthly_filename}")
    
    return daily_filename, monthly_filename

def main():
    """メイン実行関数"""
    print("日次・月次レポート機能デモ")
    print("="*50)
    
    # デモ取引データ生成
    trades = generate_demo_trades()
    
    # 各日のレポート生成
    target_dates = [
        datetime(2025, 7, 16),
        datetime(2025, 7, 17), 
        datetime(2025, 7, 18)
    ]
    
    print("\\n=== 日次レポート ===")
    for target_date in target_dates:
        daily_report = generate_daily_report(trades, target_date)
        print(daily_report)
    
    # 月次レポート生成
    print("\\n=== 月次レポート ===")
    start_date = datetime(2025, 7, 1)
    end_date = datetime(2025, 7, 31)
    monthly_report = generate_monthly_report(trades, start_date, end_date)
    print(monthly_report)
    
    # ファイル保存
    print("\\n=== ファイル保存 ===")
    daily_filename, monthly_filename = save_reports_to_file(trades, datetime(2025, 7, 18))
    
    print(f"\\n完了！生成されたファイル:")
    print(f"- {daily_filename}")
    print(f"- {monthly_filename}")

if __name__ == "__main__":
    main()
