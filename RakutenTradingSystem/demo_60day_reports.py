#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
60日間の日次レポートデモ生成
実際の運用を想定したサンプルレポート作成
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

def generate_demo_60day_reports():
    """60日間のデモ日次レポートを生成"""
    
    # レポート保存ディレクトリ作成
    daily_reports_dir = "daily_reports_demo"
    if not os.path.exists(daily_reports_dir):
        os.makedirs(daily_reports_dir)
    
    print(f"60日間日次レポートデモ生成開始")
    print(f"保存先: {daily_reports_dir}/")
    print("="*60)
    
    # 基準日から60日前まで
    end_date = datetime(2025, 7, 18)  # 実際に取引があった日として設定
    start_date = end_date - timedelta(days=89)  # 土日を考慮して89日前から
    
    symbols = ['7203', '6758', '8306', '9984', '6861']
    generated_reports = []
    
    current_date = start_date
    report_count = 0
    
    while current_date <= end_date and report_count < 60:
        # 平日のみ処理（土日祝日スキップ）
        if current_date.weekday() < 5:  # 月-金
            report_count += 1
            
            # その日の取引データを生成
            trades = generate_daily_trades(current_date, symbols)
            
            # レポート生成
            daily_report = generate_daily_report_content(trades, current_date, report_count)
            
            # ファイル保存
            filename = f"daily_report_{current_date.strftime('%Y%m%d')}.txt"
            filepath = os.path.join(daily_reports_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(daily_report)
            
            # 統計用データ
            total_return = sum([t['return'] for t in trades]) if trades else 0
            generated_reports.append({
                'date': current_date,
                'filename': filename,
                'trades_count': len(trades),
                'total_return': total_return
            })
            
            if report_count % 10 == 0:
                print(f"進捗: {report_count}/60 レポート生成完了")
        
        current_date += timedelta(days=1)
    
    # サマリーレポート生成
    generate_summary_report(generated_reports, daily_reports_dir)
    
    print(f"\\n生成完了！")
    print(f"- 日次レポート: {len(generated_reports)}個")
    print(f"- 保存先: {daily_reports_dir}/")
    
    return generated_reports

def generate_daily_trades(date, symbols):
    """その日の取引データを生成（リアルな変動パターン）"""
    trades = []
    
    # 日によって取引数を変動（0-5取引）
    trade_probability = 0.7  # 70%の確率で取引発生
    if random.random() > trade_probability:
        return trades  # 取引なしの日
    
    num_trades = random.randint(1, min(len(symbols), 4))
    selected_symbols = random.sample(symbols, num_trades)
    
    for symbol in selected_symbols:
        # 基準価格設定（銘柄ごと）
        base_prices = {
            '7203': 3200,   # トヨタ
            '6758': 24500,  # ソニー
            '8306': 950,    # 三菱UFJ
            '9984': 12200,  # ソフトバンクグループ
            '6861': 1950    # キーエンス
        }
        
        base_price = base_prices.get(symbol, 1000)
        
        # 価格変動（-3%～+5%の範囲）
        price_change = random.uniform(-0.03, 0.05)
        entry_price = base_price * (1 + random.uniform(-0.02, 0.02))  # 基準価格から±2%
        exit_price = entry_price * (1 + price_change)
        
        # 取引手数料を考慮した実際の収益率
        return_rate = (exit_price / entry_price - 1) - 0.001  # 手数料0.1%
        
        # 決済理由の決定
        if return_rate >= 0.03:
            exit_reason = '利確'
        elif return_rate <= -0.02:
            exit_reason = '損切'
        else:
            exit_reason = '時間切れ'
        
        # 取引時間（9:30-15:00の間でランダム）
        entry_hour = random.randint(9, 14)
        entry_minute = random.randint(0, 59) if entry_hour < 14 else random.randint(0, 30)
        
        # 決済時間はエントリーから30分後以降
        entry_total_minutes = entry_hour * 60 + entry_minute
        exit_total_minutes = random.randint(entry_total_minutes + 30, 15 * 60)  # 15:00まで
        
        exit_hour = min(exit_total_minutes // 60, 15)
        exit_minute = exit_total_minutes % 60 if exit_hour < 15 else 0
        
        trade = {
            'symbol': symbol,
            'entry_time': date.replace(hour=entry_hour, minute=entry_minute),
            'exit_time': date.replace(hour=exit_hour, minute=exit_minute),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': return_rate,
            'exit_reason': exit_reason,
            'ensemble_pred': random.uniform(0.005, 0.025),  # 予測値
            'individual_preds': {
                'RandomForest': random.uniform(0.005, 0.030),
                'LinearRegression': random.uniform(0.000, 0.020),
                'LightGBM': random.uniform(0.005, 0.025)
            }
        }
        trades.append(trade)
    
    return trades

def generate_daily_report_content(trades, date, day_number):
    """日次レポートの内容を生成"""
    report = f"{'='*60}\\n"
    report += f"{date.date()} 日次取引レポート (Day {day_number}/60)\\n"
    report += f"{'='*60}\\n"
    
    # 前日取引量フィルター結果（サンプル）
    report += f"前営業日取引量フィルター結果:\\n"
    filter_results = [
        ('8306', random.randint(30000000, 50000000)),
        ('7203', random.randint(20000000, 35000000)),
        ('9984', random.randint(8000000, 15000000)),
        ('6758', random.randint(5000000, 10000000)),
        ('6861', random.randint(300000, 800000))
    ]
    
    for symbol, volume in filter_results:
        report += f"  {symbol}: {volume:,}株\\n"
    report += f"\\n"
    
    if not trades:
        report += "取引なし（市況不良またはシグナルなし）\\n"
        report += f"\\n市況コメント: "
        comments = [
            "前日大幅下落の影響で様子見",
            "重要指標発表前で取引手控え",
            "ボラティリティ低下で機会なし",
            "予測信頼度が閾値を下回り取引見送り"
        ]
        report += random.choice(comments) + "\\n"
        return report
    
    # 基本統計
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['return'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_return = sum([t['return'] for t in trades])
    
    report += f"取引数: {total_trades}\\n"
    report += f"勝率: {win_rate:.1%} ({winning_trades}/{total_trades})\\n"
    report += f"日次収益率: {total_return:.2%}\\n"
    
    if total_trades > 0:
        avg_return = total_return / total_trades
        max_return = max([t['return'] for t in trades])
        min_return = min([t['return'] for t in trades])
        
        report += f"平均収益率: {avg_return:.3%}\\n"
        report += f"最大利益: {max_return:.3%}\\n"
        report += f"最大損失: {min_return:.3%}\\n"
    
    # 取引詳細
    report += f"\\n取引詳細:\\n"
    for i, trade in enumerate(trades, 1):
        entry_time = trade['entry_time'].strftime('%H:%M')
        exit_time = trade['exit_time'].strftime('%H:%M')
        report += f"  [{i}] {trade['symbol']}: {entry_time}-{exit_time} "
        report += f"¥{trade['entry_price']:.0f}→¥{trade['exit_price']:.0f} "
        report += f"({trade['return']:+.2%}) [{trade['exit_reason']}]\\n"
        
        # 予測詳細
        preds = trade['individual_preds']
        report += f"      予測: {trade['ensemble_pred']:.3f} "
        report += f"(RF:{preds['RandomForest']:.3f}, "
        report += f"LR:{preds['LinearRegression']:.3f}, "
        report += f"LGB:{preds['LightGBM']:.3f})\\n"
    
    # 銘柄別統計
    if len(trades) > 1:
        symbol_returns = {}
        for trade in trades:
            symbol = trade['symbol']
            if symbol not in symbol_returns:
                symbol_returns[symbol] = []
            symbol_returns[symbol].append(trade['return'])
        
        report += f"\\n銘柄別成績:\\n"
        for symbol, returns in symbol_returns.items():
            count = len(returns)
            total_ret = sum(returns)
            avg_ret = total_ret / count
            report += f"  {symbol}: {count}回, 収益率{total_ret:+.2%} (平均{avg_ret:+.3%})\\n"
    
    # その日の特記事項
    if total_return > 0.05:
        report += f"\\n📈 優秀日: 日次収益率{total_return:.1%}の好成績\\n"
    elif total_return < -0.03:
        report += f"\\n📉 注意日: 日次収益率{total_return:.1%}の損失\\n"
    
    return report

def generate_summary_report(reports, reports_dir):
    """60日間のサマリーレポート生成"""
    if not reports:
        return
    
    summary_report = f"{'='*60}\\n"
    summary_report += f"60日間日次取引レポート 総合サマリー\\n"
    summary_report += f"{'='*60}\\n"
    summary_report += f"期間: {reports[0]['date'].date()} ～ {reports[-1]['date'].date()}\\n"
    summary_report += f"総営業日数: {len(reports)}日\\n"
    
    # 基本統計
    total_trades = sum([r['trades_count'] for r in reports])
    total_return = sum([r['total_return'] for r in reports])
    trading_days = len([r for r in reports if r['trades_count'] > 0])
    profitable_days = len([r for r in reports if r['total_return'] > 0])
    
    summary_report += f"\\n📊 基本統計:\\n"
    summary_report += f"総取引数: {total_trades}回\\n"
    summary_report += f"取引実行日数: {trading_days}日 ({trading_days/len(reports):.1%})\\n"
    summary_report += f"利益日数: {profitable_days}日 ({profitable_days/len(reports):.1%})\\n"
    summary_report += f"総収益率: {total_return:.2%}\\n"
    
    if len(reports) > 0:
        avg_daily_return = total_return / len(reports)
        summary_report += f"日次平均収益率: {avg_daily_return:.3%}\\n"
        
        # 月間複利計算（概算）
        monthly_return = (1 + avg_daily_return) ** 20 - 1  # 月20営業日
        summary_report += f"月間期待収益率: {monthly_return:.2%}\\n"
    
    # パフォーマンス分析
    daily_returns = [r['total_return'] for r in reports]
    
    summary_report += f"\\n📈 パフォーマンス分析:\\n"
    summary_report += f"最高日次収益: {max(daily_returns):.2%}\\n"
    summary_report += f"最低日次収益: {min(daily_returns):.2%}\\n"
    summary_report += f"収益標準偏差: {np.std(daily_returns):.3%}\\n"
    
    # シャープレシオ（簡易）
    risk_free_rate = 0.001
    excess_return = avg_daily_return - risk_free_rate
    sharpe_ratio = excess_return / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
    summary_report += f"シャープレシオ: {sharpe_ratio:.3f}\\n"
    
    # 最高・最低の日
    best_day = max(reports, key=lambda x: x['total_return'])
    worst_day = min(reports, key=lambda x: x['total_return'])
    
    summary_report += f"\\n🏆 ベスト・ワースト:\\n"
    summary_report += f"最高収益日: {best_day['date'].date()} ({best_day['total_return']:+.2%}, {best_day['trades_count']}取引)\\n"
    summary_report += f"最低収益日: {worst_day['date'].date()} ({worst_day['total_return']:+.2%}, {worst_day['trades_count']}取引)\\n"
    
    # 週別集計
    weekly_stats = {}
    for report in reports:
        week_start = report['date'] - timedelta(days=report['date'].weekday())
        week_key = week_start.strftime('%Y-%m-%d')
        
        if week_key not in weekly_stats:
            weekly_stats[week_key] = {'trades': 0, 'return': 0, 'days': 0}
        
        weekly_stats[week_key]['trades'] += report['trades_count']
        weekly_stats[week_key]['return'] += report['total_return']
        weekly_stats[week_key]['days'] += 1
    
    summary_report += f"\\n📅 週別成績 (上位5週):\\n"
    sorted_weeks = sorted(weekly_stats.items(), key=lambda x: x[1]['return'], reverse=True)
    for i, (week, stats) in enumerate(sorted_weeks[:5]):
        summary_report += f"  Week {week}: {stats['days']}日, {stats['trades']}取引, 収益率{stats['return']:+.2%}\\n"
    
    # 月別集計
    monthly_stats = {}
    for report in reports:
        month_key = report['date'].strftime('%Y-%m')
        
        if month_key not in monthly_stats:
            monthly_stats[month_key] = {'trades': 0, 'return': 0, 'days': 0}
        
        monthly_stats[month_key]['trades'] += report['trades_count']
        monthly_stats[month_key]['return'] += report['total_return']
        monthly_stats[month_key]['days'] += 1
    
    summary_report += f"\\n📆 月別成績:\\n"
    for month, stats in sorted(monthly_stats.items()):
        summary_report += f"  {month}: {stats['days']}日, {stats['trades']}取引, 収益率{stats['return']:+.2%}\\n"
    
    # 連勝・連敗分析
    streaks = analyze_streaks([r['total_return'] for r in reports])
    summary_report += f"\\n🔥 連勝・連敗記録:\\n"
    summary_report += f"最長連勝: {streaks['max_win_streak']}日\\n"
    summary_report += f"最長連敗: {streaks['max_loss_streak']}日\\n"
    
    # ファイル保存
    summary_filename = f"60day_summary_report_{datetime.now().strftime('%Y%m%d')}.txt"
    summary_filepath = os.path.join(reports_dir, summary_filename)
    
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        f.write(summary_report)
    
    print(f"\\n{'='*60}")
    print("60日間サマリーレポート")
    print(f"{'='*60}")
    print(summary_report)
    print(f"\\nサマリー保存: {summary_filename}")

def analyze_streaks(returns):
    """連勝・連敗を分析"""
    current_win_streak = 0
    current_loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    
    for ret in returns:
        if ret > 0:
            current_win_streak += 1
            current_loss_streak = 0
            max_win_streak = max(max_win_streak, current_win_streak)
        elif ret < 0:
            current_loss_streak += 1
            current_win_streak = 0
            max_loss_streak = max(max_loss_streak, current_loss_streak)
        else:
            current_win_streak = 0
            current_loss_streak = 0
    
    return {
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak
    }

if __name__ == "__main__":
    reports = generate_demo_60day_reports()
    print(f"\\n🎉 60日間の日次レポート生成完了！")
    print(f"📁 保存先: daily_reports_demo/")
    print(f"📄 レポート数: {len(reports)}個")
