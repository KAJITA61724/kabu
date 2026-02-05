#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
60日間の日次取引レポート一括生成システム
毎日の取引結果を個別にレポート化
"""
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import warnings
import os
import jpholiday

warnings.filterwarnings('ignore')

class DailyReportGenerator:
    def __init__(self, db_path='trading_data.db'):
        self.db_path = db_path
        self.volume_threshold = 300000  # 30万株
        self.max_data_days = 60  # yfinanceの上限
        self.risk_free_rate = 0.001  # リスクフリーレート
        
        # 取引パラメータ
        self.stop_loss = 0.02  # 2%
        self.take_profit = 0.03  # 3%
        self.transaction_cost = 0.001  # 0.1%
        
        print(f"日次レポート生成システム初期化:")
        print(f"- 最大データ期間: {self.max_data_days}日")
        print(f"- 取引量閾値: {self.volume_threshold:,}株")
        
        # レポート保存ディレクトリ作成
        self.daily_reports_dir = "daily_reports"
        if not os.path.exists(self.daily_reports_dir):
            os.makedirs(self.daily_reports_dir)
            print(f"- レポートディレクトリ作成: {self.daily_reports_dir}")
    
    def get_previous_trading_day(self, date):
        """前営業日を取得（日本の祝日を考慮）"""
        prev_day = date - timedelta(days=1)
        
        # 土日と祝日をスキップ
        while prev_day.weekday() >= 5 or jpholiday.is_holiday(prev_day):
            prev_day -= timedelta(days=1)
            
        return prev_day
    
    def collect_max_period_data(self, symbols):
        """最大期間のデータを取得"""
        conn = sqlite3.connect(self.db_path)
        
        # テーブル作成
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chart_data (
                symbol TEXT,
                datetime TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, datetime)
            )
        ''')
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.max_data_days)
        
        print(f"\\n最大期間データ収集: {start_date.date()} ～ {end_date.date()}")
        
        success_count = 0
        for i, symbol in enumerate(symbols, 1):
            try:
                ticker = yf.Ticker(f'{symbol}.T')
                data = ticker.history(start=start_date, end=end_date, interval='5m')
                
                if not data.empty:
                    # データベースに保存
                    for timestamp, row in data.iterrows():
                        conn.execute('''
                            INSERT OR REPLACE INTO chart_data 
                            (symbol, datetime, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (symbol, timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                              row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
                    
                    success_count += 1
                    print(f"[{i:2d}/{len(symbols)}] {symbol}: {len(data):4d}件")
                else:
                    print(f"[{i:2d}/{len(symbols)}] {symbol}: データなし")
                    
            except Exception as e:
                print(f"[{i:2d}/{len(symbols)}] {symbol}: エラー - {e}")
        
        conn.commit()
        conn.close()
        
        print(f"\\nデータ収集完了: {success_count}/{len(symbols)} 銘柄")
        return success_count
    
    def get_trading_days(self):
        """データベースから実際の取引日を取得"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT DISTINCT DATE(datetime) as trading_date 
            FROM chart_data 
            ORDER BY trading_date
        '''
        
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        trading_days = []
        for date_str in result['trading_date']:
            trading_days.append(datetime.strptime(date_str, '%Y-%m-%d'))
        
        return trading_days
    
    def filter_high_volume_symbols(self, target_date, symbols):
        """指定日の前営業日の取引量でフィルタリング"""
        prev_date = self.get_previous_trading_day(target_date)
        
        conn = sqlite3.connect(self.db_path)
        
        # 前営業日の取引量を集計
        query = '''
            SELECT symbol, SUM(volume) as total_volume
            FROM chart_data 
            WHERE DATE(datetime) = ? AND symbol IN ({})
            GROUP BY symbol
            HAVING total_volume >= ?
            ORDER BY total_volume DESC
        '''.format(','.join(['?' for _ in symbols]))
        
        params = [prev_date.strftime('%Y-%m-%d')] + symbols + [self.volume_threshold]
        
        result = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        filtered_symbols = result['symbol'].tolist()
        
        return filtered_symbols, result
    
    def create_features(self, data):
        """テクニカル指標の作成"""
        df = data.copy()
        
        # カラム名を小文字に統一
        df.columns = df.columns.str.lower()
        
        # 移動平均
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # ボリンジャーバンド
        bb_window = 20
        df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        bb_std = df['close'].rolling(window=bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # 価格変化率
        for period in [1, 3, 5]:
            df[f'price_change_{period}'] = df['close'].pct_change(period)
        
        # 出来高変化率
        df['volume_change'] = df['volume'].pct_change()
        
        # ボラティリティ
        df['volatility'] = df['close'].rolling(window=10).std()
        
        return df
    
    def simulate_daily_trading(self, target_date, symbols):
        """1日の取引をシミュレーション（簡易版）"""
        # 前日取引量でフィルタリング
        filtered_symbols, volume_data = self.filter_high_volume_symbols(target_date, symbols)
        
        if not filtered_symbols:
            return [], volume_data
        
        trades = []
        
        # 簡易的な取引シミュレーション
        for symbol in filtered_symbols[:5]:  # 上位5銘柄
            conn = sqlite3.connect(self.db_path)
            
            # その日のデータを取得
            query = '''
                SELECT * FROM chart_data 
                WHERE symbol = ? AND DATE(datetime) = ?
                ORDER BY datetime
            '''
            
            daily_data = pd.read_sql_query(query, conn, 
                                         params=[symbol, target_date.strftime('%Y-%m-%d')])
            conn.close()
            
            if len(daily_data) < 10:  # 最低限のデータ数
                continue
            
            # 簡易的な取引判定（寄り付きから30分後にエントリー、終値で決済）
            if len(daily_data) >= 6:  # 30分分のデータ（5分足×6）
                entry_price = daily_data.iloc[6]['close']  # 30分後の価格
                exit_price = daily_data.iloc[-1]['close']   # 終値
                
                return_rate = (exit_price / entry_price - 1) - self.transaction_cost
                
                # 簡易的な決済理由判定
                if return_rate >= self.take_profit:
                    exit_reason = '利確'
                elif return_rate <= -self.stop_loss:
                    exit_reason = '損切'
                else:
                    exit_reason = '時間切れ'
                
                trade = {
                    'symbol': symbol,
                    'entry_time': target_date.replace(hour=10, minute=0),  # 10:00エントリー
                    'exit_time': target_date.replace(hour=15, minute=0),   # 15:00決済
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': return_rate,
                    'exit_reason': exit_reason,
                    'ensemble_pred': np.random.normal(0.01, 0.005)  # 仮の予測値
                }
                trades.append(trade)
        
        return trades, volume_data
    
    def generate_daily_report(self, trades, target_date, volume_data):
        """1日単位のレポート生成"""
        report = f"{'='*60}\\n"
        report += f"{target_date.date()} 日次取引レポート\\n"
        report += f"{'='*60}\\n"
        
        # 前日取引量情報
        if not volume_data.empty:
            prev_date = self.get_previous_trading_day(target_date)
            report += f"前営業日({prev_date.date()})取引量フィルター結果:\\n"
            for _, row in volume_data.head(5).iterrows():
                report += f"  {row['symbol']}: {row['total_volume']:,}株\\n"
            report += f"\\n"
        
        if not trades:
            report += "取引なし（取引量条件を満たす銘柄なし）\\n"
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
        for trade in trades:
            entry_time = trade['entry_time'].strftime('%H:%M')
            exit_time = trade['exit_time'].strftime('%H:%M')
            report += f"  {trade['symbol']}: {entry_time}-{exit_time} "
            report += f"¥{trade['entry_price']:.0f}→¥{trade['exit_price']:.0f} "
            report += f"({trade['return']:+.2%}) [{trade['exit_reason']}]\\n"
            report += f"    予測: {trade['ensemble_pred']:.3f}\\n"
        
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
        
        return report
    
    def generate_all_daily_reports(self, symbols):
        """全期間の日次レポートを生成"""
        print(f"\\n{'='*60}")
        print("60日間日次レポート一括生成開始")
        print(f"{'='*60}")
        
        # データ収集
        success_count = self.collect_max_period_data(symbols)
        if success_count == 0:
            print("データが取得できませんでした。")
            return
        
        # 取引日を取得
        trading_days = self.get_trading_days()
        print(f"\\n取引日数: {len(trading_days)}日")
        
        if not trading_days:
            print("取引日データがありません。")
            return
        
        # 各日のレポート生成
        generated_reports = []
        
        for i, trading_day in enumerate(trading_days[1:], 1):  # 最初の日は前日データなしのためスキップ
            print(f"\\n[{i:2d}/{len(trading_days)-1}] {trading_day.date()} 処理中...")
            
            # その日の取引をシミュレーション
            trades, volume_data = self.simulate_daily_trading(trading_day, symbols)
            
            # レポート生成
            daily_report = self.generate_daily_report(trades, trading_day, volume_data)
            
            # ファイル保存
            filename = f"daily_report_{trading_day.strftime('%Y%m%d')}.txt"
            filepath = os.path.join(self.daily_reports_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(daily_report)
            
            generated_reports.append({
                'date': trading_day,
                'filename': filename,
                'trades_count': len(trades),
                'total_return': sum([t['return'] for t in trades]) if trades else 0
            })
            
            print(f"    取引数: {len(trades)}, レポート保存: {filename}")
        
        # サマリー生成
        self.generate_summary_report(generated_reports)
        
        return generated_reports
    
    def generate_summary_report(self, reports):
        """全期間のサマリーレポート生成"""
        if not reports:
            return
        
        summary_report = f"{'='*60}\\n"
        summary_report += f"60日間日次レポート サマリー\\n"
        summary_report += f"{'='*60}\\n"
        summary_report += f"期間: {reports[0]['date'].date()} ～ {reports[-1]['date'].date()}\\n"
        summary_report += f"総日数: {len(reports)}日\\n"
        
        total_trades = sum([r['trades_count'] for r in reports])
        total_return = sum([r['total_return'] for r in reports])
        trading_days = len([r for r in reports if r['trades_count'] > 0])
        profitable_days = len([r for r in reports if r['total_return'] > 0])
        
        summary_report += f"総取引数: {total_trades}\\n"
        summary_report += f"取引実行日数: {trading_days}日\\n"
        summary_report += f"利益日数: {profitable_days}日 ({profitable_days/len(reports):.1%})\\n"
        summary_report += f"総収益率: {total_return:.2%}\\n"
        
        if len(reports) > 0:
            avg_daily_return = total_return / len(reports)
            summary_report += f"日次平均収益率: {avg_daily_return:.3%}\\n"
        
        # 最高・最低収益日
        best_day = max(reports, key=lambda x: x['total_return'])
        worst_day = min(reports, key=lambda x: x['total_return'])
        
        summary_report += f"\\n最高収益日: {best_day['date'].date()} ({best_day['total_return']:+.2%})\\n"
        summary_report += f"最低収益日: {worst_day['date'].date()} ({worst_day['total_return']:+.2%})\\n"
        
        # 週別統計
        weekly_stats = {}
        for report in reports:
            week_start = report['date'] - timedelta(days=report['date'].weekday())
            week_key = week_start.strftime('%Y-%m-%d')
            
            if week_key not in weekly_stats:
                weekly_stats[week_key] = {'trades': 0, 'return': 0}
            
            weekly_stats[week_key]['trades'] += report['trades_count']
            weekly_stats[week_key]['return'] += report['total_return']
        
        summary_report += f"\\n週別成績 (上位5週):\\n"
        sorted_weeks = sorted(weekly_stats.items(), key=lambda x: x[1]['return'], reverse=True)
        for i, (week, stats) in enumerate(sorted_weeks[:5]):
            summary_report += f"  {week}週: 取引{stats['trades']}回, 収益率{stats['return']:+.2%}\\n"
        
        # ファイル保存
        summary_filename = f"daily_reports_summary_{datetime.now().strftime('%Y%m%d')}.txt"
        summary_filepath = os.path.join(self.daily_reports_dir, summary_filename)
        
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"\\n{'='*60}")
        print("サマリーレポート")
        print(f"{'='*60}")
        print(summary_report)
        print(f"\\nサマリー保存: {summary_filename}")

def main():
    """メイン実行関数"""
    # 主要銘柄リスト
    symbols = ['7203', '6758', '8306', '9984', '6861']
    
    # システム初期化
    generator = DailyReportGenerator()
    
    # 全期間の日次レポート生成
    reports = generator.generate_all_daily_reports(symbols)
    
    if reports:
        print(f"\\n{'='*60}")
        print(f"日次レポート生成完了!")
        print(f"生成されたレポート数: {len(reports)}個")
        print(f"保存先: {generator.daily_reports_dir}/")
        print(f"{'='*60}")
    else:
        print("レポート生成に失敗しました。")

if __name__ == "__main__":
    main()
