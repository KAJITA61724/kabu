#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最大期間対応の包括的取引システム
毎日の取引サイクル: 前日取引量フィルター → モデル作成 → 当日取引
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

class ComprehensiveTradingSystem:
    def __init__(self, db_path='trading_data.db'):
        self.db_path = db_path
        self.volume_threshold = 300000  # 30万株
        self.max_data_days = 60  # yfinanceの上限
        self.risk_free_rate = 0.001  # リスクフリーレート
        
        # 取引パラメータ
        self.stop_loss = 0.02  # 2%
        self.take_profit = 0.03  # 3%
        self.transaction_cost = 0.001  # 0.1%
        
        print(f"包括的取引システム初期化:")
        print(f"- 最大データ期間: {self.max_data_days}日")
        print(f"- 取引量閾値: {self.volume_threshold:,}株")
        print(f"- ストップロス: {self.stop_loss*100}%")
        print(f"- 利確: {self.take_profit*100}%")
    
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
        
        print(f"\\n{target_date.date()}の取引対象フィルタリング:")
        print(f"基準日: {prev_date.date()}")
        print(f"対象銘柄: {len(filtered_symbols)}/{len(symbols)} 銘柄")
        
        for _, row in result.head(10).iterrows():
            print(f"  {row['symbol']}: {row['total_volume']:,}株")
        
        return filtered_symbols
    
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
    
    def prepare_ml_data(self, symbol, target_date):
        """機械学習用データの準備"""
        conn = sqlite3.connect(self.db_path)
        
        # 指定日より前のデータを取得
        query = '''
            SELECT * FROM chart_data 
            WHERE symbol = ? AND datetime < ?
            ORDER BY datetime
        '''
        
        data = pd.read_sql_query(query, conn, 
                               params=[symbol, target_date.strftime('%Y-%m-%d')])
        conn.close()
        
        if len(data) < 100:  # 最低限のデータ数
            return None, None
        
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
        
        # 特徴量作成
        data = self.create_features(data)
        
        # 目標変数（1時間後の価格変化率）
        data['target'] = data['close'].shift(-12).pct_change()  # 12期間後（1時間後）
        
        # 特徴量列を選択
        feature_cols = [col for col in data.columns 
                       if col not in ['symbol', 'target', 'open', 'high', 'low', 'close']]
        
        # NaNを除去
        ml_data = data[feature_cols + ['target']].dropna()
        
        if len(ml_data) < 50:
            return None, None
        
        X = ml_data[feature_cols]
        y = ml_data['target']
        
        return X, y
    
    def train_models(self, X, y):
        """複数のモデルを訓練"""
        models = {}
        
        # RandomForest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        models['RandomForest'] = rf
        
        # LinearRegression
        lr = LinearRegression()
        lr.fit(X, y)
        models['LinearRegression'] = lr
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        lgb_model.fit(X, y)
        models['LightGBM'] = lgb_model
        
        return models
    
    def get_current_features(self, symbol, current_time):
        """現在時刻の特徴量を取得"""
        conn = sqlite3.connect(self.db_path)
        
        # 現在時刻より前の最新データを取得
        query = '''
            SELECT * FROM chart_data 
            WHERE symbol = ? AND datetime <= ?
            ORDER BY datetime DESC
            LIMIT 100
        '''
        
        data = pd.read_sql_query(query, conn, 
                               params=[symbol, current_time.strftime('%Y-%m-%d %H:%M:%S')])
        conn.close()
        
        if len(data) < 50:
            return None
        
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime').sort_index()
        
        # 特徴量作成
        data = self.create_features(data)
        
        # 最新の特徴量を取得
        feature_cols = [col for col in data.columns 
                       if col not in ['symbol', 'open', 'high', 'low', 'close']]
        
        latest_features = data[feature_cols].iloc[-1]
        
        return latest_features.dropna()
    
    def ensemble_predict(self, models, features):
        """アンサンブル予測"""
        predictions = {}
        
        for name, model in models.items():
            try:
                pred = model.predict([features])[0]
                predictions[name] = pred
            except:
                predictions[name] = 0
        
        # 単純平均
        ensemble_pred = np.mean(list(predictions.values()))
        
        return ensemble_pred, predictions
    
    def simulate_trading_day(self, target_date, symbols):
        """1日の取引をシミュレーション"""
        print(f"\\n{'='*60}")
        print(f"{target_date.date()} 取引シミュレーション")
        print(f"{'='*60}")
        
        # 前日取引量でフィルタリング
        filtered_symbols = self.filter_high_volume_symbols(target_date, symbols)
        
        if not filtered_symbols:
            print("取引対象銘柄なし")
            return []
        
        trades = []
        
        for symbol in filtered_symbols[:10]:  # 上位10銘柄
            print(f"\\n--- {symbol} 分析開始 ---")
            
            # モデル訓練
            X, y = self.prepare_ml_data(symbol, target_date)
            if X is None:
                print(f"{symbol}: データ不足")
                continue
            
            models = self.train_models(X, y)
            print(f"{symbol}: モデル訓練完了 (データ数: {len(X)})")
            
            # 取引時間のシミュレーション
            trading_start = target_date.replace(hour=9, minute=0)
            trading_end = target_date.replace(hour=15, minute=0)
            
            current_time = trading_start
            position = None
            
            while current_time <= trading_end:
                # 現在の特徴量を取得
                features = self.get_current_features(symbol, current_time)
                
                if features is not None and len(features) > 0:
                    # 予測
                    ensemble_pred, individual_preds = self.ensemble_predict(models, features)
                    
                    # 取引判断
                    if position is None and ensemble_pred > 0.005:  # 0.5%以上の上昇予測で買い
                        # 現在価格を取得
                        conn = sqlite3.connect(self.db_path)
                        price_query = '''
                            SELECT close FROM chart_data 
                            WHERE symbol = ? AND datetime <= ?
                            ORDER BY datetime DESC LIMIT 1
                        '''
                        price_result = conn.execute(price_query, 
                                                  [symbol, current_time.strftime('%Y-%m-%d %H:%M:%S')]).fetchone()
                        conn.close()
                        
                        if price_result:
                            entry_price = price_result[0]
                            position = {
                                'symbol': symbol,
                                'entry_time': current_time,
                                'entry_price': entry_price,
                                'ensemble_pred': ensemble_pred,
                                'individual_preds': individual_preds,
                                'stop_loss_price': entry_price * (1 - self.stop_loss),
                                'take_profit_price': entry_price * (1 + self.take_profit)
                            }
                            print(f"{current_time.strftime('%H:%M')} 買い注文: ¥{entry_price:.2f} (予測: {ensemble_pred:.3f})")
                    
                    elif position is not None:
                        # 現在価格をチェック
                        conn = sqlite3.connect(self.db_path)
                        price_result = conn.execute(price_query, 
                                                  [symbol, current_time.strftime('%Y-%m-%d %H:%M:%S')]).fetchone()
                        conn.close()
                        
                        if price_result:
                            current_price = price_result[0]
                            
                            # 利確・損切判定
                            if current_price >= position['take_profit_price']:
                                # 利確
                                position['exit_time'] = current_time
                                position['exit_price'] = current_price
                                position['exit_reason'] = '利確'
                                position['return'] = (current_price / position['entry_price'] - 1) - self.transaction_cost
                                trades.append(position)
                                print(f"{current_time.strftime('%H:%M')} 利確: ¥{current_price:.2f} (収益: {position['return']:.3f})")
                                position = None
                                
                            elif current_price <= position['stop_loss_price']:
                                # 損切
                                position['exit_time'] = current_time
                                position['exit_price'] = current_price
                                position['exit_reason'] = '損切'
                                position['return'] = (current_price / position['entry_price'] - 1) - self.transaction_cost
                                trades.append(position)
                                print(f"{current_time.strftime('%H:%M')} 損切: ¥{current_price:.2f} (損失: {position['return']:.3f})")
                                position = None
                
                # 次の時刻へ（5分刻み）
                current_time += timedelta(minutes=5)
            
            # 取引時間終了時にポジションがあれば強制決済
            if position is not None:
                conn = sqlite3.connect(self.db_path)
                price_result = conn.execute('''
                    SELECT close FROM chart_data 
                    WHERE symbol = ? AND datetime <= ?
                    ORDER BY datetime DESC LIMIT 1
                ''', [symbol, trading_end.strftime('%Y-%m-%d %H:%M:%S')]).fetchone()
                conn.close()
                
                if price_result:
                    exit_price = price_result[0]
                    position['exit_time'] = trading_end
                    position['exit_price'] = exit_price
                    position['exit_reason'] = '時間切れ'
                    position['return'] = (exit_price / position['entry_price'] - 1) - self.transaction_cost
                    trades.append(position)
                    print(f"15:00 強制決済: ¥{exit_price:.2f} (収益: {position['return']:.3f})")
        
        return trades
    
    def run_comprehensive_backtest(self, symbols, days=30):
        """包括的バックテスト実行"""
        print(f"\\n包括的取引システム バックテスト開始")
        print(f"期間: 過去{days}日間")
        print(f"対象銘柄: {len(symbols)}銘柄")
        
        # データ収集
        self.collect_max_period_data(symbols)
        
        # バックテスト期間設定
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        all_trades = []
        current_date = start_date
        
        while current_date <= end_date:
            # 平日のみ実行
            if current_date.weekday() < 5 and not jpholiday.is_holiday(current_date):
                daily_trades = self.simulate_trading_day(current_date, symbols)
                all_trades.extend(daily_trades)
            
            current_date += timedelta(days=1)
        
        # 結果分析
        self.analyze_backtest_results(all_trades)
        
        # 日次・月次レポート生成
        if all_trades:
            print(f"\\n{'='*60}")
            print("レポート生成中...")
            
            # 各取引日のレポート生成
            trade_dates = set()
            for trade in all_trades:
                trade_date = pd.to_datetime(trade['entry_time']).date()
                trade_dates.add(trade_date)
            
            for trade_date in sorted(trade_dates):
                date_obj = datetime.combine(trade_date, datetime.min.time())
                daily_report = self.generate_daily_report(all_trades, date_obj)
                print(daily_report)
            
            # 月次レポート生成（全期間）
            monthly_report = self.generate_monthly_report(all_trades, start_date, end_date)
            print(monthly_report)
            
            # ファイルに保存
            report_date = end_date
            daily_filename, monthly_filename = self.save_reports_to_file(all_trades, report_date)
        
        return all_trades
    
    def analyze_backtest_results(self, trades):
        """バックテスト結果の分析"""
        if not trades:
            print("\\n取引データなし")
            return
        
        df = pd.DataFrame(trades)
        
        print(f"\\n{'='*60}")
        print(f"包括的バックテスト結果分析")
        print(f"{'='*60}")
        
        # 基本統計
        total_trades = len(trades)
        winning_trades = len(df[df['return'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_return = df['return'].sum()
        avg_return = df['return'].mean()
        
        print(f"総取引数: {total_trades}")
        print(f"勝率: {win_rate:.2%} ({winning_trades}/{total_trades})")
        print(f"総収益率: {total_return:.2%}")
        print(f"平均取引収益率: {avg_return:.3%}")
        
        if total_trades > 0:
            print(f"最大利益: {df['return'].max():.3%}")
            print(f"最大損失: {df['return'].min():.3%}")
            print(f"標準偏差: {df['return'].std():.3%}")
        
        # 銘柄別統計
        print(f"\\n銘柄別成績 (上位10銘柄):")
        symbol_stats = df.groupby('symbol').agg({
            'return': ['count', 'mean', 'sum'],
            'exit_reason': lambda x: (x == '利確').sum()
        }).round(3)
        
        symbol_stats.columns = ['取引数', '平均収益率', '総収益率', '利確回数']
        symbol_stats['勝率'] = symbol_stats['利確回数'] / symbol_stats['取引数']
        
        print(symbol_stats.head(10))
        
        # 時間別統計
        df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
        hourly_stats = df.groupby('hour')['return'].agg(['count', 'mean']).round(3)
        hourly_stats.columns = ['取引数', '平均収益率']
        
        print(f"\\n時間別成績:")
        print(hourly_stats)
    
    def generate_daily_report(self, trades, target_date):
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
    
    def generate_monthly_report(self, trades, start_date, end_date):
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
            excess_return = period_trades['return'].mean() - self.risk_free_rate
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
        
        # 週次統計
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
    
    def save_reports_to_file(self, trades, report_date):
        """レポートをファイルに保存"""
        # 1日レポート
        daily_report = self.generate_daily_report(trades, report_date)
        daily_filename = f"daily_report_{report_date.strftime('%Y%m%d')}.txt"
        
        with open(daily_filename, 'w', encoding='utf-8') as f:
            f.write(daily_report)
        
        print(f"日次レポート保存: {daily_filename}")
        
        # 月次レポート（過去30日間）
        month_start = report_date - timedelta(days=30)
        monthly_report = self.generate_monthly_report(trades, month_start, report_date)
        monthly_filename = f"monthly_report_{report_date.strftime('%Y%m%d')}.txt"
        
        with open(monthly_filename, 'w', encoding='utf-8') as f:
            f.write(monthly_report)
        
        print(f"月次レポート保存: {monthly_filename}")
        
        return daily_filename, monthly_filename

def main():
    """メイン実行関数"""
    # 主要銘柄リスト
    symbols = ['7203', '6758', '8306', '9984', '6861']
    
    # システム初期化
    system = ComprehensiveTradingSystem()
    
    # 包括的バックテスト実行
    trades = system.run_comprehensive_backtest(symbols, days=30)
    
    print(f"\\n{'='*60}")
    print(f"包括的取引システム実行完了!")
    print(f"{'='*60}")
    
    if trades:
        print(f"総取引数: {len(trades)}")
        
        # 最新日のデータでデモレポート生成
        latest_date = datetime.now()
        
        print(f"\\n=== デモレポート生成 ===")
        
        # サンプル日次レポート
        sample_daily_report = system.generate_daily_report(trades, latest_date)
        print(sample_daily_report)
        
        # サンプル月次レポート  
        month_start = latest_date - timedelta(days=30)
        sample_monthly_report = system.generate_monthly_report(trades, month_start, latest_date)
        print(sample_monthly_report)
        
        print(f"\\nレポート機能が正常に動作しています！")
    else:
        print("取引データがありません。")

if __name__ == "__main__":
    main()
