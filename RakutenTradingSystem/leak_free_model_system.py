#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yfinance実データを使用したリークなし日次モデル更新システム
- 前日17:00までのデータでモデル学習（リーク防止）
- 当日9:00からの5分足データで予測・取引
- 全て5分足単位で統一
"""
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os
import jpholiday
import pickle

warnings.filterwarnings('ignore')

class LeakFreeModelSystem:
    def __init__(self, db_path='leak_free_trading.db'):
        self.db_path = db_path
        self.volume_threshold = 300000  # 30万株
        self.max_data_days = 60  # yfinanceの上限
        
        # 取引パラメータ
        self.stop_loss = 0.02  # 2%
        self.take_profit = 0.03  # 3%
        self.transaction_cost = 0.001  # 0.1%
        
        # ディレクトリ作成
        self.models_dir = "leak_free_models"
        self.reports_dir = "leak_free_reports"
        
        for dir_name in [self.models_dir, self.reports_dir]:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        
        print(f"リークなし日次モデル更新システム初期化:")
        print(f"- 最大データ期間: {self.max_data_days}日")
        print(f"- モデル保存: {self.models_dir}/")
        print(f"- レポート保存: {self.reports_dir}/")
        print(f"- リーク防止: 前日17:00までのデータのみ使用")
    
    def get_previous_trading_day(self, date):
        """前営業日を取得"""
        prev_day = date - timedelta(days=1)
        while prev_day.weekday() >= 5 or jpholiday.is_holiday(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day
    
    def collect_yfinance_data(self, symbols, start_date, end_date):
        """yfinanceから5分足データを収集"""
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
                adj_close REAL,
                PRIMARY KEY (symbol, datetime)
            )
        ''')
        
        print(f"\\nyfinanceデータ収集: {start_date.date()} ～ {end_date.date()}")
        
        success_count = 0
        for i, symbol in enumerate(symbols, 1):
            try:
                ticker = yf.Ticker(f'{symbol}.T')
                
                # 5分足データ取得
                data = ticker.history(start=start_date, end=end_date, interval='5m')
                
                if not data.empty:
                    # 取引時間のみフィルタ（9:00-15:00）
                    data = data.between_time('09:00', '15:00')
                    
                    print(f"[{i:2d}/{len(symbols)}] {symbol}: {len(data):4d}件 ({data.index[0].date()} ～ {data.index[-1].date()})")
                    
                    # データベースに保存
                    for timestamp, row in data.iterrows():
                        conn.execute('''
                            INSERT OR REPLACE INTO chart_data 
                            (symbol, datetime, open, high, low, close, volume, adj_close)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (symbol, timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                              row['Open'], row['High'], row['Low'], row['Close'], 
                              row['Volume'], row['Close']))
                    
                    success_count += 1
                else:
                    print(f"[{i:2d}/{len(symbols)}] {symbol}: データなし")
                    
            except Exception as e:
                print(f"[{i:2d}/{len(symbols)}] {symbol}: エラー - {str(e)[:100]}")
        
        conn.commit()
        conn.close()
        
        print(f"\\nデータ収集完了: {success_count}/{len(symbols)} 銘柄")
        return success_count
    
    def create_enhanced_features(self, data):
        """5分足に最適化された特徴量作成"""
        df = data.copy()
        df.columns = df.columns.str.lower()
        
        # 基本価格情報
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']  # レンジ比率
        df['oc_ratio'] = (df['close'] - df['open']) / df['open']  # 始値からの変化率
        
        # 短期移動平均（5分足に適合）
        for window in [6, 12, 24, 48]:  # 30分、1時間、2時間、4時間
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}'] - 1
            df[f'ma_slope_{window}'] = (df[f'ma_{window}'] - df[f'ma_{window}'].shift(window//2)) / df[f'ma_{window}'].shift(window//2)
        
        # 価格モメンタム（短期・中期・長期）
        for period in [1, 3, 6, 12, 24]:  # 5分、15分、30分、1時間、2時間
            df[f'price_momentum_{period}'] = df['close'].pct_change(period)
            df[f'high_momentum_{period}'] = df['high'].pct_change(period)
            df[f'low_momentum_{period}'] = df['low'].pct_change(period)
        
        # RSI（5分足調整）
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=28).mean()  # 14期間の2倍
        loss = (-delta.where(delta < 0, 0)).rolling(window=28).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # -1 to 1に正規化
        
        # MACD（5分足調整）
        ema_fast = df['close'].ewm(span=24).mean()  # 2時間
        ema_slow = df['close'].ewm(span=52).mean()  # 4時間強
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=18).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_normalized'] = df['macd'] / df['close']  # 価格で正規化
        
        # ボリンジャーバンド
        bb_window = 40  # 約3時間強
        df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        bb_std = df['close'].rolling(window=bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 出来高分析
        df['volume_ma'] = df['volume'].rolling(window=24).mean()  # 2時間平均
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_price_trend'] = df['volume_ratio'] * df['oc_ratio']  # 出来高×価格変化
        
        # 時間帯特徴量（重要：イントラデイ取引）
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['time_of_day'] = df['hour'] + df['minute'] / 60
        df['morning_session'] = ((df['hour'] >= 9) & (df['hour'] < 11.5)).astype(int)
        df['afternoon_session'] = ((df['hour'] >= 12.5) & (df['hour'] < 15)).astype(int)
        df['session_start'] = ((df['hour'] == 9) | (df['hour'] == 12.5)).astype(int)
        df['session_end'] = ((df['hour'] >= 14.5)).astype(int)
        
        # ボラティリティ指標
        df['volatility_5m'] = df['close'].rolling(window=12).std()  # 1時間のボラティリティ
        df['volatility_30m'] = df['close'].rolling(window=72).std()  # 6時間のボラティリティ
        df['volatility_ratio'] = df['volatility_5m'] / df['volatility_30m']
        
        # 価格ギャップ
        df['gap_from_open'] = (df['close'] - df['open'].iloc[0]) / df['open'].iloc[0]
        df['gap_from_prev_close'] = df['close'].pct_change()
        
        # トレンド強度
        for window in [12, 24, 48]:
            highs = df['high'].rolling(window=window).max()
            lows = df['low'].rolling(window=window).min()
            df[f'trend_strength_{window}'] = (df['close'] - lows) / (highs - lows)
        
        return df
    
    def prepare_leak_free_training_data(self, symbol, target_date):
        """リークなし学習データ準備（前日17:00まで）"""
        conn = sqlite3.connect(self.db_path)
        
        # 前日の17:00まで（当日データは絶対に使わない）
        prev_day = self.get_previous_trading_day(target_date)
        cutoff_time = prev_day.replace(hour=17, minute=0, second=0, microsecond=0)
        
        # 学習用データ期間（過去30営業日分）
        start_date = cutoff_time - timedelta(days=45)  # 余裕を持って
        
        query = '''
            SELECT * FROM chart_data 
            WHERE symbol = ? AND datetime BETWEEN ? AND ?
            ORDER BY datetime
        '''
        
        data = pd.read_sql_query(query, conn, params=[
            symbol,
            start_date.strftime('%Y-%m-%d %H:%M:%S'),
            cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
        ])
        conn.close()
        
        if len(data) < 200:  # 最低限のデータ数
            return None, None, None, cutoff_time
        
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
        
        # 特徴量作成
        data = self.create_enhanced_features(data)
        
        # 目標変数：次の6期間後（30分後）の価格変化率
        data['target'] = data['close'].shift(-6).pct_change()
        
        # 特徴量列を選択（基本価格情報は除外）
        exclude_cols = ['symbol', 'target', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # NaNを除去
        ml_data = data[feature_cols + ['target']].dropna()
        
        if len(ml_data) < 100:
            return None, None, None, cutoff_time
        
        X = ml_data[feature_cols]
        y = ml_data['target']
        
        print(f"    学習データ準備完了: {len(X)}件 (カットオフ: {cutoff_time})")
        
        return X, y, feature_cols, cutoff_time
    
    def train_enhanced_models(self, symbol, target_date):
        """強化されたモデル学習"""
        print(f"  🤖 {symbol} モデル学習開始...")
        
        # リークなし学習データ準備
        X, y, feature_cols, cutoff_time = self.prepare_leak_free_training_data(symbol, target_date)
        
        if X is None:
            print(f"    ❌ {symbol}: データ不足")
            return None, None, cutoff_time
        
        # 時系列分割（最新20%を検証用）
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        models = {}
        model_scores = {}
        
        # RandomForest（パラメータ調整）
        try:
            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_val)
            rf_score = r2_score(y_val, rf_pred)
            models['RandomForest'] = rf
            model_scores['RandomForest'] = rf_score
        except Exception as e:
            print(f"      RF学習エラー: {e}")
        
        # LinearRegression（正則化）
        try:
            from sklearn.linear_model import Ridge
            lr = Ridge(alpha=1.0)
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_val)
            lr_score = r2_score(y_val, lr_pred)
            models['Ridge'] = lr
            model_scores['Ridge'] = lr_score
        except Exception as e:
            print(f"      Ridge学習エラー: {e}")
        
        # LightGBM（パラメータ調整）
        try:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_val)
            lgb_score = r2_score(y_val, lgb_pred)
            models['LightGBM'] = lgb_model
            model_scores['LightGBM'] = lgb_score
        except Exception as e:
            print(f"      LGB学習エラー: {e}")
        
        if not models:
            print(f"    ❌ {symbol}: 全モデル学習失敗")
            return None, None, cutoff_time
        
        # モデル保存
        model_date_str = target_date.strftime('%Y%m%d')
        model_file = os.path.join(self.models_dir, f"{symbol}_models_{model_date_str}.pkl")
        
        with open(model_file, 'wb') as f:
            pickle.dump({
                'models': models,
                'feature_cols': feature_cols,
                'scores': model_scores,
                'train_date': target_date,
                'cutoff_time': cutoff_time,
                'data_size': len(X),
                'val_size': len(X_val)
            }, f)
        
        # スコア表示
        scores_str = ", ".join([f"{k}:{v:.3f}" for k, v in model_scores.items()])
        best_score = max(model_scores.values())
        status = "🟢" if best_score > 0.1 else "🟡" if best_score > 0.0 else "🔴"
        
        print(f"    {status} {symbol}: 学習完了 (データ:{len(X)}, 検証:{len(X_val)})")
        print(f"      スコア: {scores_str}")
        
        return models, model_scores, cutoff_time
    
    def get_trading_day_features(self, symbol, target_date, prediction_time):
        """当日の指定時刻の特徴量を取得（リークなし）"""
        conn = sqlite3.connect(self.db_path)
        
        # 予測時刻まで（含まない）のデータを取得
        query = '''
            SELECT * FROM chart_data 
            WHERE symbol = ? AND datetime < ?
            ORDER BY datetime DESC
            LIMIT 200
        '''
        
        data = pd.read_sql_query(query, conn, params=[
            symbol,
            prediction_time.strftime('%Y-%m-%d %H:%M:%S')
        ])
        conn.close()
        
        if len(data) < 50:
            return None
        
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime').sort_index()
        
        # 特徴量作成
        data = self.create_enhanced_features(data)
        
        # 特徴量列を選択
        exclude_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # 最新の特徴量を取得
        latest_features = data[feature_cols].iloc[-1]
        
        return latest_features.dropna()
    
    def run_daily_cycle(self, symbols, target_dates):
        """日次サイクル実行：モデル更新→予測→取引→レポート"""
        print(f"\\n{'='*70}")
        print(f"yfinance実データ日次モデル更新サイクル開始")
        print(f"期間: {target_dates[0].date()} ～ {target_dates[-1].date()}")
        print(f"{'='*70}")
        
        # 1. yfinanceからデータ収集
        # 実際のデータ収集期間を調整（取引日の前日も含める）
        start_date = target_dates[0] - timedelta(days=10)  # 前日データが必要なので余裕を持つ
        end_date = target_dates[-1] + timedelta(days=1)
        
        if self.collect_yfinance_data(symbols, start_date, end_date) == 0:
            print("❌ データ収集に失敗しました")
            return []
        
        all_daily_results = []
        
        # 2. 各日のサイクル実行
        for day_num, target_date in enumerate(target_dates, 1):
            if target_date.weekday() >= 5:  # 土日スキップ
                continue
                
            print(f"\\n{'─'*50}")
            print(f"📅 Day {day_num}: {target_date.date()}")
            print(f"{'─'*50}")
            
            daily_result = {
                'date': target_date,
                'models_trained': 0,
                'trades': [],
                'model_scores': {}
            }
            
            # Step 1: 前日データでモデル学習
            print(f"🔄 Step 1: モデル学習 (前日17:00までのデータ)")
            
            for symbol in symbols:
                models, scores, cutoff_time = self.train_enhanced_models(symbol, target_date)
                if models:
                    daily_result['models_trained'] += 1
                    daily_result['model_scores'][symbol] = scores
            
            if daily_result['models_trained'] == 0:
                print("❌ モデル学習失敗")
                continue
            
            # Step 2: 当日取引シミュレーション
            print(f"📈 Step 2: 取引シミュレーション")
            
            # 9:30から14:30まで30分間隔で予測・取引判定
            trading_times = []
            current_time = target_date.replace(hour=9, minute=30)
            while current_time.hour < 15:
                trading_times.append(current_time)
                current_time += timedelta(minutes=30)
            
            for trading_time in trading_times:
                trades = self.simulate_trading_session(symbols, target_date, trading_time)
                daily_result['trades'].extend(trades)
            
            # Step 3: 日次レポート生成
            print(f"📊 Step 3: レポート生成")
            self.generate_daily_cycle_report(daily_result)
            
            all_daily_results.append(daily_result)
            
            print(f"✅ Day {day_num} 完了: モデル{daily_result['models_trained']}個, 取引{len(daily_result['trades'])}回")
        
        # 全期間サマリー
        self.generate_cycle_summary(all_daily_results)
        
        return all_daily_results
    
    def simulate_trading_session(self, symbols, target_date, prediction_time):
        """取引セッションシミュレーション"""
        trades = []
        
        for symbol in symbols:
            # 現在の特徴量取得
            features = self.get_trading_day_features(symbol, target_date, prediction_time)
            if features is None:
                continue
            
            # モデルで予測
            prediction = self.predict_with_saved_models(symbol, target_date, features)
            if prediction is None:
                continue
            
            ensemble_pred, individual_preds = prediction
            
            # 取引判定（より現実的な閾値）
            if abs(ensemble_pred) > 0.002:  # 0.2%以上の予測で取引
                trade = self.execute_simulated_trade(symbol, target_date, prediction_time, 
                                                   ensemble_pred, individual_preds)
                if trade:
                    trades.append(trade)
        
        return trades
    
    def predict_with_saved_models(self, symbol, target_date, features):
        """保存されたモデルで予測"""
        model_date_str = target_date.strftime('%Y%m%d')
        model_file = os.path.join(self.models_dir, f"{symbol}_models_{model_date_str}.pkl")
        
        if not os.path.exists(model_file):
            return None
        
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            models = model_data['models']
            feature_cols = model_data['feature_cols']
            scores = model_data['scores']
            
            # 特徴量の順序を合わせる
            features_aligned = features.reindex(feature_cols).fillna(0)
            
            predictions = {}
            for name, model in models.items():
                try:
                    pred = model.predict([features_aligned])[0]
                    predictions[name] = pred
                except:
                    predictions[name] = 0
            
            # スコア重み付きアンサンブル
            if scores and any(score > 0 for score in scores.values()):
                # 正のスコアのみ使用
                positive_scores = {k: max(v, 0.001) for k, v in scores.items()}
                weighted_pred = sum(pred * positive_scores.get(name, 0) for name, pred in predictions.items())
                weighted_pred /= sum(positive_scores.values())
            else:
                weighted_pred = np.mean(list(predictions.values()))
            
            return weighted_pred, predictions
            
        except Exception as e:
            return None
    
    def execute_simulated_trade(self, symbol, target_date, entry_time, ensemble_pred, individual_preds):
        """シミュレーション取引実行"""
        # 実際の価格データから取引価格を取得
        conn = sqlite3.connect(self.db_path)
        
        # エントリー価格
        entry_query = '''
            SELECT close FROM chart_data 
            WHERE symbol = ? AND datetime = ?
        '''
        entry_result = conn.execute(entry_query, [
            symbol, entry_time.strftime('%Y-%m-%d %H:%M:%S')
        ]).fetchone()
        
        if not entry_result:
            conn.close()
            return None
        
        entry_price = entry_result[0]
        
        # 決済価格（30分後）
        exit_time = entry_time + timedelta(minutes=30)
        exit_query = '''
            SELECT close FROM chart_data 
            WHERE symbol = ? AND datetime >= ?
            ORDER BY datetime LIMIT 1
        '''
        exit_result = conn.execute(exit_query, [
            symbol, exit_time.strftime('%Y-%m-%d %H:%M:%S')
        ]).fetchone()
        
        conn.close()
        
        if not exit_result:
            return None
        
        exit_price = exit_result[0]
        return_rate = (exit_price / entry_price - 1) - self.transaction_cost
        
        # 決済理由判定
        if return_rate >= self.take_profit:
            exit_reason = '利確'
        elif return_rate <= -self.stop_loss:
            exit_reason = '損切'
        else:
            exit_reason = '時間切れ'
        
        return {
            'symbol': symbol,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': return_rate,
            'exit_reason': exit_reason,
            'ensemble_pred': ensemble_pred,
            'individual_preds': individual_preds
        }
    
    def generate_daily_cycle_report(self, daily_result):
        """日次サイクルレポート生成"""
        date = daily_result['date']
        report_filename = f"leak_free_daily_{date.strftime('%Y%m%d')}.txt"
        report_path = os.path.join(self.reports_dir, report_filename)
        
        report = f"{'='*70}\n"
        report += f"{date.date()} リークなし日次取引サイクル レポート\n"
        report += f"{'='*70}\n"
        
        # モデル学習結果
        report += f"\n🤖 モデル学習結果:\n"
        report += f"学習済みモデル数: {daily_result['models_trained']}\n"
        
        for symbol, scores in daily_result['model_scores'].items():
            best_score = max(scores.values())
            best_model = max(scores.items(), key=lambda x: x[1])[0]
            report += f"  {symbol}: ベスト {best_model} (スコア: {best_score:.3f})\n"
        
        # 取引結果
        trades = daily_result['trades']
        if trades:
            total_return = sum(t['return'] for t in trades)
            winning_trades = len([t for t in trades if t['return'] > 0])
            
            report += f"\n📈 取引結果:\n"
            report += f"取引数: {len(trades)}\n"
            report += f"勝率: {winning_trades/len(trades):.1%} ({winning_trades}/{len(trades)})\n"
            report += f"日次収益率: {total_return:.2%}\n"
            
            report += f"\n取引詳細:\n"
            for i, trade in enumerate(trades, 1):
                report += f"  [{i}] {trade['symbol']} {trade['entry_time'].strftime('%H:%M')}\n"
                report += f"      ¥{trade['entry_price']:.0f} → ¥{trade['exit_price']:.0f} "
                report += f"({trade['return']:+.2%}) [{trade['exit_reason']}]\n"
                report += f"      予測: {trade['ensemble_pred']:+.3f}\n"
        else:
            report += f"\n📊 取引結果: 本日は取引なし\n"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def generate_cycle_summary(self, all_results):
        """全期間サマリー生成"""
        summary_path = os.path.join(self.reports_dir, f"leak_free_summary_{datetime.now().strftime('%Y%m%d')}.txt")
        
        total_trades = sum(len(r['trades']) for r in all_results)
        total_return = sum(sum(t['return'] for t in r['trades']) for r in all_results)
        
        summary = f"{'='*70}\n"
        summary += f"yfinance実データ リークなし日次サイクル 総合結果\n"
        summary += f"{'='*70}\n"
        summary += f"期間: {all_results[0]['date'].date()} ～ {all_results[-1]['date'].date()}\n"
        summary += f"実行日数: {len(all_results)}日\n"
        summary += f"総取引数: {total_trades}回\n"
        summary += f"総収益率: {total_return:.2%}\n"
        summary += f"日次平均収益率: {total_return/len(all_results):.3%}\n"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\n📋 サマリー保存: {summary_path}")

def main():
    """メイン実行"""
    print("yfinance実データ リークなし日次モデル更新システム")
    print("="*60)
    
    symbols = ['7203', '6758', '8306', '9984', '6861']
    system = LeakFreeModelSystem()
    
    # yfinanceで利用可能な最近の期間を使用
    # 実際のデータ範囲: 2025-07-14 ～ 2025-07-18
    start_date = datetime(2025, 7, 15)  # 火曜日から
    end_date = datetime(2025, 7, 18)    # 金曜日まで
    
    print(f"データ期間設定: {start_date.date()} ～ {end_date.date()}")
    
    target_dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # 平日のみ
            target_dates.append(current)
        current += timedelta(days=1)
    
    print(f"対象取引日: {len(target_dates)}日")
    
    # 日次サイクル実行
    results = system.run_daily_cycle(symbols, target_dates)
    
    print(f"\\n🎉 リークなし日次モデル更新サイクル完了!")
    print(f"実行日数: {len(results)}日")
    print(f"レポート保存先: {system.reports_dir}/")

if __name__ == "__main__":
    main()
