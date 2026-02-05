#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真の日次モデル更新取引システム
毎日前日データでモデルを再学習してからレポート生成
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

class DailyModelUpdateSystem:
    def __init__(self, db_path='daily_trading_data.db'):
        self.db_path = db_path
        self.volume_threshold = 300000  # 30万株
        self.risk_free_rate = 0.001
        
        # 取引パラメータ
        self.stop_loss = 0.02  # 2%
        self.take_profit = 0.03  # 3%
        self.transaction_cost = 0.001  # 0.1%
        
        # モデル保存ディレクトリ
        self.models_dir = "daily_models"
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        # レポート保存ディレクトリ
        self.reports_dir = "daily_model_reports"
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
            
        print(f"日次モデル更新システム初期化:")
        print(f"- データベース: {self.db_path}")
        print(f"- モデル保存先: {self.models_dir}/")
        print(f"- レポート保存先: {self.reports_dir}/")
    
    def get_previous_trading_day(self, date):
        """前営業日を取得"""
        prev_day = date - timedelta(days=1)
        while prev_day.weekday() >= 5 or jpholiday.is_holiday(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day
    
    def create_demo_database(self):
        """デモ用データベースを作成"""
        conn = sqlite3.connect(self.db_path)
        
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
        
        # デモデータ生成（60日間）
        symbols = ['7203', '6758', '8306', '9984', '6861']
        end_date = datetime(2025, 7, 18)
        start_date = end_date - timedelta(days=90)
        
        print("デモデータ生成中...")
        
        for symbol in symbols:
            base_prices = {
                '7203': 3200, '6758': 24500, '8306': 950, 
                '9984': 12200, '6861': 1950
            }
            base_price = base_prices[symbol]
            current_price = base_price
            
            current_date = start_date
            while current_date <= end_date:
                if current_date.weekday() < 5:  # 平日のみ
                    # 1日の取引データ生成（5分足、9:00-15:00）
                    trading_hours = []
                    for hour in range(9, 15):
                        for minute in range(0, 60, 5):
                            if hour == 14 and minute > 30:  # 14:30まで
                                break
                            trading_hours.append((hour, minute))
                    
                    daily_volume_base = np.random.randint(5000000, 20000000)
                    
                    for hour, minute in trading_hours:
                        # 価格変動
                        price_change = np.random.normal(0, 0.005)  # 0.5%の標準偏差
                        current_price = max(current_price * (1 + price_change), 100)
                        
                        # 出来高
                        volume = max(int(daily_volume_base * np.random.uniform(0.5, 1.5) / len(trading_hours)), 1000)
                        
                        timestamp = current_date.replace(hour=hour, minute=minute)
                        
                        conn.execute('''
                            INSERT OR REPLACE INTO chart_data 
                            (symbol, datetime, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (symbol, timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                              current_price, current_price * 1.002, current_price * 0.998, 
                              current_price, volume))
                
                current_date += timedelta(days=1)
        
        conn.commit()
        conn.close()
        print("デモデータ生成完了")
    
    def create_features(self, data):
        """テクニカル指標の作成"""
        df = data.copy()
        df.columns = df.columns.str.lower()
        
        # 移動平均
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}'] - 1
        
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
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # ボリンジャーバンド
        bb_window = 20
        df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        bb_std = df['close'].rolling(window=bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_ratio'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 価格変化率
        for period in [1, 3, 5, 10]:
            df[f'price_change_{period}'] = df['close'].pct_change(period)
        
        # 出来高関連
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['price_volume'] = df['close'].pct_change() * df['volume_ratio']
        
        # ボラティリティ
        df['volatility'] = df['close'].rolling(window=10).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=20).mean()
        
        return df
    
    def prepare_training_data(self, symbol, target_date, lookback_days=30):
        """指定日の前日までのデータで学習データを準備"""
        conn = sqlite3.connect(self.db_path)
        
        # 前日まで30日分のデータを取得
        start_date = target_date - timedelta(days=lookback_days + 10)  # 余裕を持って
        end_date = self.get_previous_trading_day(target_date)
        
        query = '''
            SELECT * FROM chart_data 
            WHERE symbol = ? AND datetime BETWEEN ? AND ?
            ORDER BY datetime
        '''
        
        data = pd.read_sql_query(query, conn, params=[
            symbol, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d 23:59:59')
        ])
        conn.close()
        
        if len(data) < 100:  # 最低限のデータ数
            return None, None, None
        
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
            return None, None, None
        
        X = ml_data[feature_cols]
        y = ml_data['target']
        
        return X, y, feature_cols
    
    def train_daily_models(self, symbol, target_date):
        """指定日用のモデルを学習"""
        print(f"    {symbol} モデル学習開始...")
        
        # 学習データ準備
        X, y, feature_cols = self.prepare_training_data(symbol, target_date)
        
        if X is None:
            print(f"    {symbol}: データ不足でモデル学習スキップ")
            return None, None
        
        # 訓練・検証分割
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        models = {}
        model_scores = {}
        
        # RandomForest
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_val)
            rf_score = r2_score(y_val, rf_pred)
            models['RandomForest'] = rf
            model_scores['RandomForest'] = rf_score
        except Exception as e:
            print(f"    RandomForest学習エラー: {e}")
        
        # LinearRegression
        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_val)
            lr_score = r2_score(y_val, lr_pred)
            models['LinearRegression'] = lr
            model_scores['LinearRegression'] = lr_score
        except Exception as e:
            print(f"    LinearRegression学習エラー: {e}")
        
        # LightGBM
        try:
            lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_val)
            lgb_score = r2_score(y_val, lgb_pred)
            models['LightGBM'] = lgb_model
            model_scores['LightGBM'] = lgb_score
        except Exception as e:
            print(f"    LightGBM学習エラー: {e}")
        
        if not models:
            print(f"    {symbol}: 全モデル学習失敗")
            return None, None
        
        # モデル保存
        model_date_str = target_date.strftime('%Y%m%d')
        model_file = os.path.join(self.models_dir, f"{symbol}_models_{model_date_str}.pkl")
        
        with open(model_file, 'wb') as f:
            pickle.dump({
                'models': models,
                'feature_cols': feature_cols,
                'scores': model_scores,
                'train_date': target_date,
                'data_size': len(X)
            }, f)
        
        print(f"    {symbol}: 学習完了 (データ数: {len(X)}, モデル数: {len(models)})")
        print(f"    スコア: " + ", ".join([f"{k}:{v:.3f}" for k, v in model_scores.items()]))
        
        return models, model_scores
    
    def predict_with_models(self, symbol, target_date, current_features):
        """保存されたモデルで予測"""
        model_date_str = target_date.strftime('%Y%m%d')
        model_file = os.path.join(self.models_dir, f"{symbol}_models_{model_date_str}.pkl")
        
        if not os.path.exists(model_file):
            return None, None
        
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            models = model_data['models']
            feature_cols = model_data['feature_cols']
            scores = model_data['scores']
            
            # 特徴量の順序を合わせる
            features_aligned = current_features.reindex(feature_cols).fillna(0)
            
            predictions = {}
            for name, model in models.items():
                try:
                    pred = model.predict([features_aligned])[0]
                    predictions[name] = pred
                except Exception as e:
                    print(f"    {symbol} {name}予測エラー: {e}")
                    predictions[name] = 0
            
            # スコア重み付きアンサンブル
            if scores and sum(scores.values()) > 0:
                weighted_pred = sum(pred * scores.get(name, 0) for name, pred in predictions.items())
                weighted_pred /= sum(scores.values())
            else:
                weighted_pred = np.mean(list(predictions.values()))
            
            return weighted_pred, predictions
            
        except Exception as e:
            print(f"    {symbol} モデル読み込みエラー: {e}")
            return None, None
    
    def generate_daily_report_with_models(self, target_date, symbols):
        """モデル更新付きの日次レポート生成"""
        print(f"\\n{'='*60}")
        print(f"{target_date.date()} モデル更新 & 取引レポート生成")
        print(f"{'='*60}")
        
        # 1. 前日までのデータで各銘柄のモデルを更新
        print(f"\\nStep 1: モデル学習・更新")
        print("-" * 30)
        
        model_info = {}
        for symbol in symbols:
            models, scores = self.train_daily_models(symbol, target_date)
            if models:
                model_info[symbol] = {
                    'models': models,
                    'scores': scores,
                    'model_count': len(models)
                }
        
        # 2. 当日のデータで予測・取引シミュレーション
        print(f"\\nStep 2: 取引シミュレーション")
        print("-" * 30)
        
        trades = []
        for symbol in symbols:
            if symbol not in model_info:
                continue
                
            # 当日の特徴量を取得（簡易版）
            current_features = self.get_current_features_demo(symbol, target_date)
            if current_features is None:
                continue
            
            # 予測実行
            ensemble_pred, individual_preds = self.predict_with_models(symbol, target_date, current_features)
            
            if ensemble_pred is None:
                continue
            
            # 取引判定（予測値が0.5%以上で買いエントリー）
            if ensemble_pred > 0.005:
                trade = self.simulate_trade(symbol, target_date, ensemble_pred, individual_preds)
                if trade:
                    trades.append(trade)
                    print(f"    {symbol}: エントリー (予測: {ensemble_pred:.3f})")
        
        # 3. レポート生成
        print(f"\\nStep 3: レポート生成")
        print("-" * 30)
        
        report = self.generate_detailed_daily_report(target_date, trades, model_info)
        
        # ファイル保存
        report_filename = f"model_daily_report_{target_date.strftime('%Y%m%d')}.txt"
        report_filepath = os.path.join(self.reports_dir, report_filename)
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\\nレポート保存: {report_filename}")
        print(f"取引数: {len(trades)}")
        
        return trades, model_info
    
    def get_current_features_demo(self, symbol, target_date):
        """デモ用：当日の特徴量を取得"""
        conn = sqlite3.connect(self.db_path)
        
        # 当日の10:00時点のデータを取得
        query_time = target_date.replace(hour=10, minute=0)
        
        query = '''
            SELECT * FROM chart_data 
            WHERE symbol = ? AND datetime <= ?
            ORDER BY datetime DESC
            LIMIT 100
        '''
        
        data = pd.read_sql_query(query, conn, params=[
            symbol, 
            query_time.strftime('%Y-%m-%d %H:%M:%S')
        ])
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
    
    def simulate_trade(self, symbol, target_date, ensemble_pred, individual_preds):
        """取引シミュレーション"""
        # 簡易的なエントリー・決済価格生成
        base_prices = {'7203': 3200, '6758': 24500, '8306': 950, '9984': 12200, '6861': 1950}
        base_price = base_prices.get(symbol, 1000)
        
        entry_price = base_price * (1 + np.random.uniform(-0.01, 0.01))
        
        # 予測に基づく価格変動（ノイズ付き）
        actual_change = ensemble_pred + np.random.normal(0, 0.01)
        exit_price = entry_price * (1 + actual_change)
        
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
            'entry_time': target_date.replace(hour=10, minute=0),
            'exit_time': target_date.replace(hour=15, minute=0),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': return_rate,
            'exit_reason': exit_reason,
            'ensemble_pred': ensemble_pred,
            'individual_preds': individual_preds
        }
    
    def generate_detailed_daily_report(self, target_date, trades, model_info):
        """詳細な日次レポート生成"""
        report = f"{'='*70}\\n"
        report += f"{target_date.date()} 日次モデル更新 & 取引レポート\\n"
        report += f"{'='*70}\\n"
        
        # モデル更新情報
        report += f"\\n🤖 モデル更新情報:\\n"
        report += f"{'─'*40}\\n"
        for symbol, info in model_info.items():
            scores_str = ", ".join([f"{k}:{v:.3f}" for k, v in info['scores'].items()])
            report += f"  {symbol}: {info['model_count']}モデル学習完了\\n"
            report += f"    スコア: {scores_str}\\n"
        
        # 取引結果
        if not trades:
            report += f"\\n📊 取引結果: 本日は取引なし\\n"
            report += f"    理由: 予測信頼度が閾値(0.5%)を下回ったため\\n"
            return report
        
        # 基本統計
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['return'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_return = sum([t['return'] for t in trades])
        
        report += f"\\n📊 取引サマリー:\\n"
        report += f"{'─'*40}\\n"
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
        report += f"\\n📈 取引詳細:\\n"
        report += f"{'─'*40}\\n"
        for i, trade in enumerate(trades, 1):
            entry_time = trade['entry_time'].strftime('%H:%M')
            exit_time = trade['exit_time'].strftime('%H:%M')
            report += f"  [{i}] {trade['symbol']}: {entry_time}-{exit_time}\\n"
            report += f"      価格: ¥{trade['entry_price']:.0f} → ¥{trade['exit_price']:.0f}\\n"
            report += f"      収益: {trade['return']:+.2%} [{trade['exit_reason']}]\\n"
            
            # モデル予測詳細
            report += f"      予測: アンサンブル {trade['ensemble_pred']:.3f}\\n"
            preds = trade['individual_preds']
            report += f"            個別モデル: "
            pred_strs = [f"{k}:{v:.3f}" for k, v in preds.items()]
            report += ", ".join(pred_strs) + "\\n"
            report += f"\\n"
        
        return report

def main():
    """メイン実行関数"""
    print("日次モデル更新取引システム")
    print("="*50)
    
    symbols = ['7203', '6758', '8306', '9984', '6861']
    system = DailyModelUpdateSystem()
    
    # デモデータベース作成
    system.create_demo_database()
    
    # サンプル日付でモデル更新&取引実行
    test_dates = [
        datetime(2025, 7, 15),
        datetime(2025, 7, 16),
        datetime(2025, 7, 17),
        datetime(2025, 7, 18)
    ]
    
    all_trades = []
    for test_date in test_dates:
        trades, model_info = system.generate_daily_report_with_models(test_date, symbols)
        all_trades.extend(trades)
    
    print(f"\\n{'='*60}")
    print(f"4日間のモデル更新取引システム実行完了!")
    print(f"総取引数: {len(all_trades)}")
    print(f"モデルファイル: {system.models_dir}/")
    print(f"レポートファイル: {system.reports_dir}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
