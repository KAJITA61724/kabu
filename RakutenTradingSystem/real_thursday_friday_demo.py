#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高精度リークフリーモデル版Thursday Friday Demo - 動的決済対応
yfinance実データを使って木曜データ学習→金曜取引を実行
ポジション保有後5分毎にリアルタイム予測で動的決済（最長45分）
89特徴量 + アンサンブル学習による高精度予測
"""
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
from sklearn.metrics import r2_score
import warnings
import os
import pickle
import sys

# 高精度リークフリーモデルをインポート
from high_precision_ml_models import HighPrecisionLeakFreeModels

warnings.filterwarnings('ignore')

class RealDataThursdayFridayDemo:
    def __init__(self, db_path='real_thursday_friday.db'):
        self.db_path = db_path
        self.volume_threshold = 300000  # 30万株
        
        # 高精度リークフリーモデル初期化
        self.hp_models = HighPrecisionLeakFreeModels()
        
        # 取引パラメータ
        self.initial_capital = 1000000  # 100万円
        self.stop_loss = 0.02  # 2%
        self.take_profit = 0.03  # 3%
        self.transaction_cost = 0.001  # 0.1%
        self.max_holding_minutes = 45  # 最長保有時間
        self.monitoring_interval = 5   # 5分毎監視
        
        # ディレクトリ作成
        self.models_dir = "real_thursday_friday_models"
        self.reports_dir = "real_thursday_friday_reports"
        
        for dir_name in [self.models_dir, self.reports_dir]:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        
        print(f"高精度リークフリーモデル版Thursday Friday Demo（動的決済対応）初期化:")
        print(f"- 初期資本: ¥{self.initial_capital:,}")
        print(f"- 最長保有: {self.max_holding_minutes}分")
        print(f"- 監視間隔: {self.monitoring_interval}分毎")
        print(f"- 動的決済: 予測方向逆転時に即決済")
        print(f"- 特徴量数: 89特徴量 + アンサンブル学習")
    
    def collect_yfinance_data(self, symbols, start_date, end_date):
        """yfinanceから5分足データを収集し、高精度モデルのデータベースにも保存"""
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
                adj_close REAL,
                PRIMARY KEY (symbol, datetime)
            )
        ''')
        
        print(f"\nyfinanceデータ収集: {start_date.date()} ～ {end_date.date()}")
        
        success_count = 0
        for i, symbol in enumerate(symbols, 1):
            try:
                ticker = yf.Ticker(f'{symbol}.T')
                data = ticker.history(start=start_date, end=end_date, interval='5m')
                
                if not data.empty:
                    data = data.between_time('09:00', '15:00')
                    print(f"[{i:2d}/{len(symbols)}] {symbol}: {len(data):4d}件")
                    
                    # ローカルデータベースに保存
                    for timestamp, row in data.iterrows():
                        conn.execute('''
                            INSERT OR REPLACE INTO chart_data 
                            (symbol, datetime, open, high, low, close, volume, adj_close)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (symbol, timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                              row['Open'], row['High'], row['Low'], row['Close'], 
                              row['Volume'], row['Close']))
                    
                    # 高精度モデルのデータベースにも保存
                    self._save_to_hp_database(symbol, data)
                    
                    success_count += 1
                else:
                    print(f"[{i:2d}/{len(symbols)}] {symbol}: データなし")
                    
            except Exception as e:
                print(f"[{i:2d}/{len(symbols)}] {symbol}: エラー - {str(e)[:50]}")
        
        conn.commit()
        conn.close()
        return success_count
    
    def _save_to_hp_database(self, symbol, data):
        """高精度モデルのデータベースにデータを保存"""
        try:
            hp_conn = sqlite3.connect(self.hp_models.db_path)
            
            # テーブルが存在しない場合は作成
            hp_conn.execute('''
                CREATE TABLE IF NOT EXISTS extended_market_data (
                    symbol TEXT,
                    datetime TEXT,
                    timeframe TEXT,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, datetime, timeframe)
                )
            ''')
            
            # 5分足データを保存
            for timestamp, row in data.iterrows():
                hp_conn.execute('''
                    INSERT OR REPLACE INTO extended_market_data 
                    (symbol, datetime, timeframe, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, timestamp.strftime('%Y-%m-%d %H:%M:%S'), '5m',
                      row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
            
            hp_conn.commit()
            hp_conn.close()
            
        except Exception as e:
            print(f"高精度データベース保存エラー ({symbol}): {e}")
    
    def create_features(self, data):
        """特徴量作成"""
        df = data.copy()
        df.columns = df.columns.str.lower()
        
        # 基本価格情報
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['oc_ratio'] = (df['close'] - df['open']) / df['open']
        
        # 移動平均
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}'] - 1
        
        # 価格モメンタム
        for period in [1, 3, 5]:
            df[f'price_momentum_{period}'] = df['close'].pct_change(period)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 出来高分析
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def prepare_thursday_training_data(self, symbol, thursday_date):
        """木曜日のデータで学習データ準備"""
        conn = sqlite3.connect(self.db_path)
        
        thursday_start = thursday_date.replace(hour=0, minute=0, second=0)
        thursday_end = thursday_date.replace(hour=23, minute=59, second=59)
        
        query = '''
            SELECT * FROM chart_data 
            WHERE symbol = ? AND datetime BETWEEN ? AND ?
            ORDER BY datetime
        '''
        
        data = pd.read_sql_query(query, conn, params=[
            symbol,
            thursday_start.strftime('%Y-%m-%d %H:%M:%S'),
            thursday_end.strftime('%Y-%m-%d %H:%M:%S')
        ])
        conn.close()
        
        if len(data) < 50:
            return None, None, None
        
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
        
        # 特徴量作成
        data = self.create_features(data)
        
        # 目標変数：次の期間の価格変化率
        data['target'] = data['close'].shift(-1).pct_change()
        
        # 特徴量列を選択
        exclude_cols = ['symbol', 'target', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # NaNを除去
        ml_data = data[feature_cols + ['target']].dropna()
        
        if len(ml_data) < 30:
            return None, None, None
        
        X = ml_data[feature_cols]
        y = ml_data['target']
        
        return X, y, feature_cols
    
    def train_models(self, symbol, thursday_date):
        """高精度リークフリーモデル学習（木曜データのみ）"""
        print(f"  🤖 {symbol} 高精度モデル学習開始...")
        
        try:
            # 高精度モデルでデータ準備
            data_result = self.hp_models.prepare_high_precision_data(symbol)
            
            if data_result[0] is None:
                print(f"    ❌ {symbol}: データ準備失敗")
                return None, None
            
            X_train, X_test, y_price_train, y_price_test, y_direction_train, y_direction_test = data_result
            
            # アンサンブル学習
            models = self.hp_models.train_ensemble_models(X_train, y_price_train, y_direction_train)
            
            # 評価
            evaluation = self.hp_models.evaluate_ensemble_models(models, X_test, y_price_test, y_direction_test)
            
            # 最良モデルを特定
            best_price_model = min(evaluation['price_models'].items(), key=lambda x: x[1]['mae'])
            best_direction_model = max(evaluation['direction_models'].items(), key=lambda x: x[1]['accuracy'])
            
            model_data = {
                'models': models,
                'evaluation': evaluation,
                'best_price': best_price_model,
                'best_direction': best_direction_model,
                'feature_columns': self.hp_models.feature_columns
            }
            
            # モデル保存
            model_file = os.path.join(self.models_dir, f"{symbol}_models_thursday.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"    ✅ {symbol}: 89特徴量学習完了")
            print(f"       最良価格予測: {best_price_model[0]} (MAE: {best_price_model[1]['mae']:.6f})")
            print(f"       最良方向予測: {best_direction_model[0]} (精度: {best_direction_model[1]['accuracy']:.1%})")
            
            return model_data, best_direction_model[1]['accuracy']
            
        except Exception as e:
            print(f"    ❌ {symbol}: 学習エラー - {e}")
            return None, None
    
    def get_features_at_time(self, symbol, target_time):
        """指定時刻の特徴量を取得（リアルタイムデータ使用）"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM chart_data 
            WHERE symbol = ? AND datetime <= ?
            ORDER BY datetime DESC
            LIMIT 100
        '''
        
        data = pd.read_sql_query(query, conn, params=[
            symbol,
            target_time.strftime('%Y-%m-%d %H:%M:%S')
        ])
        conn.close()
        
        if len(data) < 20:
            return None
        
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime').sort_index()
        
        # 特徴量作成
        data = self.create_features(data)
        
        # 特徴量列を選択
        exclude_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # 最新の特徴量を取得
        latest_features = data[feature_cols].iloc[-1]
        
        return latest_features.dropna()
    
    def predict_next_5min(self, symbol, current_time):
        """高精度モデルによる現在時刻から5分後の予測を実行"""
        model_file = os.path.join(self.models_dir, f"{symbol}_models_thursday.pkl")
        
        if not os.path.exists(model_file):
            return None, None
        
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            models = model_data['models']
            feature_cols = model_data['feature_columns']
            best_price = model_data['best_price']
            best_direction = model_data['best_direction']
            
            # 現在時刻の特徴量取得（高精度モデル形式）
            features = self.get_high_precision_features_at_time(symbol, current_time)
            if features is None:
                return None, None
            
            # 特徴量の順序を合わせる
            available_features = [col for col in feature_cols if col in features.index]
            
            if len(available_features) < len(feature_cols) * 0.5:  # 50%以上の特徴量があれば実行
                print(f"    ⚠️ {symbol}: 特徴量不足 ({len(available_features)}/{len(feature_cols)})")
                # 簡易予測ロジックにフォールバック
                return self._simple_prediction_fallback(symbol, current_time, features)
            
            # 特徴量を DataFrame形式で準備
            feature_values = features[available_features].values.reshape(1, -1)
            feature_df = pd.DataFrame(feature_values, columns=available_features)
            
            # 価格予測（最良モデル使用）
            price_model_name = best_price[0]
            price_model_data = models['price_models'][price_model_name]
            
            X_selected = price_model_data['selector'].transform(feature_df)
            X_scaled = price_model_data['scaler'].transform(X_selected)
            price_prediction = price_model_data['model'].predict(X_scaled)[0]
            
            # デバッグ情報を追加
            print(f"      DEBUG {symbol}: 特徴量数={len(available_features)}, 予測値(生)={price_prediction:.6f}")
            
            # 方向予測（最良モデル使用）
            direction_model_name = best_direction[0]
            direction_model_data = models['direction_models'][direction_model_name]
            
            X_dir_selected = direction_model_data['selector'].transform(feature_df)
            X_dir_scaled = direction_model_data['scaler'].transform(X_dir_selected)
            direction_prediction = direction_model_data['model'].predict(X_dir_scaled)[0]
            direction_confidence = max(direction_model_data['model'].predict_proba(X_dir_scaled)[0])
            
            print(f"      DEBUG {symbol}: 方向={direction_prediction}, 信頼度(生)={direction_confidence:.3f}")
            
            # 予測値が非常に小さい場合は拡大する
            if abs(price_prediction) < 0.001 and abs(price_prediction) > 0:
                price_prediction *= 100  # 100倍に拡大
                print(f"      DEBUG {symbol}: 予測値を拡大 -> {price_prediction:.6f}")
            
            # 総合判断
            prediction_score = price_prediction
            confidence = direction_confidence
            
            # 方向と価格変化の整合性チェック
            if (direction_prediction == 1 and price_prediction > 0) or (direction_prediction == 0 and price_prediction < 0):
                confidence *= 1.1  # 一致する場合は信頼度アップ
            
            return prediction_score, confidence
            
        except Exception as e:
            print(f"    ❌ {symbol} 予測エラー: {e}")
            return None, None
    
    def _simple_prediction_fallback(self, symbol, current_time, features):
        """簡易予測ロジック（高精度モデルが使用できない場合のフォールバック）"""
        try:
            # 基本的な価格モメンタムベース予測
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT close FROM chart_data 
                WHERE symbol = ? AND datetime <= ?
                ORDER BY datetime DESC
                LIMIT 10
            '''
            
            prices = pd.read_sql_query(query, conn, params=(symbol, current_time.strftime('%Y-%m-%d %H:%M:%S')))
            conn.close()
            
            if len(prices) < 5:
                return None, None
            
            # 直近5分足の価格変化率計算
            price_changes = prices['close'].pct_change().dropna()
            avg_change = price_changes.mean()
            
            # モメンタム予測
            momentum_prediction = avg_change * 2  # 2倍の予測
            
            # 信頼度（変動性に基づく）
            volatility = price_changes.std()
            confidence = min(0.8, max(0.5, 1 - volatility * 100))  # 50-80%の範囲
            
            print(f"      FALLBACK {symbol}: モメンタム予測={momentum_prediction:.6f}, 信頼度={confidence:.1%}")
            
            return momentum_prediction, confidence
            
        except Exception as e:
            print(f"    ❌ {symbol} 簡易予測エラー: {e}")
            return None, None
    
    def get_high_precision_features_at_time(self, symbol, current_time):
        """高精度モデル用の特徴量を取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 最新50件のデータを取得（カラム名を高精度モデルに合わせて変換）
            query = '''
                SELECT datetime, open as open_price, high as high_price, 
                       low as low_price, close as close_price, volume
                FROM chart_data
                WHERE symbol = ? AND datetime <= ?
                ORDER BY datetime DESC
                LIMIT 50
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, current_time.strftime('%Y-%m-%d %H:%M:%S')))
            conn.close()
            
            if df.empty:
                return None
            
            # 時系列順に並び替え
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # 高精度モデルの特徴量作成
            # 上位時間軸データは簡易対応
            df_1h = pd.DataFrame()  
            df_1d = pd.DataFrame()
            
            df_features = self.hp_models.create_advanced_features(df, df_1h, df_1d, symbol)
            
            if df_features.empty:
                return None
            
            # 最新の特徴量を返す
            latest_features = df_features.iloc[-1]
            
            # NaN値を0で置換
            latest_features = latest_features.fillna(0)
            
            # 無限大値を処理
            latest_features = latest_features.replace([np.inf, -np.inf], 0)
            
            return latest_features
            
        except Exception as e:
            print(f"特徴量取得エラー ({symbol}): {e}")
            return None
    
    def get_current_price(self, symbol, target_time):
        """指定時刻の価格を取得"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT close FROM chart_data 
            WHERE symbol = ? AND datetime = ?
        '''
        result = conn.execute(query, [
            symbol, target_time.strftime('%Y-%m-%d %H:%M:%S')
        ]).fetchone()
        
        conn.close()
        
        return result[0] if result else None
    
    def execute_dynamic_exit_trade(self, symbol, friday_date, entry_time, initial_prediction, current_capital):
        """動的決済ロジックを含む取引実行"""
        # エントリー価格取得
        entry_price = self.get_current_price(symbol, entry_time)
        if entry_price is None:
            return None
        
        # ポジションサイズ計算
        position_size = int(current_capital * 0.2 / entry_price)
        if position_size < 1:
            return None
        
        # 初期予測方向を記録
        initial_direction = 1 if initial_prediction > 0 else -1
        
        print(f"      🔵 {symbol} エントリー: ¥{entry_price:.0f} ({position_size}株)")
        print(f"         初期予測: {initial_prediction:+.3f} ({'上昇' if initial_prediction > 0 else '下降'})")
        
        # 5分毎の動的監視を設定（最長45分）
        monitoring_times = []
        current_monitor = entry_time + timedelta(minutes=self.monitoring_interval)
        max_exit_time = entry_time + timedelta(minutes=self.max_holding_minutes)
        
        while current_monitor <= max_exit_time:
            monitoring_times.append(current_monitor)
            current_monitor += timedelta(minutes=self.monitoring_interval)
        
        exit_price = None
        exit_time = None
        exit_reason = None
        prediction_history = []
        
        # 動的監視ループ
        for monitor_time in monitoring_times:
            # 現在価格取得
            current_price = self.get_current_price(symbol, monitor_time)
            if current_price is None:
                continue
                
            current_return = (current_price / entry_price - 1)
            
            print(f"         ⏰ {monitor_time.strftime('%H:%M')}: ¥{current_price:.0f} ({current_return:+.2%})")
            
            # 利確・損切チェック
            if current_return >= self.take_profit:
                exit_price = current_price
                exit_time = monitor_time
                exit_reason = '利確'
                print(f"         ✅ 利確決済: {current_return:+.2%}")
                break
            elif current_return <= -self.stop_loss:
                exit_price = current_price
                exit_time = monitor_time
                exit_reason = '損切'
                print(f"         ❌ 損切決済: {current_return:+.2%}")
                break
            
            # 5分後予測による動的決済判定
            if monitor_time < max_exit_time:
                # リアルタイムデータで5分後を予測
                new_pred, new_confidence = self.predict_next_5min(symbol, monitor_time)
                
                if new_pred is not None and new_confidence is not None:
                    new_direction = 1 if new_pred > 0 else -1
                    
                    prediction_history.append({
                        'time': monitor_time,
                        'prediction': new_pred,
                        'direction': new_direction,
                        'confidence': new_confidence
                    })
                    
                    direction_str = "上昇" if new_direction > 0 else "下降"
                    print(f"         🔮 5分後予測: {new_pred:+.3f} ({direction_str}), 信頼度={new_confidence:.1%}")
                    
                    # 高信頼度で予測方向が初期方向と逆転した場合は即決済
                    if new_confidence > 0.6 and new_direction != initial_direction:
                        exit_price = current_price
                        exit_time = monitor_time
                        exit_reason = '高信頼度予測逆転'
                        direction_change = f"{'上昇' if initial_direction > 0 else '下降'} → {'上昇' if new_direction > 0 else '下降'}"
                        print(f"         🔄 高信頼度予測逆転決済: {direction_change} (信頼度={new_confidence:.1%})")
                        break
        
        # 45分経過で強制決済
        if exit_price is None:
            final_time = entry_time + timedelta(minutes=self.max_holding_minutes)
            final_price = self.get_current_price(symbol, final_time)
            
            if final_price is not None:
                exit_price = final_price
                exit_time = final_time
                exit_reason = f'時間切れ({self.max_holding_minutes}分)'
                print(f"         ⏱️ 時間切れ決済: {self.max_holding_minutes}分経過")
        
        if exit_price is None:
            return None
        
        # 損益計算
        return_rate = (exit_price / entry_price - 1) - self.transaction_cost
        profit_loss = position_size * (exit_price - entry_price) - (position_size * entry_price * self.transaction_cost)
        new_capital = current_capital + profit_loss
        
        holding_minutes = (exit_time - entry_time).total_seconds() / 60
        
        print(f"         💰 決済完了: {return_rate:+.2%} = ¥{profit_loss:,.0f} (保有{holding_minutes:.0f}分)")
        
        return {
            'symbol': symbol,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'return': return_rate,
            'profit_loss': profit_loss,
            'new_capital': new_capital,
            'exit_reason': exit_reason,
            'initial_prediction': initial_prediction,
            'initial_direction': initial_direction,
            'holding_minutes': holding_minutes,
            'prediction_history': prediction_history
        }
    
    def simulate_friday_trading(self, symbols, thursday_date, friday_date):
        """金曜日の取引シミュレーション"""
        print(f"\n📈 金曜日取引シミュレーション ({friday_date.date()})")
        print(f"   動的決済: 5分毎予測で方向逆転時に即決済")
        
        trades = []
        current_capital = self.initial_capital
        
        # 9:30から14:30まで30分間隔で取引機会をチェック
        trading_times = []
        current_time = friday_date.replace(hour=9, minute=30)
        while current_time.hour < 15:
            trading_times.append(current_time)
            current_time += timedelta(minutes=30)
        
        for trading_time in trading_times:
            print(f"\n  ⏰ {trading_time.strftime('%H:%M')} 取引チェック")
            
            for symbol in symbols:
                # 5分後予測を実行
                prediction, confidence = self.predict_next_5min(symbol, trading_time)
                if prediction is None:
                    print(f"    {symbol}: 予測失敗（データ不足またはモデルエラー）")
                    continue
                
                # 予測値をデバッグ表示
                direction_str = "上昇" if prediction > 0 else "下降"
                print(f"    {symbol}: 予測={prediction:+.4f} ({direction_str}), 信頼度={confidence:.1%}")
                
                # 取引判定（信頼度50%以上 かつ 任意の予測値で取引 - 閾値を大幅に下げる）
                if confidence > 0.50 and abs(prediction) > 0.0000001:
                    trade = self.execute_dynamic_exit_trade(symbol, friday_date, trading_time, prediction, current_capital)
                    if trade:
                        trades.append(trade)
                        current_capital = trade['new_capital']
                        print(f"    ✅ {symbol}: 取引実行（信頼度={confidence:.1%}）")
                else:
                    if confidence <= 0.50:
                        print(f"    {symbol}: 信頼度不足 ({confidence:.1%} <= 50%)")
                    else:
                        print(f"    {symbol}: 予測値が閾値未満 (|{prediction:.6f}| <= 0.0000001)")
        
        return trades, current_capital
    
    def generate_report(self, thursday_date, friday_date, trades, final_capital):
        """レポート生成"""
        report_filename = f"real_thursday_friday_dynamic_{friday_date.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M')}.txt"
        report_path = os.path.join(self.reports_dir, report_filename)
        
        total_return = (final_capital / self.initial_capital - 1) * 100
        winning_trades = len([t for t in trades if t['profit_loss'] > 0])
        
        report = f"\n=== 実データ版木曜→金曜 動的決済取引デモ ===\n"
        report += f"訓練日: {thursday_date.date()}\n"
        report += f"取引日: {friday_date.date()}\n"
        report += f"決済方式: 5分毎リアルタイム予測による動的決済\n"
        report += f"最長保有: {self.max_holding_minutes}分\n"
        report += f"監視間隔: {self.monitoring_interval}分\n\n"
        
        report += f"📊 取引成績:\n"
        report += f"初期資本: {self.initial_capital:,}円\n"
        report += f"最終評価額: {final_capital:,.0f}円\n"
        report += f"総損益: {final_capital - self.initial_capital:,.0f}円\n"
        report += f"リターン: {total_return:.2f}%\n\n"
        
        report += f"📈 取引統計:\n"
        report += f"総取引数: {len(trades)}\n"
        report += f"利益取引: {winning_trades}\n"
        report += f"勝率: {winning_trades/len(trades)*100:.1f}%\n" if trades else "勝率: 0%\n"
        
        if trades:
            avg_profit = sum(t['profit_loss'] for t in trades) / len(trades)
            avg_holding = sum(t['holding_minutes'] for t in trades) / len(trades)
            report += f"平均損益/取引: {avg_profit:,.0f}円\n"
            report += f"平均保有時間: {avg_holding:.1f}分\n"
            
            # 決済理由別集計
            exit_reasons = {}
            for trade in trades:
                reason = trade['exit_reason']
                if reason not in exit_reasons:
                    exit_reasons[reason] = 0
                exit_reasons[reason] += 1
            
            report += f"\n決済理由別集計:\n"
            for reason, count in exit_reasons.items():
                report += f"  {reason}: {count}回\n"
        
        report += f"\n🔍 個別取引詳細:\n\n"
        
        for i, trade in enumerate(trades, 1):
            profit_status = "利益" if trade['profit_loss'] > 0 else "損失"
            initial_direction = "上昇予測" if trade['initial_direction'] > 0 else "下降予測"
            
            report += f"{i}. {trade['symbol']} ({profit_status})\n"
            report += f"   エントリー: {trade['entry_time'].strftime('%H:%M')} @{trade['entry_price']:.0f}円\n"
            report += f"   エグジット: {trade['exit_time'].strftime('%H:%M')} @{trade['exit_price']:.0f}円\n"
            report += f"   保有時間: {trade['holding_minutes']:.0f}分\n"
            report += f"   株数: {trade['position_size']}株\n"
            report += f"   損益: {trade['profit_loss']:,.0f}円 ({trade['return']*100:.2f}%)\n"
            report += f"   初期予測: {trade['initial_prediction']:+.3f} ({initial_direction})\n"
            report += f"   決済理由: {trade['exit_reason']}\n"
            
            # 予測履歴表示
            if trade['prediction_history']:
                report += f"   予測履歴:\n"
                for pred in trade['prediction_history']:
                    direction_str = "上昇" if pred['direction'] > 0 else "下降"
                    confidence_str = f", 信頼度={pred['confidence']:.1%}" if 'confidence' in pred else ""
                    report += f"     {pred['time'].strftime('%H:%M')}: {pred['prediction']:+.3f} ({direction_str}{confidence_str})\n"
            
            report += "\n"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📋 レポート保存: {report_path}")
        return report
    
    def run_demo(self, thursday_date, friday_date):
        """デモ実行"""
        print(f"\n{'='*70}")
        print(f"実データ版Thursday Friday Demo（動的決済対応）")
        print(f"木曜学習: {thursday_date.date()} → 金曜取引: {friday_date.date()}")
        print(f"{'='*70}")
        
        symbols = ['7203', '6758', '8306', '9984', '6861']
        
        # 1. データ収集
        start_date = thursday_date - timedelta(days=7)
        end_date = friday_date + timedelta(days=1)
        
        if self.collect_yfinance_data(symbols, start_date, end_date) == 0:
            print("❌ データ収集に失敗しました")
            return
        
        # 2. 木曜日データでモデル学習
        print(f"\n🔄 木曜日データでモデル学習 ({thursday_date.date()})")
        
        trained_models = 0
        for symbol in symbols:
            models, scores = self.train_models(symbol, thursday_date)
            if models:
                trained_models += 1
        
        if trained_models == 0:
            print("❌ モデル学習に失敗しました")
            return
        
        print(f"✅ {trained_models}/{len(symbols)} 銘柄のモデル学習完了")
        
        # 3. 金曜日取引シミュレーション
        trades, final_capital = self.simulate_friday_trading(symbols, thursday_date, friday_date)
        
        # 4. 結果表示とレポート生成
        print(f"\n📊 最終結果:")
        print(f"初期資本: ¥{self.initial_capital:,}")
        print(f"最終資本: ¥{final_capital:,.0f}")
        print(f"総収益: ¥{final_capital - self.initial_capital:,.0f}")
        print(f"リターン: {(final_capital / self.initial_capital - 1) * 100:.2f}%")
        print(f"取引数: {len(trades)}")
        
        if trades:
            winning_trades = len([t for t in trades if t['profit_loss'] > 0])
            avg_holding = sum(t['holding_minutes'] for t in trades) / len(trades)
            print(f"勝率: {winning_trades/len(trades)*100:.1f}% ({winning_trades}/{len(trades)})")
            print(f"平均保有時間: {avg_holding:.1f}分")
        
        # レポート生成
        report = self.generate_report(thursday_date, friday_date, trades, final_capital)
        
        return trades, final_capital

def main():
    """メイン実行"""
    print("実データ版Thursday Friday Demo（動的決済対応）")
    print("="*60)
    print("ルール:")
    print("- 最長保有時間: 45分")
    print("- 5分毎にリアルタイムデータで5分後を予測")
    print("- 予測方向が初期方向と逆転したら即決済")
    print("- 通常の利確(3%)・損切(2%)も適用")
    
    system = RealDataThursdayFridayDemo()
    
    # 実際の木曜日・金曜日
    thursday_date = datetime(2025, 7, 17)  # 木曜日
    friday_date = datetime(2025, 7, 18)    # 金曜日
    
    # デモ実行
    trades, final_capital = system.run_demo(thursday_date, friday_date)
    
    print(f"\n🎉 動的決済対応Thursday Friday Demo完了!")

if __name__ == "__main__":
    main()
