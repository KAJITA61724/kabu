#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前々日データをベースにしたモデル精度評価スクリプト
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# scikit-learnとLightGBMの利用可能性をチェック
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    sklearn_available = True
except ImportError:
    sklearn_available = False

try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    lightgbm_available = False

def load_data_for_evaluation():
    """前々日のデータを取得"""
    try:
        conn = sqlite3.connect('trading_data.db')
        
        # 前々日のデータを取得 (2025-07-16)
        target_date = '2025-07-16'
        
        query = """
        SELECT symbol, datetime, open_price, high_price, low_price, close_price, volume
        FROM chart_data 
        WHERE datetime LIKE ? 
        ORDER BY symbol, datetime
        """
        
        df = pd.read_sql_query(query, conn, params=(f'{target_date}%',))
        conn.close()
        
        if len(df) == 0:
            print(f"❌ {target_date}のデータが見つかりません")
            return None
        
        print(f"✅ {target_date}のデータを読み込みました: {len(df)}件")
        return df
    
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None

def create_features(df):
    """特徴量を作成"""
    # 時系列データを処理
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['symbol', 'datetime'])
    
    feature_data = []
    
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].copy()
        
        # 価格関連の特徴量
        symbol_data['price_change'] = symbol_data['close_price'].pct_change()
        symbol_data['high_low_ratio'] = symbol_data['high_price'] / symbol_data['low_price']
        symbol_data['open_close_ratio'] = symbol_data['open_price'] / symbol_data['close_price']
        symbol_data['volume_price_ratio'] = symbol_data['volume'] / symbol_data['close_price']
        
        # 移動平均
        symbol_data['sma_5'] = symbol_data['close_price'].rolling(window=5).mean()
        symbol_data['sma_10'] = symbol_data['close_price'].rolling(window=10).mean()
        symbol_data['sma_ratio'] = symbol_data['close_price'] / symbol_data['sma_5']
        
        # ボラティリティ
        symbol_data['volatility_5'] = symbol_data['close_price'].rolling(window=5).std()
        symbol_data['volatility_10'] = symbol_data['close_price'].rolling(window=10).std()
        
        # 出来高関連
        symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume'].rolling(window=5).mean()
        
        # RSI
        delta = symbol_data['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        symbol_data['rsi'] = 100 - (100 / (1 + rs))
        
        # ラグ特徴量
        for lag in [1, 2, 3, 5]:
            symbol_data[f'close_lag_{lag}'] = symbol_data['close_price'].shift(lag)
            symbol_data[f'volume_lag_{lag}'] = symbol_data['volume'].shift(lag)
            symbol_data[f'change_lag_{lag}'] = symbol_data['price_change'].shift(lag)
        
        # 時刻関連の特徴量
        symbol_data['hour'] = symbol_data['datetime'].dt.hour
        symbol_data['minute'] = symbol_data['datetime'].dt.minute
        symbol_data['time_of_day'] = symbol_data['hour'] * 60 + symbol_data['minute']
        
        # 目標変数（次の時刻の価格変化）
        symbol_data['target'] = symbol_data['close_price'].shift(-1)
        
        feature_data.append(symbol_data)
    
    return pd.concat(feature_data, ignore_index=True)

def evaluate_models(df):
    """モデルの精度評価"""
    feature_cols = [
        'price_change', 'high_low_ratio', 'open_close_ratio', 'volume_price_ratio',
        'sma_ratio', 'volatility_5', 'volatility_10', 'volume_ratio', 'rsi',
        'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
        'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
        'change_lag_1', 'change_lag_2', 'change_lag_3', 'change_lag_5',
        'hour', 'minute', 'time_of_day'
    ]
    
    # 欠損値を削除
    df = df.dropna()
    
    if len(df) < 50:
        print(f"❌ 有効なデータが不足しています ({len(df)}件)")
        return None
    
    # 特徴量と目標変数を分離
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    # NaNや無限大の値を処理
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    # 訓練・テストデータの分割 (70:30)
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"✅ 特徴量準備完了: {len(feature_cols)}個の特徴量, {len(X_train)}件の訓練データ, {len(X_test)}件のテストデータ")
    
    # モデルの定義
    models = {}
    
    if sklearn_available:
        models['RandomForest'] = RandomForestRegressor(n_estimators=100, random_state=42)
        models['LinearRegression'] = LinearRegression()
    
    if lightgbm_available:
        models['LightGBM'] = lgb.LGBMRegressor(random_state=42, verbose=-1)
    
    # 単純移動平均ベースライン
    models['SimpleMovingAverage'] = None
    
    if not models:
        print("❌ 利用可能なモデルがありません")
        return None
    
    # 各モデルの評価
    results = {}
    
    for name, model in models.items():
        try:
            if model is None:
                # 単純移動平均の実装
                pred = [np.mean(y_train[-5:]) for _ in range(len(y_test))]
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            
            # 精度指標の計算
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            
            # 方向性の精度（価格上昇・下降の予測精度）
            actual_direction = np.sign(y_test - X_test['close_lag_1'])
            pred_direction = np.sign(pred - X_test['close_lag_1'])
            direction_accuracy = np.mean(actual_direction == pred_direction)
            
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'direction_accuracy': direction_accuracy
            }
            
            print(f"\n{name}:")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  方向性精度: {direction_accuracy:.4f}")
            
        except Exception as e:
            print(f"{name} エラー: {e}")
            results[name] = {'error': str(e)}
    
    return results

def main():
    """メイン処理"""
    print("=== 前々日データによるモデル精度評価 ===")
    print("評価対象日: 2025-07-16")
    print(f"LightGBM利用可能: {lightgbm_available}")
    print(f"scikit-learn利用可能: {sklearn_available}")
    
    # データの読み込み
    df = load_data_for_evaluation()
    if df is None:
        return
    
    # 特徴量の作成
    print("\n=== 特徴量作成 ===")
    df_features = create_features(df)
    
    # モデル評価
    print("\n=== モデル精度評価 ===")
    results = evaluate_models(df_features)
    
    if results:
        print("\n=== 評価結果サマリー ===")
        print("| モデル | RMSE | MAE | R² | 方向性精度 |")
        print("|-------|------|-----|-----|-----------|")
        for name, metrics in results.items():
            if 'error' not in metrics:
                print(f"| {name} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | {metrics['r2']:.4f} | {metrics['direction_accuracy']:.4f} |")
            else:
                print(f"| {name} | エラー | エラー | エラー | エラー |")
        
        # 最適モデルの選択
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model = min(valid_results.items(), key=lambda x: x[1]['rmse'])
            print(f"\n🏆 最適モデル (RMSE基準): {best_model[0]} (RMSE: {best_model[1]['rmse']:.4f})")
            
            best_direction = max(valid_results.items(), key=lambda x: x[1]['direction_accuracy'])
            print(f"🎯 方向性予測最優秀: {best_direction[0]} (精度: {best_direction[1]['direction_accuracy']:.4f})")

if __name__ == "__main__":
    main()
