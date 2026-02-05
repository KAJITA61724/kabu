"""
高精度リークフリーモデル - シンプル予測デモ
既存の高精度モデルを使用した最新データでの予測
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

# 高精度モデルをインポート
from high_precision_ml_models import HighPrecisionLeakFreeModels

def run_simple_prediction_demo():
    """シンプルな予測デモ"""
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("🚀 高精度リークフリーモデル - 予測デモ")
    print("=" * 60)
    
    # モデル初期化
    model = HighPrecisionLeakFreeModels()
    
    # テスト銘柄
    symbols = ['7203', '6758', '8306']
    
    # まず既存データを収集
    print("\n📊 最新データ収集中...")
    if model.collect_extended_market_data(symbols, days=5):
        print("✅ 最新データ収集完了")
    else:
        print("⚠️ 最新データ収集失敗、既存データを使用")
    
    # 各銘柄で予測実行
    results = {}
    
    for symbol in symbols:
        print(f"\n🔍 {symbol} 分析開始")
        print("-" * 40)
        
        try:
            # データ準備
            data_result = model.prepare_high_precision_data(symbol)
            if data_result[0] is None:
                print(f"❌ {symbol}: データ準備失敗")
                continue
            
            X_train, X_test, y_price_train, y_price_test, y_direction_train, y_direction_test = data_result
            
            print(f"📈 データサイズ: 訓練{len(X_train)}件, テスト{len(X_test)}件")
            print(f"🔧 特徴量数: {len(model.feature_columns)}個")
            
            # モデル訓練
            print("🤖 モデル訓練中...")
            models = model.train_ensemble_models(X_train, y_price_train, y_direction_train)
            
            # 評価
            print("📊 モデル評価中...")
            evaluation = model.evaluate_ensemble_models(models, X_test, y_price_test, y_direction_test)
            
            # 最良モデル選択
            best_price = min(evaluation['price_models'].items(), key=lambda x: x[1]['mae'])
            best_direction = max(evaluation['direction_models'].items(), key=lambda x: x[1]['accuracy'])
            
            # 最新データで予測
            print("\n🎯 最新データ予測:")
            
            # 最新の特徴量を作成
            latest_features = X_test.iloc[-1:].values
            
            # 価格予測
            price_model_data = models['price_models'][best_price[0]]
            price_scaler = price_model_data['scaler']
            price_selector = price_model_data['selector']
            price_model = price_model_data['model']
            
            latest_features_selected = price_selector.transform(X_test.iloc[-1:])
            latest_features_scaled = price_scaler.transform(latest_features_selected)
            price_prediction = price_model.predict(latest_features_scaled)[0]
            
            # 方向予測
            direction_model_data = models['direction_models'][best_direction[0]]
            direction_scaler = direction_model_data['scaler']
            direction_selector = direction_model_data['selector']
            direction_model = direction_model_data['model']
            
            latest_features_dir_selected = direction_selector.transform(X_test.iloc[-1:])
            latest_features_dir_scaled = direction_scaler.transform(latest_features_dir_selected)
            direction_prediction = direction_model.predict(latest_features_dir_scaled)[0]
            direction_probability = direction_model.predict_proba(latest_features_dir_scaled)[0]
            
            # 実際の価格情報を取得
            conn = sqlite3.connect(model.db_path)
            query = '''
                SELECT datetime, close_price
                FROM extended_market_data
                WHERE symbol = ? AND timeframe = '5m'
                ORDER BY datetime DESC
                LIMIT 1
            '''
            latest_data = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if not latest_data.empty:
                latest_price = latest_data['close_price'].iloc[0]
                latest_time = latest_data['datetime'].iloc[0]
                predicted_price = latest_price * (1 + price_prediction)
            else:
                latest_price = "不明"
                latest_time = "不明"
                predicted_price = "算出不可"
            
            # 結果表示
            print(f"📅 最新時刻: {latest_time}")
            print(f"💰 現在価格: {latest_price}")
            print(f"🎯 予測価格変化率: {price_prediction:.4f} ({price_prediction*100:.2f}%)")
            if predicted_price != "算出不可":
                print(f"💡 予測価格: {predicted_price:.2f}")
            
            direction_text = "上昇" if direction_prediction == 1 else "下降"
            confidence = max(direction_probability) * 100
            
            print(f"📈 予測方向: {direction_text}")
            print(f"🎲 信頼度: {confidence:.1f}%")
            
            # 投資判断
            if confidence >= 60:
                if direction_prediction == 1:
                    recommendation = "🟢 買い推奨"
                else:
                    recommendation = "🔴 売り推奨"
            elif confidence >= 55:
                recommendation = "🟡 様子見推奨"
            else:
                recommendation = "⚪ 取引非推奨"
            
            print(f"💼 投資判断: {recommendation}")
            
            # モデル性能表示
            print(f"\n📊 モデル性能:")
            print(f"  最良価格予測: {best_price[0]} (MAE: {best_price[1]['mae']:.6f})")
            print(f"  最良方向予測: {best_direction[0]} (精度: {best_direction[1]['accuracy']:.1%})")
            
            results[symbol] = {
                'latest_price': latest_price,
                'predicted_change': price_prediction,
                'predicted_direction': direction_text,
                'confidence': confidence,
                'recommendation': recommendation,
                'model_performance': {
                    'price_mae': best_price[1]['mae'],
                    'direction_accuracy': best_direction[1]['accuracy']
                }
            }
            
        except Exception as e:
            print(f"❌ {symbol} 予測エラー: {e}")
            logger.error(f"予測エラー ({symbol}): {e}")
    
    # 総合結果表示
    print("\n" + "=" * 60)
    print("📋 総合予測結果サマリー")
    print("=" * 60)
    
    for symbol, result in results.items():
        print(f"\n🏢 {symbol}:")
        print(f"  現在価格: {result['latest_price']}")
        print(f"  予測方向: {result['predicted_direction']} (信頼度: {result['confidence']:.1f}%)")
        print(f"  投資判断: {result['recommendation']}")
        print(f"  モデル精度: {result['model_performance']['direction_accuracy']:.1%}")
    
    # パフォーマンス統計
    if results:
        avg_accuracy = np.mean([r['model_performance']['direction_accuracy'] for r in results.values()])
        avg_confidence = np.mean([r['confidence'] for r in results.values()])
        
        print(f"\n📈 全体統計:")
        print(f"  平均予測精度: {avg_accuracy:.1%}")
        print(f"  平均信頼度: {avg_confidence:.1f}%")
        
        # 投資推奨統計
        recommendations = [r['recommendation'] for r in results.values()]
        buy_count = sum(1 for r in recommendations if "買い" in r)
        sell_count = sum(1 for r in recommendations if "売り" in r)
        hold_count = len(recommendations) - buy_count - sell_count
        
        print(f"  買い推奨: {buy_count}銘柄")
        print(f"  売り推奨: {sell_count}銘柄") 
        print(f"  様子見: {hold_count}銘柄")
    
    print("\n✅ 予測デモ完了")
    return results

if __name__ == "__main__":
    results = run_simple_prediction_demo()
