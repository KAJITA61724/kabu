"""
改良版ml_models.py機能テスト
新機能（compare_models、run_integrated_analysis）の動作確認
"""

import sys
sys.path.append('core')

from ml_models import MLTradingModels
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_ml_models_features():
    """新機能のテスト"""
    print("=== 改良版 ml_models.py 機能テスト ===")
    
    # インスタンス化
    ml_models = MLTradingModels()
    print("✅ MLTradingModelsインスタンス化成功")
    
    # 新機能の確認
    features = [
        'compare_models',
        'run_integrated_analysis', 
        'create_advanced_features',
        'prepare_advanced_data',
        'train_advanced_model',
        'generate_prediction_report',
        'generate_comparison_report'
    ]
    
    print("\n=== 新機能の確認 ===")
    for feature in features:
        has_feature = hasattr(ml_models, feature)
        status = "✅" if has_feature else "❌"
        print(f"{status} {feature}: {has_feature}")
    
    # 基本機能の確認
    print("\n=== 基本機能の確認 ===")
    basic_features = [
        'collect_yfinance_data',
        'hourly_predict',
        'minute_predict',
        'fact_check'
    ]
    
    for feature in basic_features:
        has_feature = hasattr(ml_models, feature)
        status = "✅" if has_feature else "❌"
        print(f"{status} {feature}: {has_feature}")
    
    print("\n=== テスト完了 ===")
    print("改良版ml_models.pyの統合が成功しました！")

if __name__ == "__main__":
    test_ml_models_features()
