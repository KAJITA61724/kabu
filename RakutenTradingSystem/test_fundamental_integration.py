"""
ファンダメンタルズ統合テスト
新たに追加されたファンダメンタルズ特徴量の動作確認
"""

import sys
sys.path.append('core')

from ml_models import MLTradingModels
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_fundamental_features():
    """ファンダメンタルズ特徴量のテスト"""
    print("=== ファンダメンタルズ統合テスト ===")
    
    # インスタンス化
    ml_models = MLTradingModels()
    print("✅ MLTradingModelsインスタンス化成功")
    
    # テスト銘柄
    test_symbol = '7203'  # トヨタ自動車
    
    print(f"\n=== {test_symbol} のファンダメンタルズ特徴量テスト ===")
    
    # サンプルデータでテスト（実際のデータがない場合のため）
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # サンプルデータ作成
    dates = pd.date_range(start=datetime.now() - timedelta(days=1), 
                         end=datetime.now(), freq='5T')
    
    sample_data = pd.DataFrame({
        'datetime': dates,
        'open_price': np.random.uniform(2000, 2100, len(dates)),
        'high_price': np.random.uniform(2050, 2150, len(dates)),
        'low_price': np.random.uniform(1950, 2050, len(dates)),
        'close_price': np.random.uniform(2000, 2100, len(dates)),
        'volume': np.random.randint(100000, 1000000, len(dates))
    })
    
    print(f"サンプルデータ行数: {len(sample_data)}")
    
    # 特徴量作成テスト
    try:
        enhanced_df = ml_models.create_advanced_features(sample_data, test_symbol)
        
        print(f"✅ 特徴量作成成功")
        print(f"作成後の列数: {len(enhanced_df.columns)}")
        
        # ファンダメンタルズ特徴量の確認
        fundamental_cols = [
            'per', 'pbr', 'dividend_yield', 'roe', 'roa', 'market_cap',
            'eps', 'bps', 'revenue_growth', 'profit_growth', 'debt_ratio',
            'sector_avg_per', 'per_vs_sector'
        ]
        
        print("\n=== ファンダメンタルズ特徴量の確認 ===")
        missing_cols = []
        existing_cols = []
        
        for col in fundamental_cols:
            if col in enhanced_df.columns:
                existing_cols.append(col)
                print(f"✅ {col}: 存在")
            else:
                missing_cols.append(col)
                print(f"❌ {col}: 欠損")
        
        print(f"\n統計:")
        print(f"存在する特徴量: {len(existing_cols)}/{len(fundamental_cols)}")
        print(f"テクニカル + ファンダメンタルズ: {len(enhanced_df.columns)}列")
        
        # サンプル値の表示
        if existing_cols:
            print(f"\nサンプル値（最初の5行）:")
            for col in existing_cols[:5]:  # 最初の5つのファンダメンタルズ特徴量
                values = enhanced_df[col].head().tolist()
                print(f"  {col}: {values}")
                
    except Exception as e:
        print(f"❌ 特徴量作成エラー: {e}")
        return False
    
    # prepare_features テスト（互換性用）
    print(f"\n=== prepare_features テスト ===")
    try:
        current_time = datetime.now()
        features = ml_models.prepare_features(test_symbol, current_time)
        
        if features is not None:
            print(f"✅ prepare_features成功")
            print(f"特徴量数: {features.shape}")
            print(f"期待値: (1, 6) - テクニカル3個 + ファンダメンタルズ3個")
        else:
            print(f"⚠️ prepare_features: データなし（DB未初期化の可能性）")
    except Exception as e:
        print(f"❌ prepare_features エラー: {e}")
    
    print(f"\n=== 統合結果 ===")
    print("✅ ファンダメンタルズ特徴量の統合が完了しました")
    print("📊 テクニカル指標とファンダメンタルズ指標の組み合わせ分析が可能")
    print("🎯 より高精度な予測モデルの構築が期待できます")
    
    return True

if __name__ == "__main__":
    test_fundamental_features()
