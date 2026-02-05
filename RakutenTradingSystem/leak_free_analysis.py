#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リークなし日次モデル更新システムの分析と改善案
"""
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

class LeakFreeAnalyzer:
    def __init__(self, models_dir='leak_free_models', reports_dir='leak_free_reports'):
        self.models_dir = models_dir
        self.reports_dir = reports_dir
    
    def analyze_model_performance(self):
        """モデルパフォーマンス分析"""
        print("📊 リークなし日次モデル更新システム 分析結果")
        print("="*60)
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        
        # 銘柄別・日付別分析
        performance_data = []
        
        for model_file in model_files:
            symbol = model_file.split('_')[0]
            date_str = model_file.split('_')[2].replace('.pkl', '')
            
            try:
                with open(os.path.join(self.models_dir, model_file), 'rb') as f:
                    model_data = pickle.load(f)
                
                scores = model_data['scores']
                best_score = max(scores.values())
                best_model = max(scores.items(), key=lambda x: x[1])[0]
                data_size = model_data['data_size']
                
                performance_data.append({
                    'symbol': symbol,
                    'date': date_str,
                    'best_score': best_score,
                    'best_model': best_model,
                    'data_size': data_size,
                    'rf_score': scores.get('RandomForest', 0),
                    'ridge_score': scores.get('Ridge', 0),
                    'lgb_score': scores.get('LightGBM', 0)
                })
                
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
        
        df = pd.DataFrame(performance_data)
        
        print("\\n🎯 モデル性能サマリー:")
        print(f"総モデル数: {len(df)}")
        print(f"正のR²スコア: {len(df[df['best_score'] > 0])}/{len(df)} ({len(df[df['best_score'] > 0])/len(df)*100:.1f}%)")
        
        print("\\n📈 銘柄別ベストスコア:")
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            avg_score = symbol_data['best_score'].mean()
            best_day = symbol_data.loc[symbol_data['best_score'].idxmax(), 'date']
            best_score = symbol_data['best_score'].max()
            print(f"  {symbol}: 平均 {avg_score:.3f}, 最高 {best_score:.3f} ({best_day})")
        
        print("\\n📊 モデル別平均性能:")
        print(f"  RandomForest: {df['rf_score'].mean():.3f}")
        print(f"  Ridge:        {df['ridge_score'].mean():.3f}")
        print(f"  LightGBM:     {df['lgb_score'].mean():.3f}")
        
        print("\\n📅 日別モデル改善:")
        daily_avg = df.groupby('date')['best_score'].mean()
        for date, score in daily_avg.items():
            print(f"  {date}: {score:.3f}")
        
        return df
    
    def suggest_improvements(self):
        """改善提案"""
        print("\\n🔧 システム改善提案:")
        print("="*40)
        
        suggestions = [
            "1. 特徴量エンジニアリング強化:",
            "   - より長期間のトレンド指標追加",
            "   - 他銘柄との相関特徴量",
            "   - マクロ経済指標の組み込み",
            "",
            "2. モデル改善:",
            "   - ハイパーパラメータ最適化",
            "   - アンサンブル重み調整",
            "   - 時系列特化モデル（LSTM等）",
            "",
            "3. 取引戦略調整:",
            "   - 予測閾値の動的調整",
            "   - リスク管理の強化",
            "   - ポジションサイジング最適化",
            "",
            "4. データ期間拡張:",
            "   - より長期間の学習データ使用",
            "   - 外部データソースの活用",
            "",
            "5. リアルタイム対応:",
            "   - モデル更新頻度の調整",
            "   - オンライン学習の導入"
        ]
        
        for suggestion in suggestions:
            print(suggestion)
    
    def performance_summary(self):
        """パフォーマンス総評"""
        print("\\n🏆 システム評価:")
        print("="*30)
        
        evaluation = [
            "✅ 成功した点:",
            "  - リークなしのモデル学習実現",
            "  - yfinance実データでの動作確認",
            "  - 5分足統一での一貫性",
            "  - 日次モデル更新サイクル構築",
            "",
            "⚠️  課題と対策:",
            "  - 負のR²スコアが多い → 特徴量改善が必要",
            "  - 取引頻度が低い → 予測閾値調整",
            "  - 短期データ制限 → 外部データ活用検討",
            "",
            "🎯 次のステップ:",
            "  1. 特徴量設計の見直し",
            "  2. より長期間でのバックテスト",
            "  3. リアルタイム取引への適用準備"
        ]
        
        for item in evaluation:
            print(item)

def main():
    analyzer = LeakFreeAnalyzer()
    
    # モデル性能分析
    df = analyzer.analyze_model_performance()
    
    # 改善提案
    analyzer.suggest_improvements()
    
    # 総評
    analyzer.performance_summary()
    
    print("\\n" + "="*60)
    print("分析完了: リークなし日次モデル更新システム")

if __name__ == "__main__":
    main()
