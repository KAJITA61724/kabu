"""
ファンダメンタルズ分析デモ
実際のファンダメンタルズデータを使用したモデル比較
"""

import pandas as pd
import numpy as np
from fundamental_data_collector import FundamentalDataCollector
from multi_model_comparison import ModelConfig, ModelPerformance
import logging
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class FundamentalAnalysisDemo:
    """ファンダメンタルズ分析デモシステム"""
    
    def __init__(self):
        self.fundamental_collector = FundamentalDataCollector()
        self.logger = logging.getLogger(__name__)
        self.results_dir = Path("reports/fundamental_demo")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_fundamental_features(self, symbols: list) -> pd.DataFrame:
        """ファンダメンタルズ特徴量生成"""
        try:
            # ファンダメンタルズデータ収集
            fundamental_data = self.fundamental_collector.collect_fundamental_data(symbols)
            
            # 業界別平均計算
            sector_averages = self.fundamental_collector.calculate_sector_averages()
            
            # データフレーム作成
            data = []
            for symbol, fund_data in fundamental_data.items():
                # 相対評価指標
                relative_metrics = self.fundamental_collector.get_relative_valuation(symbol)
                
                # 投資魅力度スコア計算
                investment_score = self.calculate_investment_score(fund_data, relative_metrics)
                
                # 予測ターゲット（価格上昇確率）
                target = 1 if investment_score > 0.6 else 0
                
                row = {
                    'symbol': symbol,
                    'per': fund_data.per,
                    'pbr': fund_data.pbr,
                    'psr': fund_data.psr,
                    'dividend_yield': fund_data.dividend_yield,
                    'roe': fund_data.roe,
                    'roa': fund_data.roa,
                    'debt_ratio': fund_data.debt_ratio,
                    'current_ratio': fund_data.current_ratio,
                    'revenue_growth': fund_data.revenue_growth,
                    'profit_growth': fund_data.profit_growth,
                    'operating_margin': fund_data.operating_margin,
                    'market_cap': fund_data.market_cap,
                    'beta': fund_data.beta,
                    'per_ratio': relative_metrics.get('per_ratio', 1.0),
                    'pbr_ratio': relative_metrics.get('pbr_ratio', 1.0),
                    'roe_ratio': relative_metrics.get('roe_ratio', 1.0),
                    'debt_ratio_ratio': relative_metrics.get('debt_ratio_ratio', 1.0),
                    'investment_score': investment_score,
                    'target': target
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            self.logger.error(f"ファンダメンタルズ特徴量生成エラー: {e}")
            return pd.DataFrame()
    
    def calculate_investment_score(self, fund_data, relative_metrics) -> float:
        """投資魅力度スコア計算"""
        try:
            score = 0.0
            
            # PER評価（低いほど良い）
            if fund_data.per > 0:
                per_score = max(0, 1 - fund_data.per / 30)  # PER30以下なら高スコア
                score += per_score * 0.2
            
            # PBR評価（低いほど良い）
            if fund_data.pbr > 0:
                pbr_score = max(0, 1 - fund_data.pbr / 3)  # PBR3以下なら高スコア
                score += pbr_score * 0.15
            
            # ROE評価（高いほど良い）
            if fund_data.roe > 0:
                roe_score = min(1, fund_data.roe / 0.15)  # ROE15%以上なら満点
                score += roe_score * 0.25
            
            # 成長性評価
            if fund_data.revenue_growth > 0:
                growth_score = min(1, fund_data.revenue_growth / 0.1)  # 10%成長で満点
                score += growth_score * 0.15
            
            # 財務健全性評価
            if fund_data.current_ratio > 1:
                health_score = min(1, fund_data.current_ratio / 2)  # 流動比率2.0で満点
                score += health_score * 0.1
            
            # 相対評価
            if relative_metrics.get('per_ratio', 1) < 0.8:  # 業界平均より20%安い
                score += 0.1
            
            if relative_metrics.get('roe_ratio', 1) > 1.2:  # 業界平均より20%高い
                score += 0.05
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"投資スコア計算エラー: {e}")
            return 0.5
    
    def train_fundamental_models(self, df: pd.DataFrame) -> dict:
        """ファンダメンタルズモデル学習"""
        try:
            # 特徴量準備
            feature_columns = [
                'per', 'pbr', 'psr', 'dividend_yield', 'roe', 'roa', 
                'debt_ratio', 'current_ratio', 'revenue_growth', 'profit_growth',
                'operating_margin', 'beta', 'per_ratio', 'pbr_ratio', 'roe_ratio'
            ]
            
            X = df[feature_columns].fillna(0)
            y = df['target']
            
            # 学習・テストデータ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # 正規化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # モデル学習
            models = {
                'バリュー投資モデル': RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                ),
                '成長株モデル': RandomForestClassifier(
                    n_estimators=150, max_depth=8, random_state=42
                ),
                '複合モデル': RandomForestClassifier(
                    n_estimators=200, max_depth=10, random_state=42
                )
            }
            
            results = {}
            
            for name, model in models.items():
                # 学習
                model.fit(X_train_scaled, y_train)
                
                # 予測
                predictions = model.predict(X_test_scaled)
                probabilities = model.predict_proba(X_test_scaled)[:, 1]
                
                # 評価
                accuracy = accuracy_score(y_test, predictions)
                
                # 特徴量重要度
                feature_importance = dict(zip(feature_columns, model.feature_importances_))
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'feature_importance': feature_importance,
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'y_test': y_test
                }
                
                print(f"{name}: 精度 {accuracy:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"モデル学習エラー: {e}")
            return {}
    
    def create_analysis_report(self, df: pd.DataFrame, model_results: dict):
        """分析レポート作成"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. ファンダメンタルズ指標分布
            fundamental_metrics = ['per', 'pbr', 'roe', 'debt_ratio']
            for i, metric in enumerate(fundamental_metrics):
                ax = axes[i//2, i%2]
                data = df[metric].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=15, alpha=0.7, edgecolor='black')
                    ax.set_title(f'{metric.upper()} 分布')
                    ax.set_xlabel(metric.upper())
                    ax.set_ylabel('銘柄数')
            
            plt.tight_layout()
            report_file = self.results_dir / f"fundamental_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(report_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 投資スコア vs 実際の指標
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 投資スコア分布
            axes[0].hist(df['investment_score'], bins=20, alpha=0.7, edgecolor='black')
            axes[0].set_title('投資魅力度スコア分布')
            axes[0].set_xlabel('投資スコア')
            axes[0].set_ylabel('銘柄数')
            
            # スコア vs ROE
            axes[1].scatter(df['investment_score'], df['roe'], alpha=0.6)
            axes[1].set_title('投資スコア vs ROE')
            axes[1].set_xlabel('投資スコア')
            axes[1].set_ylabel('ROE')
            
            plt.tight_layout()
            score_file = self.results_dir / f"investment_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(score_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 特徴量重要度比較
            if model_results:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                importance_data = []
                for model_name, result in model_results.items():
                    for feature, importance in result['feature_importance'].items():
                        importance_data.append({
                            'model': model_name,
                            'feature': feature,
                            'importance': importance
                        })
                
                importance_df = pd.DataFrame(importance_data)
                importance_pivot = importance_df.pivot(index='feature', columns='model', values='importance')
                
                importance_pivot.plot(kind='bar', ax=ax)
                ax.set_title('特徴量重要度比較')
                ax.set_xlabel('特徴量')
                ax.set_ylabel('重要度')
                ax.legend(title='モデル')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                importance_file = self.results_dir / f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(importance_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"分析レポート保存完了: {self.results_dir}")
            
        except Exception as e:
            self.logger.error(f"レポート作成エラー: {e}")
    
    def run_demo(self):
        """デモ実行"""
        try:
            print("=" * 50)
            print("ファンダメンタルズ分析デモ")
            print("=" * 50)
            
            # 対象銘柄
            symbols = ["7203", "9984", "6758", "8306", "6501", "4063", "9432", "8035", "4519", "6861"]
            
            # ファンダメンタルズデータ生成
            print("\\n1. ファンダメンタルズデータ収集・分析...")
            df = self.generate_fundamental_features(symbols)
            
            if df.empty:
                print("データが不足しています。")
                return
            
            print(f"分析対象: {len(df)} 銘柄")
            
            # 基本統計
            print("\\n2. ファンダメンタルズ指標統計:")
            print(df[['per', 'pbr', 'roe', 'debt_ratio', 'investment_score']].describe())
            
            # モデル学習
            print("\\n3. 機械学習モデル学習・評価...")
            model_results = self.train_fundamental_models(df)
            
            # 投資推奨銘柄
            print("\\n4. 投資推奨銘柄（投資スコア上位5銘柄）:")
            top_stocks = df.nlargest(5, 'investment_score')
            for _, stock in top_stocks.iterrows():
                print(f"  {stock['symbol']}: スコア {stock['investment_score']:.3f} "
                      f"(PER: {stock['per']:.1f}, PBR: {stock['pbr']:.1f}, ROE: {stock['roe']:.1%})")
            
            # レポート作成
            print("\\n5. 分析レポート作成中...")
            self.create_analysis_report(df, model_results)
            
            # 結果保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.results_dir / f"fundamental_demo_results_{timestamp}.json"
            
            demo_results = {
                'timestamp': timestamp,
                'total_stocks': len(df),
                'model_accuracy': {name: result['accuracy'] for name, result in model_results.items()},
                'top_recommendations': top_stocks[['symbol', 'investment_score', 'per', 'pbr', 'roe']].to_dict('records'),
                'fundamental_summary': df[['per', 'pbr', 'roe', 'debt_ratio', 'investment_score']].describe().to_dict()
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(demo_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\\n結果保存完了: {results_file}")
            print("=" * 50)
            print("ファンダメンタルズ分析デモ完了")
            print("=" * 50)
            
        except Exception as e:
            self.logger.error(f"デモ実行エラー: {e}")

def main():
    """メイン実行"""
    logging.basicConfig(level=logging.INFO)
    
    demo = FundamentalAnalysisDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
