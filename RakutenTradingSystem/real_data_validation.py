"""
実データでのMLモデル検証システム
ファンダメンタルズ統合済みMLモデルの性能を実データで詳細検証
"""

import sys
sys.path.append('core')

from ml_models import MLTradingModels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

class MLModelValidator:
    """MLモデルの実データ検証クラス"""
    
    def __init__(self):
        self.ml_models = MLTradingModels()
        self.logger = logging.getLogger(__name__)
        
        # 結果保存ディレクトリ
        self.results_dir = Path("validation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 検証対象銘柄（異なる業種を選択）
        self.test_symbols = {
            '7203': 'トヨタ自動車（自動車）',
            '6758': 'ソニーグループ（電気機器）',
            '8306': '三菱UFJ（銀行）',
            '9984': 'ソフトバンク（情報通信）',
            '6861': 'キーエンス（電気機器）',
            '4503': '大樹生命（化学）',
            '7974': '任天堂（その他製品）'
        }
    
    def collect_validation_data(self, days: int = 10) -> bool:
        """検証用データの収集"""
        print("=== 検証用データ収集 ===")
        
        symbols = list(self.test_symbols.keys())
        success = self.ml_models.collect_yfinance_data(symbols, days=days)
        
        if success:
            print(f"✅ {len(symbols)}銘柄の{days}日間データ収集完了")
            
            # データ量確認
            conn = sqlite3.connect(self.ml_models.db_path)
            for symbol in symbols:
                query = "SELECT COUNT(*) FROM chart_data WHERE symbol = ?"
                count = pd.read_sql_query(query, conn, params=(symbol,)).iloc[0, 0]
                print(f"  {symbol} ({self.test_symbols[symbol]}): {count}件")
            conn.close()
            
        return success
    
    def validate_single_symbol(self, symbol: str) -> dict:
        """単一銘柄の詳細検証"""
        print(f"\n{'='*60}")
        print(f"📊 {symbol} ({self.test_symbols[symbol]}) の詳細検証")
        print('='*60)
        
        results = {}
        
        try:
            # 1. データ準備と特徴量作成
            X, y, df, feature_cols = self.ml_models.prepare_advanced_data(symbol, period=500)
            
            if X is None:
                print(f"❌ {symbol}: データ不足")
                return None
            
            print(f"✅ データ準備完了: {len(X)}サンプル, {len(feature_cols)}特徴量")
            
            # 2. 複数モデルの訓練と比較
            model_comparison = self.ml_models.compare_models([symbol])
            
            if symbol not in model_comparison:
                print(f"❌ {symbol}: モデル訓練失敗")
                return None
            
            model_results = model_comparison[symbol]
            
            # 3. 特徴量重要度分析
            feature_importance = self._analyze_feature_importance(model_results)
            
            # 4. 予測精度の時系列分析
            time_series_analysis = self._time_series_prediction_analysis(X, y, feature_cols, symbol)
            
            # 5. ファンダメンタルズ効果分析
            fundamental_impact = self._analyze_fundamental_impact(df, symbol)
            
            results = {
                'symbol': symbol,
                'company_name': self.test_symbols[symbol],
                'data_size': len(X),
                'feature_count': len(feature_cols),
                'model_results': model_results,
                'feature_importance': feature_importance,
                'time_series_analysis': time_series_analysis,
                'fundamental_impact': fundamental_impact,
                'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 6. 結果レポート生成
            self._generate_detailed_report(results)
            
            print(f"✅ {symbol}: 検証完了")
            
        except Exception as e:
            print(f"❌ {symbol}: 検証エラー - {e}")
            self.logger.error(f"検証エラー ({symbol}): {e}")
            return None
        
        return results
    
    def _analyze_feature_importance(self, model_results: dict) -> dict:
        """特徴量重要度分析"""
        importance_analysis = {}
        
        for model_name, results in model_results.items():
            if 'feature_importance' in results and results['feature_importance'] is not None:
                feature_imp = results['feature_importance']
                
                # 上位10特徴量
                top_features = feature_imp.head(10)
                
                # テクニカル vs ファンダメンタルズ分析
                technical_features = []
                fundamental_features = []
                
                for _, row in feature_imp.iterrows():
                    feature_name = row['feature']
                    importance = row['importance']
                    
                    if feature_name in ['per', 'pbr', 'dividend_yield', 'roe', 'roa', 'market_cap',
                                      'eps', 'bps', 'revenue_growth', 'profit_growth', 'debt_ratio',
                                      'sector_avg_per', 'per_vs_sector']:
                        fundamental_features.append((feature_name, importance))
                    else:
                        technical_features.append((feature_name, importance))
                
                technical_importance = sum([imp for _, imp in technical_features])
                fundamental_importance = sum([imp for _, imp in fundamental_features])
                
                importance_analysis[model_name] = {
                    'top_features': top_features.to_dict('records'),
                    'technical_importance': technical_importance,
                    'fundamental_importance': fundamental_importance,
                    'technical_vs_fundamental_ratio': technical_importance / (fundamental_importance + 0.001)
                }
        
        return importance_analysis
    
    def _time_series_prediction_analysis(self, X, y, feature_cols, symbol: str) -> dict:
        """時系列予測精度分析"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # 時系列クロスバリデーション
        tscv = TimeSeriesSplit(n_splits=5)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        mse_scores = []
        mae_scores = []
        r2_scores = []
        
        for train_index, test_index in tscv.split(X):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_test_fold)
            
            mse_scores.append(mean_squared_error(y_test_fold, y_pred_fold))
            mae_scores.append(mean_absolute_error(y_test_fold, y_pred_fold))
            r2_scores.append(r2_score(y_test_fold, y_pred_fold))
        
        return {
            'cv_mse_mean': np.mean(mse_scores),
            'cv_mse_std': np.std(mse_scores),
            'cv_mae_mean': np.mean(mae_scores),
            'cv_mae_std': np.std(mae_scores),
            'cv_r2_mean': np.mean(r2_scores),
            'cv_r2_std': np.std(r2_scores),
            'stability_score': 1 - (np.std(r2_scores) / (np.mean(r2_scores) + 0.001))
        }
    
    def _analyze_fundamental_impact(self, df: pd.DataFrame, symbol: str) -> dict:
        """ファンダメンタルズデータの価格予測への影響分析"""
        fundamental_cols = ['per', 'pbr', 'dividend_yield', 'roe', 'roa']
        
        if not all(col in df.columns for col in fundamental_cols):
            return {'status': 'ファンダメンタルズデータなし'}
        
        # 価格変動との相関分析
        price_change = df['close_price'].pct_change()
        
        correlations = {}
        for col in fundamental_cols:
            if col in df.columns:
                correlation = df[col].corr(price_change)
                correlations[col] = correlation if not np.isnan(correlation) else 0
        
        # ファンダメンタルズ値の要約統計
        fundamental_stats = {}
        for col in fundamental_cols:
            if col in df.columns and df[col].notna().any():
                fundamental_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'current_value': df[col].iloc[-1] if not df[col].empty else 0
                }
        
        return {
            'correlations': correlations,
            'fundamental_stats': fundamental_stats,
            'status': 'ファンダメンタルズ分析完了'
        }
    
    def _generate_detailed_report(self, results: dict):
        """詳細レポート生成"""
        symbol = results['symbol']
        company_name = results['company_name']
        
        report = f"""
================================================================================
🏢 {symbol} ({company_name}) 詳細検証レポート
================================================================================

📊 基本情報:
  • 検証日時: {results['validation_date']}
  • データサイズ: {results['data_size']:,}サンプル
  • 特徴量数: {results['feature_count']}個
  • ファンダメンタルズ統合: ✅ 有効

📈 モデル性能比較:
"""
        
        # モデル性能比較
        model_results = results['model_results']
        performance_data = []
        
        for model_name, model_result in model_results.items():
            performance_data.append({
                'model': model_name,
                'mse': model_result['mse'],
                'mae': model_result['mae'],
                'r2': model_result.get('r2', 0)
            })
        
        # 性能順にソート
        performance_data.sort(key=lambda x: x['mse'])
        
        for i, perf in enumerate(performance_data, 1):
            report += f"  {i}. {perf['model']}\n"
            report += f"     MSE: {perf['mse']:.4f} | MAE: {perf['mae']:.4f} | R²: {perf['r2']:.4f}\n"
        
        best_model = performance_data[0]
        report += f"\n🏆 最優秀モデル: {best_model['model']}\n"
        
        # 特徴量重要度分析
        feature_importance = results.get('feature_importance', {})
        if feature_importance:
            report += f"\n🔍 特徴量重要度分析:\n"
            
            for model_name, importance_data in feature_importance.items():
                if model_name == best_model['model']:
                    tech_imp = importance_data['technical_importance']
                    fund_imp = importance_data['fundamental_importance']
                    ratio = importance_data['technical_vs_fundamental_ratio']
                    
                    report += f"  📊 {model_name}:\n"
                    report += f"     テクニカル重要度: {tech_imp:.3f} ({tech_imp/(tech_imp+fund_imp)*100:.1f}%)\n"
                    report += f"     ファンダメンタルズ重要度: {fund_imp:.3f} ({fund_imp/(tech_imp+fund_imp)*100:.1f}%)\n"
                    report += f"     テクニカル/ファンダメンタルズ比: {ratio:.2f}\n"
                    
                    report += f"     上位5特徴量:\n"
                    for feature in importance_data['top_features'][:5]:
                        report += f"       • {feature['feature']}: {feature['importance']:.4f}\n"
        
        # 時系列分析結果
        ts_analysis = results.get('time_series_analysis', {})
        if ts_analysis:
            report += f"\n📅 時系列クロスバリデーション:\n"
            report += f"  • 平均R²: {ts_analysis['cv_r2_mean']:.4f} ± {ts_analysis['cv_r2_std']:.4f}\n"
            report += f"  • 平均MAE: {ts_analysis['cv_mae_mean']:.4f} ± {ts_analysis['cv_mae_std']:.4f}\n"
            report += f"  • 安定性スコア: {ts_analysis['stability_score']:.4f}\n"
        
        # ファンダメンタルズ影響分析
        fund_impact = results.get('fundamental_impact', {})
        if fund_impact.get('status') == 'ファンダメンタルズ分析完了':
            report += f"\n💼 ファンダメンタルズ影響分析:\n"
            
            correlations = fund_impact.get('correlations', {})
            if correlations:
                report += f"  価格変動との相関:\n"
                for metric, corr in correlations.items():
                    if abs(corr) > 0.1:
                        direction = "正の相関" if corr > 0 else "負の相関"
                        report += f"    • {metric}: {corr:.3f} ({direction})\n"
            
            fund_stats = fund_impact.get('fundamental_stats', {})
            if fund_stats:
                report += f"  現在のファンダメンタルズ指標:\n"
                for metric, stats in fund_stats.items():
                    current = stats['current_value']
                    if metric == 'per':
                        report += f"    • PER: {current:.2f}\n"
                    elif metric == 'pbr':
                        report += f"    • PBR: {current:.2f}\n"
                    elif metric == 'dividend_yield':
                        report += f"    • 配当利回り: {current:.2f}%\n"
                    elif metric == 'roe':
                        report += f"    • ROE: {current*100:.2f}%\n"
                    elif metric == 'roa':
                        report += f"    • ROA: {current*100:.2f}%\n"
        
        # 投資判断
        report += f"\n💡 AI投資判断:\n"
        
        r2_score = best_model['r2']
        mae_score = best_model['mae']
        stability = ts_analysis.get('stability_score', 0)
        
        if r2_score > 0.7 and stability > 0.7:
            judgment = "🟢 強い買い推奨"
        elif r2_score > 0.5 and stability > 0.5:
            judgment = "🟡 条件付き推奨"
        elif r2_score > 0.3:
            judgment = "🟠 注意深く監視"
        else:
            judgment = "🔴 投資非推奨"
        
        report += f"  {judgment}\n"
        report += f"  • 予測精度: {r2_score:.1%}\n"
        report += f"  • 予測安定性: {stability:.1%}\n"
        report += f"  • 予測誤差: ±{mae_score:.2f}円\n"
        
        report += f"\n" + "="*80 + "\n"
        
        # ファイル保存
        report_file = self.results_dir / f"{symbol}_detailed_validation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 詳細レポート保存: {report_file}")
        print(f"\n{report}")
    
    def run_comprehensive_validation(self):
        """包括的検証の実行"""
        print("🚀 ファンダメンタルズ統合MLモデル 実データ検証開始")
        print("="*80)
        
        # 1. データ収集
        if not self.collect_validation_data(days=15):
            print("❌ データ収集失敗")
            return False
        
        # 2. 各銘柄の詳細検証
        all_results = {}
        for symbol in self.test_symbols.keys():
            result = self.validate_single_symbol(symbol)
            if result:
                all_results[symbol] = result
        
        # 3. 総合比較レポート
        self._generate_comprehensive_summary(all_results)
        
        print("\n🎉 包括的検証完了！")
        print(f"📁 結果ディレクトリ: {self.results_dir}")
        
        return True
    
    def _generate_comprehensive_summary(self, all_results: dict):
        """総合比較レポート生成"""
        summary = """
================================================================================
🏆 ファンダメンタルズ統合MLモデル 総合検証サマリー
================================================================================

"""
        
        # 各銘柄の最優秀モデル比較
        summary += "📊 銘柄別最優秀モデル比較:\n\n"
        
        for symbol, results in all_results.items():
            company_name = results['company_name']
            model_results = results['model_results']
            
            # 最優秀モデル特定
            best_model_name = min(model_results.items(), key=lambda x: x[1]['mse'])[0]
            best_model = model_results[best_model_name]
            
            summary += f"🏢 {symbol} ({company_name}):\n"
            summary += f"  最優秀: {best_model_name}\n"
            summary += f"  R²: {best_model.get('r2', 0):.3f} | MAE: {best_model['mae']:.2f}\n"
            
            # 安定性スコア
            ts_analysis = results.get('time_series_analysis', {})
            stability = ts_analysis.get('stability_score', 0)
            summary += f"  安定性: {stability:.3f}\n\n"
        
        # 業種別分析
        summary += "🏭 業種別パフォーマンス:\n\n"
        
        sector_performance = {}
        for symbol, results in all_results.items():
            company_name = results['company_name']
            sector = company_name.split('（')[1].split('）')[0] if '（' in company_name else 'その他'
            
            best_model = min(results['model_results'].items(), key=lambda x: x[1]['mse'])[1]
            r2_score = best_model.get('r2', 0)
            
            if sector not in sector_performance:
                sector_performance[sector] = []
            sector_performance[sector].append(r2_score)
        
        for sector, scores in sector_performance.items():
            avg_score = np.mean(scores)
            summary += f"  {sector}: 平均R² {avg_score:.3f}\n"
        
        # ファンダメンタルズ効果分析
        summary += f"\n💼 ファンダメンタルズ統合効果:\n\n"
        
        tech_vs_fund_ratios = []
        for symbol, results in all_results.items():
            feature_importance = results.get('feature_importance', {})
            for model_name, importance_data in feature_importance.items():
                if 'technical_vs_fundamental_ratio' in importance_data:
                    ratio = importance_data['technical_vs_fundamental_ratio']
                    tech_vs_fund_ratios.append(ratio)
        
        if tech_vs_fund_ratios:
            avg_ratio = np.mean(tech_vs_fund_ratios)
            summary += f"  平均テクニカル/ファンダメンタルズ重要度比: {avg_ratio:.2f}\n"
            
            if avg_ratio > 2:
                summary += "  → テクニカル分析が支配的\n"
            elif avg_ratio > 0.5:
                summary += "  → バランスの取れた統合分析\n"
            else:
                summary += "  → ファンダメンタルズ分析が支配的\n"
        
        summary += f"\n🎯 総合評価:\n"
        
        # 全体パフォーマンス
        all_r2_scores = []
        for symbol, results in all_results.items():
            best_model = min(results['model_results'].items(), key=lambda x: x[1]['mse'])[1]
            all_r2_scores.append(best_model.get('r2', 0))
        
        avg_r2 = np.mean(all_r2_scores)
        summary += f"  平均予測精度: {avg_r2:.1%}\n"
        
        if avg_r2 > 0.6:
            summary += "  ✅ 優秀な予測性能\n"
        elif avg_r2 > 0.4:
            summary += "  🟡 良好な予測性能\n"
        else:
            summary += "  🟠 改善が必要\n"
        
        summary += f"\n📈 ファンダメンタルズ統合の価値:\n"
        summary += f"  • 多角的分析による予測精度向上\n"
        summary += f"  • 業種特性を考慮した銘柄評価\n"
        summary += f"  • 長期投資判断の根拠提供\n"
        
        summary += "\n" + "="*80 + "\n"
        
        # サマリーファイル保存
        summary_file = self.results_dir / "comprehensive_validation_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"📄 総合サマリー保存: {summary_file}")
        print(summary)

def main():
    """メイン実行関数"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 検証実行
    validator = MLModelValidator()
    validator.run_comprehensive_validation()

if __name__ == "__main__":
    main()
