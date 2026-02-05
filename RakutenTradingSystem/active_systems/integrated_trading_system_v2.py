"""
統合取引システムランチャー
ファンダメンタルズ分析対応版
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import os

# 必要なモジュールのインポート
from fundamental_data_collector import FundamentalDataCollector
from multi_model_comparison import MultiModelComparison
from core.enhanced_data_collector import EnhancedDataCollector
from core.ml_models import MLTradingModels
from systems.demo_trading_system import DemoTradingSystem

class IntegratedTradingSystem:
    """統合取引システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 各システムの初期化
        self.fundamental_collector = FundamentalDataCollector()
        self.model_comparison = MultiModelComparison()
        self.data_collector = EnhancedDataCollector()
        self.ml_models = MLTradingModels()
        self.demo_system = DemoTradingSystem()
        
        # 設定
        self.symbols = self.load_symbols()
        self.trading_active = False
        
        # 結果保存ディレクトリ
        self.results_dir = Path("reports/integrated_system")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # スレッド管理
        self.threads = []
        
    def load_symbols(self) -> list:
        """取引対象銘柄ロード"""
        try:
            import pandas as pd
            df = pd.read_csv("prime_symbols.csv")
            # 上位時価総額の銘柄を選択
            symbols = df['symbol'].astype(str).tolist()[:20]
            return symbols
        except Exception as e:
            self.logger.error(f"銘柄ロードエラー: {e}")
            return ["7203", "9984", "6758", "8306", "6501", "4063", "9432", "8035"]
    
    def collect_fundamental_data(self):
        """ファンダメンタルズデータ収集"""
        try:
            self.logger.info("ファンダメンタルズデータ収集開始...")
            
            # 全銘柄のファンダメンタルズデータ収集
            fundamental_data = self.fundamental_collector.collect_fundamental_data(
                self.symbols, use_excel=False
            )
            
            # 業界別平均計算
            sector_averages = self.fundamental_collector.calculate_sector_averages()
            
            self.logger.info(f"ファンダメンタルズデータ収集完了: {len(fundamental_data)} 銘柄")
            return fundamental_data
            
        except Exception as e:
            self.logger.error(f"ファンダメンタルズデータ収集エラー: {e}")
            return {}
    
    def run_model_comparison(self):
        """モデル比較実行"""
        try:
            self.logger.info("モデル比較検証開始...")
            
            # モデル比較実行
            performances = self.model_comparison.compare_models(self.symbols)
            
            # 結果保存
            self.save_comparison_results(performances)
            
            # 最優秀モデル選定
            if performances:
                best_model = max(performances, key=lambda x: x.sharpe_ratio)
                self.logger.info(f"最優秀モデル: {best_model.model_name}")
                self.logger.info(f"シャープ比率: {best_model.sharpe_ratio:.3f}")
                
                # 設定更新
                self.update_trading_config(best_model.model_name)
                
                return best_model
            
            return None
            
        except Exception as e:
            self.logger.error(f"モデル比較エラー: {e}")
            return None
    
    def save_comparison_results(self, performances):
        """比較結果保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 詳細結果
            results = {
                'timestamp': timestamp,
                'models': [
                    {
                        'name': p.model_name,
                        'accuracy': p.accuracy,
                        'sharpe_ratio': p.sharpe_ratio,
                        'total_return': p.total_return,
                        'win_rate': p.win_rate,
                        'max_drawdown': p.max_drawdown
                    }
                    for p in performances
                ],
                'best_model': max(performances, key=lambda x: x.sharpe_ratio).model_name if performances else None
            }
            
            # JSON保存
            results_file = self.results_dir / f"comparison_results_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"比較結果保存完了: {results_file}")
            
        except Exception as e:
            self.logger.error(f"比較結果保存エラー: {e}")
    
    def update_trading_config(self, best_model_name: str):
        """取引設定更新"""
        try:
            config = {
                'best_model': best_model_name,
                'use_fundamental': 'fundamental' in best_model_name or 'hybrid' in best_model_name,
                'updated_at': datetime.now().isoformat()
            }
            
            config_file = Path("configs/trading_config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"取引設定更新完了: {best_model_name}")
            
        except Exception as e:
            self.logger.error(f"取引設定更新エラー: {e}")
    
    def run_comparison_only(self):
        """比較検証のみ実行"""
        try:
            self.logger.info("=== モデル比較検証のみ実行 ===")
            
            # ファンダメンタルズデータ収集
            print("ファンダメンタルズデータ収集...")
            fundamental_data = self.collect_fundamental_data()
            
            # モデル比較実行
            print("モデル比較検証実行...")
            performances = self.model_comparison.compare_models(self.symbols)
            
            # 結果表示
            print("\\n=== モデル比較結果 ===")
            if performances:
                for p in performances:
                    print(f"\\n{p.model_name}:")
                    print(f"  精度: {p.accuracy:.3f}")
                    print(f"  シャープ比率: {p.sharpe_ratio:.3f}")
                    print(f"  総収益率: {p.total_return:.3f}")
                    print(f"  勝率: {p.win_rate:.3f}")
                    print(f"  最大ドローダウン: {p.max_drawdown:.3f}")
                
                # 最優秀モデル
                best_model = max(performances, key=lambda x: x.sharpe_ratio)
                print(f"\\n最優秀モデル: {best_model.model_name}")
                print(f"シャープ比率: {best_model.sharpe_ratio:.3f}")
                
                # 結果保存
                self.save_comparison_results(performances)
                
            else:
                print("比較結果なし")
            
            print("\\n=== 比較検証完了 ===")
            
        except Exception as e:
            self.logger.error(f"比較検証エラー: {e}")

def main():
    """メイン実行"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('integrated_trading.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # システム初期化
    system = IntegratedTradingSystem()
    
    # 実行モード選択
    import sys
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "comparison":
            system.run_comparison_only()
        else:
            print("使用法: python integrated_trading_system.py comparison")
    else:
        # デフォルトは比較検証のみ
        system.run_comparison_only()

if __name__ == "__main__":
    main()
