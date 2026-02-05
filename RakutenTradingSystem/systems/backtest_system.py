"""
バックテストシステム - 統合版
既存のbacktest_systemを整理統合
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple

# 既存モジュールをインポート
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Tuple

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backtest_system import BacktestEngine
from core.enhanced_data_collector import EnhancedDataCollector
from core.ml_models import MLTradingSystem
from core.strategy_system import TradingStrategy
from core.technical_indicators import TechnicalIndicators

class BacktestSystem:
    """バックテストシステム統合クラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # バックテストエンジン初期化
        self.backtest_engine = BacktestEngine()
        
        self.logger.info("バックテストシステム初期化完了")
    
    def run_ml_backtest(self):
        """方法1（ML）バックテスト実行"""
        print("\n🤖 方法1（ML）バックテスト開始")
        print("="*50)
        
        # 期間設定
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)  # 過去30日
        
        print(f"📅 バックテスト期間: {start_date} - {end_date}")
        
        try:
            # 方法1バックテスト実行
            results = self.backtest_engine.simulate_method1(start_date, end_date)
            
            if results:
                self._display_backtest_results("方法1（ML）", results)
                
                # グラフ生成
                self.backtest_engine.plot_equity_curve(
                    results, 
                    title="方法1（ML）バックテスト結果",
                    filename="reports/backtest/ml_backtest_result.png"
                )
                
                print("✅ 方法1バックテスト完了")
            else:
                print("❌ バックテスト実行に失敗しました")
                
        except Exception as e:
            print(f"❌ 方法1バックテストエラー: {e}")
    
    def run_strategy_backtest(self):
        """方法2（戦略）バックテスト実行"""
        print("\n📊 方法2（戦略）バックテスト開始")
        print("="*50)
        
        # 期間設定
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)
        
        print(f"📅 バックテスト期間: {start_date} - {end_date}")
        
        try:
            # 方法2バックテスト実行
            results = self.backtest_engine.simulate_method2(start_date, end_date)
            
            if results:
                self._display_backtest_results("方法2（戦略）", results)
                
                # グラフ生成
                self.backtest_engine.plot_equity_curve(
                    results,
                    title="方法2（戦略）バックテスト結果", 
                    filename="reports/backtest/strategy_backtest_result.png"
                )
                
                print("✅ 方法2バックテスト完了")
            else:
                print("❌ バックテスト実行に失敗しました")
                
        except Exception as e:
            print(f"❌ 方法2バックテストエラー: {e}")
    
    def run_comparison_backtest(self):
        """両方法比較バックテスト実行"""
        print("\n⚡ 両方法比較バックテスト開始")
        print("="*60)
        
        # 期間設定
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)
        
        print(f"📅 バックテスト期間: {start_date} - {end_date}")
        
        try:
            # 両方法実行
            ml_results = self.backtest_engine.simulate_method1(start_date, end_date)
            strategy_results = self.backtest_engine.simulate_method2(start_date, end_date)
            
            if ml_results and strategy_results:
                # 比較分析
                comparison = self.backtest_engine.compare_methods(ml_results, strategy_results)
                
                self._display_comparison_results(comparison)
                
                # 比較グラフ生成
                self.backtest_engine.plot_comparison(
                    ml_results, strategy_results,
                    filename="reports/backtest/method_comparison_backtest.png"
                )
                
                print("✅ 比較バックテスト完了")
            else:
                print("❌ バックテスト実行に失敗しました")
                
        except Exception as e:
            print(f"❌ 比較バックテストエラー: {e}")
    
    def run_custom_backtest(self):
        """カスタム期間バックテスト実行"""
        print("\n🔧 カスタム期間バックテスト")
        print("="*50)
        
        try:
            # 期間入力
            print("バックテスト期間を入力してください:")
            
            start_str = input("開始日 (YYYY-MM-DD): ").strip()
            end_str = input("終了日 (YYYY-MM-DD): ").strip()
            
            start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_str, '%Y-%m-%d').date()
            
            if start_date >= end_date:
                print("❌ 開始日は終了日より前に設定してください")
                return
            
            # 方法選択
            print("\n実行する方法を選択してください:")
            print("1. 方法1（ML）のみ")
            print("2. 方法2（戦略）のみ")
            print("3. 両方法比較")
            
            method_choice = input("選択 (1-3): ").strip()
            
            print(f"\n📅 期間: {start_date} - {end_date}")
            
            if method_choice == '1':
                results = self.backtest_engine.simulate_method1(start_date, end_date)
                if results:
                    self._display_backtest_results("方法1（ML）", results)
                    
            elif method_choice == '2':
                results = self.backtest_engine.simulate_method2(start_date, end_date)
                if results:
                    self._display_backtest_results("方法2（戦略）", results)
                    
            elif method_choice == '3':
                ml_results = self.backtest_engine.simulate_method1(start_date, end_date)
                strategy_results = self.backtest_engine.simulate_method2(start_date, end_date)
                
                if ml_results and strategy_results:
                    comparison = self.backtest_engine.compare_methods(ml_results, strategy_results)
                    self._display_comparison_results(comparison)
            else:
                print("❌ 無効な選択です")
                return
            
            print("✅ カスタムバックテスト完了")
            
        except ValueError:
            print("❌ 日付形式が正しくありません (YYYY-MM-DD)")
        except Exception as e:
            print(f"❌ カスタムバックテストエラー: {e}")
    
    def show_backtest_results(self):
        """過去バックテスト結果確認"""
        print("\n📋 過去バックテスト結果")
        print("="*60)
        
        try:
            import sqlite3
            
            # バックテスト結果テーブル確認
            conn = sqlite3.connect("enhanced_trading.db")
            
            # テーブル存在確認
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_results'"
            ).fetchall()
            
            if not tables:
                print("過去のバックテスト結果がありません")
                conn.close()
                return
            
            # 最新結果取得
            query = """
                SELECT backtest_id, method, start_date, end_date, 
                       total_trades, win_rate, total_return, sharpe_ratio,
                       created_at
                FROM backtest_results
                ORDER BY created_at DESC
                LIMIT 10
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                print("過去のバックテスト結果がありません")
                return
            
            for _, row in df.iterrows():
                print(f"📊 {row['backtest_id']} ({row['method']})")
                print(f"   期間: {row['start_date']} - {row['end_date']}")
                print(f"   取引数: {row['total_trades']}")
                print(f"   勝率: {row['win_rate']:.1%}")
                print(f"   総利回り: {row['total_return']:.2%}")
                print(f"   シャープレシオ: {row['sharpe_ratio']:.3f}")
                print(f"   実行日: {row['created_at']}")
                print()
                
        except Exception as e:
            print(f"❌ 結果表示エラー: {e}")
        
        print("="*60)
    
    def _display_backtest_results(self, method_name: str, results: Dict):
        """バックテスト結果表示"""
        print(f"\n📊 {method_name} バックテスト結果")
        print("="*50)
        
        stats = results.get('statistics', {})
        
        print(f"期間: {results.get('start_date')} - {results.get('end_date')}")
        print(f"総取引数: {stats.get('total_trades', 0)}")
        print(f"勝率: {stats.get('win_rate', 0):.1%}")
        print(f"総利回り: {stats.get('total_return', 0):.2%}")
        print(f"最大ドローダウン: {stats.get('max_drawdown', 0):.2%}")
        print(f"シャープレシオ: {stats.get('sharpe_ratio', 0):.3f}")
        print(f"プロフィットファクター: {stats.get('profit_factor', 0):.2f}")
        
        if 'trades' in results and results['trades']:
            avg_profit = np.mean([t['pnl'] for t in results['trades'] if t['pnl'] > 0])
            avg_loss = np.mean([t['pnl'] for t in results['trades'] if t['pnl'] < 0])
            
            print(f"平均利益: {avg_profit:,.0f}円")
            print(f"平均損失: {avg_loss:,.0f}円")
        
        print("="*50)
    
    def _display_comparison_results(self, comparison: Dict):
        """比較結果表示"""
        print("\n⚡ 両方法比較結果")
        print("="*60)
        
        method1_stats = comparison.get('method1_stats', {})
        method2_stats = comparison.get('method2_stats', {})
        
        print("📊 方法1（ML）:")
        print(f"   総利回り: {method1_stats.get('total_return', 0):.2%}")
        print(f"   勝率: {method1_stats.get('win_rate', 0):.1%}")
        print(f"   シャープレシオ: {method1_stats.get('sharpe_ratio', 0):.3f}")
        
        print("\n📊 方法2（戦略）:")
        print(f"   総利回り: {method2_stats.get('total_return', 0):.2%}")
        print(f"   勝率: {method2_stats.get('win_rate', 0):.1%}")
        print(f"   シャープレシオ: {method2_stats.get('sharpe_ratio', 0):.3f}")
        
        # 優位性判定
        comparison_results = comparison.get('comparison', {})
        
        print(f"\n🏆 優位性分析:")
        print(f"   利回り優位: {comparison_results.get('better_return', 'N/A')}")
        print(f"   勝率優位: {comparison_results.get('better_win_rate', 'N/A')}")
        print(f"   シャープレシオ優位: {comparison_results.get('better_sharpe', 'N/A')}")
        
        # 総合判定
        if 'overall_better' in comparison_results:
            print(f"   総合優位: {comparison_results['overall_better']}")
        
        print("="*60)
