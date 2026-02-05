"""
デイトレード特化モデル比較検証システム
短期取引での利益率・勝率を重視した複数モデル比較
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'

@dataclass
class DayTradingPerformance:
    """デイトレード性能評価"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # デイトレード特化指標
    profit_rate: float          # 利益率
    win_rate: float             # 勝率
    avg_profit_per_trade: float # 平均利益/取引
    max_drawdown: float         # 最大ドローダウン
    sharpe_ratio: float         # シャープ比率
    
    # 取引統計
    total_trades: int           # 総取引数
    profitable_trades: int      # 利益取引数
    avg_holding_time: float     # 平均保有時間(分)
    
    # リスク指標
    max_consecutive_losses: int # 最大連続損失
    profit_factor: float        # プロフィットファクター
    
    # 実行時間
    training_time: float
    prediction_time: float

class DayTradingModelComparison:
    """デイトレード特化モデル比較システム"""
    
    def __init__(self, db_path: str = "enhanced_trading.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.performance_history = []
        
        # デイトレード特化モデル設定
        self.model_configs = self._define_daytrading_models()
        
        # 結果保存ディレクトリ
        self.results_dir = Path("reports/daytrading_comparison")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _define_daytrading_models(self) -> Dict:
        """デイトレード特化モデル定義"""
        return {
            # 1. 高速スキャルピングモデル（1-5分）
            'scalping_rf': {
                'model': RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42),
                'features': ['price_change_1min', 'volume_spike', 'bid_ask_spread', 'momentum_5min'],
                'target_timeframe': '1-5min',
                'description': '超短期スキャルピング特化'
            },
            
            # 2. モメンタムブレイクアウトモデル（5-15分）
            'momentum_gb': {
                'model': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
                'features': ['momentum_5min', 'momentum_15min', 'volume_ratio', 'price_acceleration'],
                'target_timeframe': '5-15min',
                'description': 'モメンタムブレイクアウト特化'
            },
            
            # 3. 平均回帰モデル（15-30分）
            'mean_reversion_lr': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'features': ['rsi', 'bollinger_position', 'price_deviation', 'volume_mean_ratio'],
                'target_timeframe': '15-30min',
                'description': '平均回帰取引特化'
            },
            
            # 4. 高頻度取引モデル（分単位）
            'high_frequency_svm': {
                'model': SVC(kernel='rbf', C=1.0, random_state=42, probability=True),
                'features': ['order_flow', 'microstructure', 'tick_momentum', 'spread_dynamics'],
                'target_timeframe': '1-3min',
                'description': '高頻度取引特化'
            },
            
            # 5. 複合デイトレードモデル
            'hybrid_daytrading': {
                'model': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
                'features': ['price_change_1min', 'momentum_5min', 'rsi', 'volume_ratio', 'bid_ask_spread'],
                'target_timeframe': '5-20min',
                'description': '複合デイトレード戦略'
            },
            
            # 6. AIトレンドフォローモデル
            'ai_trend_follow': {
                'model': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
                'features': ['trend_strength', 'momentum_15min', 'volume_trend', 'price_acceleration'],
                'target_timeframe': '10-30min',
                'description': 'AIトレンドフォロー特化'
            }
        }
    
    def prepare_daytrading_features(self, symbol: str, lookback_minutes: int = 120) -> pd.DataFrame:
        """デイトレード特化特徴量準備"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 5分足データ取得
            query = '''
                SELECT timestamp, symbol, open_price, high_price, low_price, close_price, volume
                FROM five_minute_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, lookback_minutes // 5))
            conn.close()
            
            if len(df) < 20:
                return pd.DataFrame()
            
            df = df.sort_values('timestamp')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # デイトレード特化特徴量計算
            df = self._calculate_daytrading_features(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"デイトレード特徴量準備エラー {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_daytrading_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """デイトレード特化特徴量計算"""
        # 価格変動特徴量
        df['price_change_1min'] = df['close_price'].pct_change()
        df['price_change_5min'] = df['close_price'].pct_change(5)
        df['price_acceleration'] = df['price_change_1min'].diff()
        
        # モメンタム指標
        df['momentum_5min'] = df['close_price'] / df['close_price'].shift(5) - 1
        df['momentum_15min'] = df['close_price'] / df['close_price'].shift(15) - 1
        
        # ボラティリティ指標
        df['volatility_5min'] = df['price_change_1min'].rolling(5).std()
        df['volatility_15min'] = df['price_change_1min'].rolling(15).std()
        
        # 出来高分析
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(15).mean()
        
        # テクニカル指標
        df['rsi'] = self._calculate_rsi(df['close_price'], 14)
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close_price'])
        df['bollinger_position'] = (df['close_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 価格位置指標
        df['price_deviation'] = (df['close_price'] - df['close_price'].rolling(20).mean()) / df['close_price'].rolling(20).std()
        
        # 高頻度取引指標（模擬）
        df['bid_ask_spread'] = np.random.normal(0.001, 0.0005, len(df))  # 模擬スプレッド
        df['order_flow'] = np.random.normal(0, 1, len(df))  # 模擬オーダーフロー
        df['microstructure'] = np.random.normal(0, 1, len(df))  # 模擬マイクロストラクチャー
        df['tick_momentum'] = df['close_price'].diff().rolling(3).mean()
        df['spread_dynamics'] = df['bid_ask_spread'].diff()
        
        # トレンド強度
        df['trend_strength'] = abs(df['close_price'].rolling(10).mean() - df['close_price'].rolling(30).mean())
        
        # 平均回帰指標
        df['volume_mean_ratio'] = df['volume'] / df['volume'].mean()
        
        return df.dropna()
    
    def generate_daytrading_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """デイトレード用ターゲット生成"""
        # 複数の時間軸でターゲット生成
        targets = {}
        
        # 1分後の価格上昇（スキャルピング）
        targets['target_1min'] = (df['close_price'].shift(-1) > df['close_price']).astype(int)
        
        # 5分後の価格上昇（短期トレード）
        targets['target_5min'] = (df['close_price'].shift(-5) > df['close_price']).astype(int)
        
        # 15分後の価格上昇（中期トレード）
        targets['target_15min'] = (df['close_price'].shift(-15) > df['close_price']).astype(int)
        
        # 利益率ベースターゲット（1%以上の利益）
        future_max = df['high_price'].rolling(10, min_periods=1).max().shift(-10)
        targets['target_profit_1pct'] = ((future_max / df['close_price'] - 1) >= 0.01).astype(int)
        
        for target_name, target_values in targets.items():
            df[target_name] = target_values
        
        return df
    
    def simulate_daytrading(self, predictions: np.ndarray, probabilities: np.ndarray, 
                           prices: np.ndarray, target_timeframe: str = '5-15min') -> Dict:
        """デイトレード取引シミュレーション"""
        try:
            # 取引パラメータ
            initial_capital = 1000000  # 100万円
            position_size = 0.1  # 10%ポジション
            transaction_cost = 0.001  # 0.1%取引コスト
            slippage = 0.0005  # 0.05%スリッページ
            
            # 時間軸別設定
            if '1-3min' in target_timeframe:
                stop_loss = 0.005  # 0.5%ストップロス
                take_profit = 0.01  # 1%利確
                holding_periods = np.random.randint(1, 4, len(predictions))
            elif '5-15min' in target_timeframe:
                stop_loss = 0.01  # 1%ストップロス
                take_profit = 0.02  # 2%利確
                holding_periods = np.random.randint(5, 16, len(predictions))
            else:
                stop_loss = 0.015  # 1.5%ストップロス
                take_profit = 0.03  # 3%利確
                holding_periods = np.random.randint(15, 31, len(predictions))
            
            # 取引シミュレーション
            portfolio_value = initial_capital
            portfolio_history = [portfolio_value]
            trades = []
            
            position = None
            consecutive_losses = 0
            max_consecutive_losses = 0
            
            for i in range(len(predictions)):
                current_price = prices[i]
                
                # エントリー判定
                if position is None and predictions[i] == 1:
                    confidence = probabilities[i]
                    if confidence > 0.6:  # 高信頼度のみエントリー
                        position = {
                            'entry_price': current_price * (1 + slippage),
                            'entry_time': i,
                            'direction': 'long',
                            'size': portfolio_value * position_size,
                            'stop_loss': current_price * (1 - stop_loss),
                            'take_profit': current_price * (1 + take_profit),
                            'holding_period': holding_periods[i]
                        }
                
                # ポジション管理
                if position is not None:
                    # 利確・損切り判定
                    if (current_price >= position['take_profit'] or 
                        current_price <= position['stop_loss'] or
                        i - position['entry_time'] >= position['holding_period']):
                        
                        # 決済
                        exit_price = current_price * (1 - slippage)
                        pnl = (exit_price - position['entry_price']) * position['size'] / position['entry_price']
                        pnl -= position['size'] * transaction_cost * 2  # 往復手数料
                        
                        portfolio_value += pnl
                        
                        # 取引記録
                        trade_result = {
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'holding_time': i - position['entry_time'],
                            'profit': pnl > 0
                        }
                        trades.append(trade_result)
                        
                        # 連続損失カウント
                        if pnl <= 0:
                            consecutive_losses += 1
                            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        else:
                            consecutive_losses = 0
                        
                        position = None
                
                portfolio_history.append(portfolio_value)
            
            # 性能計算
            if trades:
                total_pnl = sum(t['pnl'] for t in trades)
                profitable_trades = sum(1 for t in trades if t['profit'])
                
                performance = {
                    'total_trades': len(trades),
                    'profitable_trades': profitable_trades,
                    'win_rate': profitable_trades / len(trades),
                    'profit_rate': total_pnl / initial_capital,
                    'avg_profit_per_trade': total_pnl / len(trades),
                    'max_consecutive_losses': max_consecutive_losses,
                    'avg_holding_time': sum(t['holding_time'] for t in trades) / len(trades)
                }
                
                # リスク指標
                returns = np.diff(portfolio_history) / np.array(portfolio_history[:-1])
                performance['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                
                # 最大ドローダウン
                peak = np.maximum.accumulate(portfolio_history)
                drawdown = (np.array(portfolio_history) - peak) / peak
                performance['max_drawdown'] = abs(np.min(drawdown))
                
                # プロフィットファクター
                gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
                gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
                performance['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
            else:
                performance = {
                    'total_trades': 0, 'profitable_trades': 0, 'win_rate': 0,
                    'profit_rate': 0, 'avg_profit_per_trade': 0, 'max_consecutive_losses': 0,
                    'avg_holding_time': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'profit_factor': 0
                }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"デイトレードシミュレーションエラー: {e}")
            return {}
    
    def compare_daytrading_models(self, symbols: List[str]) -> List[DayTradingPerformance]:
        """デイトレードモデル比較実行"""
        try:
            self.logger.info("デイトレードモデル比較開始...")
            
            performances = []
            
            for model_name, config in self.model_configs.items():
                self.logger.info(f"モデル '{model_name}' の学習・評価開始...")
                
                # 全銘柄のデータ統合
                all_features = []
                all_targets = []
                all_prices = []
                
                for symbol in symbols:
                    df = self.prepare_daytrading_features(symbol)
                    if df.empty:
                        continue
                    
                    # ターゲット生成
                    df = self.generate_daytrading_targets(df)
                    
                    # 特徴量選択
                    available_features = [f for f in config['features'] if f in df.columns]
                    if not available_features:
                        continue
                    
                    X = df[available_features].fillna(0).values
                    y = df['target_5min'].fillna(0).values  # 5分後予測をメイン使用
                    prices = df['close_price'].values
                    
                    # データ追加
                    all_features.extend(X)
                    all_targets.extend(y)
                    all_prices.extend(prices)
                
                if len(all_features) == 0:
                    self.logger.warning(f"学習データが不足: {model_name}")
                    continue
                
                X = np.array(all_features)
                y = np.array(all_targets)
                prices = np.array(all_prices)
                
                # 時系列分割
                tscv = TimeSeriesSplit(n_splits=3)
                fold_performances = []
                
                for train_index, test_index in tscv.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    prices_test = prices[test_index]
                    
                    # スケーリング
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # モデル学習
                    start_time = datetime.now()
                    model = config['model']
                    model.fit(X_train_scaled, y_train)
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    # 予測
                    start_time = datetime.now()
                    predictions = model.predict(X_test_scaled)
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        probabilities = model.decision_function(X_test_scaled)
                        probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())
                    prediction_time = (datetime.now() - start_time).total_seconds()
                    
                    # 基本性能評価
                    accuracy = accuracy_score(y_test, predictions)
                    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
                    
                    # デイトレード性能評価
                    trading_performance = self.simulate_daytrading(
                        predictions, probabilities, prices_test, config['target_timeframe']
                    )
                    
                    fold_performance = {
                        'accuracy': accuracy,
                        'precision': report['weighted avg']['precision'],
                        'recall': report['weighted avg']['recall'],
                        'f1_score': report['weighted avg']['f1-score'],
                        'training_time': training_time,
                        'prediction_time': prediction_time,
                        **trading_performance
                    }
                    
                    fold_performances.append(fold_performance)
                
                # 平均性能計算
                if fold_performances:
                    avg_performance = {}
                    for key in fold_performances[0].keys():
                        avg_performance[key] = np.mean([fp[key] for fp in fold_performances])
                    
                    performance = DayTradingPerformance(
                        model_name=model_name,
                        accuracy=avg_performance['accuracy'],
                        precision=avg_performance['precision'],
                        recall=avg_performance['recall'],
                        f1_score=avg_performance['f1_score'],
                        profit_rate=avg_performance['profit_rate'],
                        win_rate=avg_performance['win_rate'],
                        avg_profit_per_trade=avg_performance['avg_profit_per_trade'],
                        max_drawdown=avg_performance['max_drawdown'],
                        sharpe_ratio=avg_performance['sharpe_ratio'],
                        total_trades=int(avg_performance['total_trades']),
                        profitable_trades=int(avg_performance['profitable_trades']),
                        avg_holding_time=avg_performance['avg_holding_time'],
                        max_consecutive_losses=int(avg_performance['max_consecutive_losses']),
                        profit_factor=avg_performance['profit_factor'],
                        training_time=avg_performance['training_time'],
                        prediction_time=avg_performance['prediction_time']
                    )
                    
                    performances.append(performance)
                    
                    self.logger.info(f"モデル '{model_name}' 評価完了 - 利益率: {performance.profit_rate:.3f}, 勝率: {performance.win_rate:.3f}")
            
            # 結果保存
            self.save_daytrading_results(performances)
            
            return performances
            
        except Exception as e:
            self.logger.error(f"デイトレードモデル比較エラー: {e}")
            return []
    
    def save_daytrading_results(self, performances: List[DayTradingPerformance]):
        """デイトレード結果保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # JSON保存
            results = [asdict(p) for p in performances]
            results_file = self.results_dir / f"daytrading_comparison_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 比較チャート作成
            self.create_daytrading_charts(performances, timestamp)
            
            self.logger.info(f"デイトレード結果保存完了: {results_file}")
            
        except Exception as e:
            self.logger.error(f"結果保存エラー: {e}")
    
    def create_daytrading_charts(self, performances: List[DayTradingPerformance], timestamp: str):
        """デイトレード比較チャート作成"""
        try:
            if not performances:
                return
            
            # 詳細比較チャート
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            model_names = [p.model_name for p in performances]
            
            # 1. 利益率比較
            profit_rates = [p.profit_rate for p in performances]
            axes[0, 0].bar(model_names, profit_rates)
            axes[0, 0].set_title('利益率比較')
            axes[0, 0].set_ylabel('利益率')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. 勝率比較
            win_rates = [p.win_rate for p in performances]
            axes[0, 1].bar(model_names, win_rates)
            axes[0, 1].set_title('勝率比較')
            axes[0, 1].set_ylabel('勝率')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. シャープ比率比較
            sharpe_ratios = [p.sharpe_ratio for p in performances]
            axes[0, 2].bar(model_names, sharpe_ratios)
            axes[0, 2].set_title('シャープ比率比較')
            axes[0, 2].set_ylabel('シャープ比率')
            axes[0, 2].tick_params(axis='x', rotation=45)
            
            # 4. 最大ドローダウン比較
            max_drawdowns = [p.max_drawdown for p in performances]
            axes[1, 0].bar(model_names, max_drawdowns)
            axes[1, 0].set_title('最大ドローダウン比較')
            axes[1, 0].set_ylabel('最大ドローダウン')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 5. 平均利益/取引比較
            avg_profits = [p.avg_profit_per_trade for p in performances]
            axes[1, 1].bar(model_names, avg_profits)
            axes[1, 1].set_title('平均利益/取引比較')
            axes[1, 1].set_ylabel('平均利益/取引')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # 6. プロフィットファクター比較
            profit_factors = [min(p.profit_factor, 10) for p in performances]  # 上限10で表示
            axes[1, 2].bar(model_names, profit_factors)
            axes[1, 2].set_title('プロフィットファクター比較')
            axes[1, 2].set_ylabel('プロフィットファクター')
            axes[1, 2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            chart_file = self.results_dir / f"daytrading_comparison_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 詳細比較表
            self.create_detailed_comparison_table(performances, timestamp)
            
        except Exception as e:
            self.logger.error(f"チャート作成エラー: {e}")
    
    def create_detailed_comparison_table(self, performances: List[DayTradingPerformance], timestamp: str):
        """詳細比較表作成"""
        try:
            fig, ax = plt.subplots(figsize=(20, 10))
            
            # データ準備
            data = []
            for p in performances:
                data.append([
                    p.model_name,
                    f"{p.profit_rate:.3f}",
                    f"{p.win_rate:.3f}",
                    f"{p.avg_profit_per_trade:.0f}",
                    f"{p.sharpe_ratio:.3f}",
                    f"{p.max_drawdown:.3f}",
                    f"{p.total_trades}",
                    f"{p.avg_holding_time:.1f}",
                    f"{p.max_consecutive_losses}",
                    f"{p.profit_factor:.2f}",
                    f"{p.training_time:.2f}s"
                ])
            
            columns = ['モデル', '利益率', '勝率', '平均利益/取引', 'シャープ比率', '最大DD', 
                      '総取引数', '平均保有時間', '最大連続損失', 'PF', '学習時間']
            
            # 表作成
            table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 2)
            
            # スタイル調整
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 最高パフォーマンス行をハイライト
            if performances:
                best_profit_idx = max(range(len(performances)), key=lambda i: performances[i].profit_rate)
                best_winrate_idx = max(range(len(performances)), key=lambda i: performances[i].win_rate)
                
                for j in range(len(columns)):
                    table[(best_profit_idx + 1, j)].set_facecolor('#90EE90')  # 最高利益率
                    if best_winrate_idx != best_profit_idx:
                        table[(best_winrate_idx + 1, j)].set_facecolor('#FFE4B5')  # 最高勝率
            
            ax.axis('off')
            ax.set_title('デイトレードモデル詳細比較', fontsize=16, fontweight='bold')
            
            table_file = self.results_dir / f"daytrading_detailed_{timestamp}.png"
            plt.savefig(table_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"比較表作成エラー: {e}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """ボリンジャーバンド計算"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, lower

def main():
    """メイン実行"""
    logging.basicConfig(level=logging.INFO)
    
    # デイトレード対象銘柄（高流動性）
    symbols = ["7203", "9984", "6758", "8306", "6501", "4063", "9432", "8035", "4519", "6861"]
    
    # デイトレードモデル比較実行
    comparison = DayTradingModelComparison()
    performances = comparison.compare_daytrading_models(symbols)
    
    # 結果表示
    print("\n" + "="*60)
    print("🏆 デイトレードモデル比較結果")
    print("="*60)
    
    if performances:
        # 利益率順でソート
        performances.sort(key=lambda x: x.profit_rate, reverse=True)
        
        print("\n📊 利益率ランキング:")
        for i, p in enumerate(performances, 1):
            print(f"{i}. {p.model_name}")
            print(f"   利益率: {p.profit_rate:.3f} | 勝率: {p.win_rate:.3f}")
            print(f"   シャープ比率: {p.sharpe_ratio:.3f} | 最大DD: {p.max_drawdown:.3f}")
            print(f"   総取引数: {p.total_trades} | 平均保有時間: {p.avg_holding_time:.1f}分")
            print(f"   プロフィットファクター: {p.profit_factor:.2f}")
            print()
        
        # 最優秀モデル
        best_model = performances[0]
        print(f"🥇 最優秀デイトレードモデル: {best_model.model_name}")
        print(f"📈 期待利益率: {best_model.profit_rate:.1%}")
        print(f"🎯 勝率: {best_model.win_rate:.1%}")
        print(f"⚡ 平均利益/取引: {best_model.avg_profit_per_trade:.0f}円")
        
        # 勝率最高モデル
        best_winrate = max(performances, key=lambda x: x.win_rate)
        if best_winrate != best_model:
            print(f"\n🎯 最高勝率モデル: {best_winrate.model_name}")
            print(f"🎯 勝率: {best_winrate.win_rate:.1%}")
            print(f"📈 利益率: {best_winrate.profit_rate:.1%}")
    
    else:
        print("❌ 比較結果が得られませんでした。")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
