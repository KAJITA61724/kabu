#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
木曜データ学習→金曜取引デモシステム
木曜のデータで予測モデルを訓練し、金曜の取引をシミュレーション
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 必要なライブラリのインポート
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    sklearn_available = True
except ImportError:
    sklearn_available = False

try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    lightgbm_available = False

class ThursdayToFridayTradingDemo:
    """木曜データ学習→金曜取引デモクラス"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # ディレクトリ設定
        self.models_dir = Path("models/thursday_friday")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.demo_reports_dir = Path("reports/thursday_friday_demo")
        self.demo_reports_dir.mkdir(parents=True, exist_ok=True)
        
        # 取引設定
        self.initial_capital = 1000000  # 100万円
        self.position_size = 0.2  # 20%ポジション
        self.stop_loss = 0.02  # 2%ストップロス
        self.take_profit = 0.03  # 3%利確
        self.transaction_cost = 0.001  # 0.1%取引コスト
        
    def get_specific_date_data(self, target_date: str, symbols: List[str]) -> pd.DataFrame:
        """特定日のデータを取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            all_data = []
            for symbol in symbols:
                query = '''
                    SELECT symbol, datetime, open_price, high_price, low_price, close_price, volume
                    FROM chart_data 
                    WHERE symbol = ? AND datetime LIKE ? AND timeframe = '5M'
                    ORDER BY datetime
                '''
                
                df = pd.read_sql_query(query, conn, params=(symbol, f'{target_date}%'))
                if not df.empty:
                    all_data.append(df)
                    self.logger.info(f"✅ {symbol}: {target_date}に{len(df)}件のデータ")
                else:
                    self.logger.warning(f"⚠️ {symbol}: {target_date}のデータなし")
            
            conn.close()
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                self.logger.info(f"総データ数: {len(combined_df)}件")
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"データ取得エラー: {e}")
            return pd.DataFrame()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量を作成"""
        if df.empty or len(df) < 10:
            return pd.DataFrame()
        
        df = df.copy()
        df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
        
        # 銘柄ごとに特徴量を計算
        enhanced_data = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if len(symbol_data) < 10:
                continue
            
            # 基本的な特徴量
            symbol_data['price_change'] = symbol_data['close_price'].pct_change()
            symbol_data['high_low_ratio'] = symbol_data['high_price'] / symbol_data['low_price'].replace(0, np.nan)
            symbol_data['open_close_ratio'] = symbol_data['open_price'] / symbol_data['close_price'].replace(0, np.nan)
            symbol_data['volume_price_ratio'] = symbol_data['volume'] / symbol_data['close_price'].replace(0, np.nan)
            
            # 移動平均
            symbol_data['sma_5'] = symbol_data['close_price'].rolling(window=5).mean()
            symbol_data['sma_10'] = symbol_data['close_price'].rolling(window=10).mean()
            symbol_data['sma_ratio'] = symbol_data['close_price'] / symbol_data['sma_5'].replace(0, np.nan)
            
            # ボラティリティ
            symbol_data['volatility_5'] = symbol_data['close_price'].rolling(window=5).std()
            symbol_data['volatility_10'] = symbol_data['close_price'].rolling(window=10).std()
            
            # 出来高系
            symbol_data['volume_sma_5'] = symbol_data['volume'].rolling(window=5).mean()
            symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_sma_5'].replace(0, np.nan)
            
            # RSI
            delta = symbol_data['close_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            symbol_data['rsi'] = 100 - (100 / (1 + rs))
            
            # ラグ特徴量
            for lag in [1, 2, 3, 5]:
                symbol_data[f'close_lag_{lag}'] = symbol_data['close_price'].shift(lag)
                symbol_data[f'volume_lag_{lag}'] = symbol_data['volume'].shift(lag)
                symbol_data[f'change_lag_{lag}'] = symbol_data['price_change'].shift(lag)
            
            # 時間系特徴量
            symbol_data['datetime'] = pd.to_datetime(symbol_data['datetime'])
            symbol_data['hour'] = symbol_data['datetime'].dt.hour
            symbol_data['minute'] = symbol_data['datetime'].dt.minute
            symbol_data['time_of_day'] = symbol_data['hour'] * 60 + symbol_data['minute']
            
            # 目標変数（次の期間の価格変化率）
            symbol_data['target'] = symbol_data['close_price'].shift(-1)
            symbol_data['target_change'] = (symbol_data['target'] / symbol_data['close_price'] - 1) * 100
            
            enhanced_data.append(symbol_data)
        
        if enhanced_data:
            result_df = pd.concat(enhanced_data, ignore_index=True)
            return result_df.dropna()
        else:
            return pd.DataFrame()
    
    def train_thursday_models(self, thursday_date: str, symbols: List[str]) -> Dict:
        """木曜のデータでモデルを訓練"""
        self.logger.info(f"=== 木曜データ（{thursday_date}）でモデル訓練 ===")
        
        # 木曜のデータを取得
        thursday_data = self.get_specific_date_data(thursday_date, symbols)
        
        if thursday_data.empty:
            self.logger.error("木曜のデータが取得できません")
            return {}
        
        # 特徴量を作成
        featured_data = self.create_features(thursday_data)
        
        if featured_data.empty:
            self.logger.error("特徴量の作成に失敗しました")
            return {}
        
        # 特徴量の列を定義
        feature_cols = [
            'price_change', 'high_low_ratio', 'open_close_ratio', 'volume_price_ratio',
            'sma_ratio', 'volatility_5', 'volatility_10', 'volume_ratio', 'rsi',
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
            'change_lag_1', 'change_lag_2', 'change_lag_3', 'change_lag_5',
            'hour', 'minute', 'time_of_day'
        ]
        
        # データ準備
        X = featured_data[feature_cols].copy()
        y = featured_data['target_change'].copy()
        
        # 無限大や異常値を処理
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        X = X.clip(-1e6, 1e6)
        
        self.logger.info(f"訓練データ: {len(X)}サンプル, {len(feature_cols)}特徴量")
        
        # モデル訓練
        models = {}
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # RandomForest
        if sklearn_available:
            self.logger.info("RandomForest 訓練中...")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_scaled, y)
            models['RandomForest'] = rf_model
            
            # LinearRegression
            self.logger.info("LinearRegression 訓練中...")
            lr_model = LinearRegression()
            lr_model.fit(X_scaled, y)
            models['LinearRegression'] = lr_model
        
        # LightGBM
        if lightgbm_available:
            try:
                self.logger.info("LightGBM 訓練中...")
                lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
                lgb_model.fit(X, y)  # LightGBMはスケーリング不要
                models['LightGBM'] = lgb_model
            except Exception as e:
                self.logger.warning(f"LightGBM訓練エラー: {e}")
        
        # モデルとスケーラーを保存
        joblib.dump(scaler, self.models_dir / f'scaler_{thursday_date}.pkl')
        joblib.dump(feature_cols, self.models_dir / f'features_{thursday_date}.pkl')
        
        for model_name, model in models.items():
            model_path = self.models_dir / f'{model_name}_{thursday_date}.pkl'
            joblib.dump(model, model_path)
            self.logger.info(f"モデル保存: {model_path}")
        
        return {
            'models': models,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'training_data_count': len(X)
        }
    
    def predict_friday_signals(self, friday_date: str, symbols: List[str], 
                              trained_models: Dict) -> pd.DataFrame:
        """金曜のデータで予測シグナルを生成"""
        self.logger.info(f"=== 金曜データ（{friday_date}）で予測シグナル生成 ===")
        
        # 金曜のデータを取得
        friday_data = self.get_specific_date_data(friday_date, symbols)
        
        if friday_data.empty:
            self.logger.error("金曜のデータが取得できません")
            return pd.DataFrame()
        
        # 特徴量を作成
        featured_data = self.create_features(friday_data)
        
        if featured_data.empty:
            self.logger.error("金曜の特徴量作成に失敗しました")
            return pd.DataFrame()
        
        # 予測実行
        models = trained_models['models']
        scaler = trained_models['scaler']
        feature_cols = trained_models['feature_cols']
        
        # 特徴量データ準備
        X = featured_data[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        X = X.clip(-1e6, 1e6)
        
        # 各モデルで予測
        predictions = {}
        
        for model_name, model in models.items():
            try:
                if model_name == 'LightGBM':
                    pred = model.predict(X)
                else:
                    X_scaled = scaler.transform(X)
                    pred = model.predict(X_scaled)
                
                predictions[f'{model_name}_prediction'] = pred
                self.logger.info(f"{model_name} 予測完了")
                
            except Exception as e:
                self.logger.error(f"{model_name} 予測エラー: {e}")
        
        # 予測結果をデータフレームに追加
        for pred_name, pred_values in predictions.items():
            featured_data[pred_name] = pred_values
        
        # アンサンブル予測（平均）
        if predictions:
            pred_cols = list(predictions.keys())
            featured_data['ensemble_prediction'] = featured_data[pred_cols].mean(axis=1)
        
        return featured_data
    
    def simulate_friday_trading(self, prediction_data: pd.DataFrame) -> Dict:
        """金曜の取引シミュレーション"""
        self.logger.info("=== 金曜取引シミュレーション ===")
        
        if prediction_data.empty:
            return {}
        
        # ポートフォリオ初期化
        portfolio_value = self.initial_capital
        portfolio_history = [portfolio_value]
        trades = []
        positions = {}
        
        # 時刻順にソート
        prediction_data = prediction_data.sort_values('datetime').reset_index(drop=True)
        
        for idx, row in prediction_data.iterrows():
            symbol = row['symbol']
            current_price = row['close_price']
            prediction = row.get('ensemble_prediction', 0)
            rf_pred = row.get('RandomForest_prediction', 0)
            lr_pred = row.get('LinearRegression_prediction', 0)
            lgb_pred = row.get('LightGBM_prediction', 0)
            
            # エントリーシグナル（予測が+1%以上）
            if prediction > 1.0 and symbol not in positions:
                position_value = portfolio_value * self.position_size
                shares = int(position_value / current_price)
                
                if shares > 0:
                    entry_cost = shares * current_price * (1 + self.transaction_cost)
                    
                    if entry_cost <= portfolio_value:
                        positions[symbol] = {
                            'shares': shares,
                            'entry_price': current_price,
                            'entry_time': row['datetime'],
                            'entry_cost': entry_cost,
                            'stop_loss': current_price * (1 - self.stop_loss),
                            'take_profit': current_price * (1 + self.take_profit),
                            'entry_prediction': prediction,
                            'entry_rf_pred': rf_pred,
                            'entry_lr_pred': lr_pred,
                            'entry_lgb_pred': lgb_pred,
                            'entry_reason': f"アンサンブル予測: {prediction:.2f}% (RF:{rf_pred:.2f}%, LR:{lr_pred:.2f}%, LGB:{lgb_pred:.2f}%)"
                        }
                        
                        portfolio_value -= entry_cost
                        
                        self.logger.info(f"📈 買い注文: {symbol} {shares}株 @{current_price} 投資額:{entry_cost:,.0f}円")
            
            # エグジットシグナル
            if symbol in positions:
                position = positions[symbol]
                exit_reason = ""
                exit_condition = ""
                
                # 利確・損切り判定
                if current_price >= position['take_profit']:
                    exit_reason = "利確"
                    exit_condition = f"目標価格{position['take_profit']:.0f}円到達"
                elif current_price <= position['stop_loss']:
                    exit_reason = "損切り"
                    exit_condition = f"ストップロス{position['stop_loss']:.0f}円到達"
                elif prediction < -0.5:
                    exit_reason = "予測悪化"
                    exit_condition = f"アンサンブル予測が{prediction:.2f}%に悪化"
                
                if exit_reason:
                    exit_value = position['shares'] * current_price * (1 - self.transaction_cost)
                    portfolio_value += exit_value
                    
                    pnl = exit_value - position['entry_cost']
                    pnl_pct = (pnl / position['entry_cost']) * 100
                    
                    trade_record = {
                        'symbol': symbol,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'shares': position['shares'],
                        'entry_cost': position['entry_cost'],
                        'exit_value': exit_value,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'entry_time': position['entry_time'],
                        'exit_time': row['datetime'],
                        'profit': pnl > 0,
                        'entry_reason': position['entry_reason'],
                        'exit_reason': exit_reason,
                        'exit_condition': exit_condition,
                        'entry_prediction': position['entry_prediction'],
                        'exit_prediction': prediction,
                        'entry_rf_pred': position['entry_rf_pred'],
                        'entry_lr_pred': position['entry_lr_pred'],
                        'entry_lgb_pred': position['entry_lgb_pred'],
                        'exit_rf_pred': rf_pred,
                        'exit_lr_pred': lr_pred,
                        'exit_lgb_pred': lgb_pred,
                        'stop_loss_price': position['stop_loss'],
                        'take_profit_price': position['take_profit']
                    }
                    
                    trades.append(trade_record)
                    del positions[symbol]
                    
                    self.logger.info(f"📉 売り注文: {symbol} {exit_reason} 売却額:{exit_value:,.0f}円 損益: {pnl:.0f}円 ({pnl_pct:.2f}%)")
            
            portfolio_history.append(portfolio_value)
        
        # 最終的な成績計算
        total_trades = len(trades)
        profitable_trades = sum(1 for t in trades if t['profit'])
        total_pnl = sum(t['pnl'] for t in trades)
        total_invested = sum(t['entry_cost'] for t in trades)
        
        performance = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': portfolio_value,
            'total_pnl': total_pnl,
            'total_return_pct': (portfolio_value / self.initial_capital - 1) * 100,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'total_invested': total_invested,
            'trades': trades,
            'portfolio_history': portfolio_history
        }
        
        return performance
    
    def generate_demo_report(self, thursday_date: str, friday_date: str, 
                           performance: Dict, symbols: List[str]) -> str:
        """デモレポートを生成"""
        report = f"""
=== 木曜→金曜 取引デモレポート ===
訓練日: {thursday_date}
取引日: {friday_date}
対象銘柄: {', '.join(symbols)}

📊 取引成績:
初期資本: {performance['initial_capital']:,}円
最終評価額: {performance['final_portfolio_value']:,.0f}円
総損益: {performance['total_pnl']:,.0f}円
リターン: {performance['total_return_pct']:.2f}%
総投資額: {performance['total_invested']:,.0f}円

📈 取引統計:
総取引数: {performance['total_trades']}
利益取引: {performance['profitable_trades']}
勝率: {performance['win_rate']:.1%}
平均損益/取引: {performance['avg_pnl_per_trade']:,.0f}円

🔍 詳細取引履歴:
"""
        
        for i, trade in enumerate(performance['trades'], 1):
            profit_loss = "✅ 利益" if trade['profit'] else "❌ 損失"
            report += f"""
━━━ 取引 {i}: {trade['symbol']} ({profit_loss}) ━━━

🔵 エントリー情報:
  時刻: {trade['entry_time']}
  価格: {trade['entry_price']:,.0f}円
  株数: {trade['shares']:,}株
  投資額: {trade['entry_cost']:,.0f}円 (手数料込み)
  
  📊 エントリー根拠:
  {trade['entry_reason']}
  
  🎯 目標設定:
  利確目標: {trade['take_profit_price']:,.0f}円 (+{((trade['take_profit_price']/trade['entry_price']-1)*100):.1f}%)
  損切設定: {trade['stop_loss_price']:,.0f}円 ({((trade['stop_loss_price']/trade['entry_price']-1)*100):.1f}%)

🔴 エグジット情報:
  時刻: {trade['exit_time']}
  価格: {trade['exit_price']:,.0f}円
  売却額: {trade['exit_value']:,.0f}円 (手数料差引後)
  
  📊 エグジット理由:
  {trade['exit_reason']}: {trade['exit_condition']}
  
  📈 予測変化:
  エントリー時: {trade['entry_prediction']:.2f}%
  エグジット時: {trade['exit_prediction']:.2f}%

💰 取引結果:
  損益: {trade['pnl']:+,.0f}円
  利益率: {trade['pnl_pct']:+.2f}%
  保有時間: {(pd.to_datetime(trade['exit_time']) - pd.to_datetime(trade['entry_time'])).total_seconds()/60:.0f}分

📊 各モデル予測詳細:
  RandomForest  - Entry: {trade['entry_rf_pred']:+.2f}% | Exit: {trade['exit_rf_pred']:+.2f}%
  LinearRegression - Entry: {trade['entry_lr_pred']:+.2f}% | Exit: {trade['exit_lr_pred']:+.2f}%
  LightGBM      - Entry: {trade['entry_lgb_pred']:+.2f}% | Exit: {trade['exit_lgb_pred']:+.2f}%

"""
        
        # リスク分析を追加
        if performance['trades']:
            max_loss = min([t['pnl'] for t in performance['trades']])
            max_gain = max([t['pnl'] for t in performance['trades']])
            avg_holding_time = sum([
                (pd.to_datetime(t['exit_time']) - pd.to_datetime(t['entry_time'])).total_seconds()/60 
                for t in performance['trades']
            ]) / len(performance['trades'])
            
            report += f"""
📊 リスク分析:
最大利益: {max_gain:+,.0f}円
最大損失: {max_loss:+,.0f}円
平均保有時間: {avg_holding_time:.1f}分
リスク・リターン比: {max_gain/abs(max_loss) if max_loss != 0 else 'N/A'}

🎯 取引戦略の評価:
- ポジションサイズ: {self.position_size:.1%} (資本の{self.position_size:.0%})
- ストップロス: {self.stop_loss:.1%}
- 利確目標: {self.take_profit:.1%}
- 取引コスト: {self.transaction_cost:.1%}

📝 システムパフォーマンス:
- 木曜データでの機械学習モデル訓練
- 3つのモデル（RF, LR, LightGBM）のアンサンブル予測
- リアルタイム価格に基づく自動売買判定
- リスク管理による自動損切り・利確実行
"""
        
        return report
    
    def run_thursday_friday_demo(self, symbols: List[str] = None) -> Dict:
        """木曜→金曜デモを実行"""
        if symbols is None:
            symbols = ['7203', '6758', '8306', '9984', '6861']
        
        # 日付設定（現在の週の木曜・金曜）
        today = datetime.now()
        
        # 今週の木曜日と金曜日を計算
        days_since_monday = today.weekday()
        
        if days_since_monday >= 3:  # 木曜日以降
            thursday = today - timedelta(days=days_since_monday - 3)
            friday = today - timedelta(days=days_since_monday - 4)
        else:  # 月曜〜水曜日の場合は前週
            thursday = today - timedelta(days=days_since_monday + 4)
            friday = today - timedelta(days=days_since_monday + 3)
        
        thursday_date = thursday.strftime('%Y-%m-%d')
        friday_date = friday.strftime('%Y-%m-%d')
        
        self.logger.info(f"📅 デモ期間: 木曜({thursday_date}) → 金曜({friday_date})")
        
        try:
            # 1. 木曜のデータでモデル訓練
            trained_models = self.train_thursday_models(thursday_date, symbols)
            
            if not trained_models:
                self.logger.error("モデル訓練に失敗しました")
                return {}
            
            # 2. 金曜のデータで予測
            prediction_data = self.predict_friday_signals(friday_date, symbols, trained_models)
            
            if prediction_data.empty:
                self.logger.error("予測データの生成に失敗しました")
                return {}
            
            # 3. 取引シミュレーション
            performance = self.simulate_friday_trading(prediction_data)
            
            if not performance:
                self.logger.error("取引シミュレーションに失敗しました")
                return {}
            
            # 4. レポート生成
            report = self.generate_demo_report(thursday_date, friday_date, performance, symbols)
            
            # レポート保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.demo_reports_dir / f'thursday_friday_demo_{timestamp}.txt'
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"📊 レポート保存: {report_path}")
            print(report)
            
            return {
                'thursday_date': thursday_date,
                'friday_date': friday_date,
                'performance': performance,
                'report': report,
                'prediction_data': prediction_data,
                'trained_models_count': len(trained_models['models'])
            }
            
        except Exception as e:
            self.logger.error(f"デモ実行エラー: {e}")
            import traceback
            traceback.print_exc()
            return {}


def main():
    """メイン実行関数"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('thursday_friday_demo.log', encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 木曜→金曜 取引デモシステム 開始")
    logger.info("="*60)
    
    # デモ実行
    demo = ThursdayToFridayTradingDemo()
    symbols = ['7203', '6758', '8306', '9984', '6861']
    
    results = demo.run_thursday_friday_demo(symbols)
    
    if results:
        performance = results['performance']
        logger.info(f"\n✅ デモ完了!")
        logger.info(f"📈 総リターン: {performance['total_return_pct']:.2f}%")
        logger.info(f"🎯 勝率: {performance['win_rate']:.1%}")
        logger.info(f"💰 総損益: {performance['total_pnl']:,.0f}円")
        logger.info(f"📊 レポート: reports/thursday_friday_demo/ フォルダ内")
    else:
        logger.error("❌ デモに失敗しました")


if __name__ == "__main__":
    main()
