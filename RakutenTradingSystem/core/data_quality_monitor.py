"""
データ品質監視システム
- 5分足データの品質チェック
- 異常値検出
- データ欠損監視
- パフォーマンス監視
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import warnings

class DataQualityMonitor:
    """データ品質監視クラス"""
    
    def __init__(self, db_path: str = "enhanced_trading.db"):
        self.db_path = db_path
        self.quality_thresholds = {
            'min_data_points_per_hour': 10,  # 1時間あたり最低データポイント数
            'max_price_change_rate': 0.15,   # 最大価格変動率（15%）
            'min_volume_threshold': 1000,    # 最小出来高
            'max_spread_rate': 0.05,         # 最大スプレッド率（5%）
            'data_freshness_minutes': 10     # データ新鮮度（10分以内）
        }
        
    def check_data_completeness(self, hours_back: int = 24) -> Dict:
        """データ完全性チェック"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        conn = sqlite3.connect(self.db_path)
        
        # 期待されるデータポイント数（5分間隔）
        expected_points = hours_back * 12
        
        query = '''
            SELECT 
                symbol,
                COUNT(*) as actual_points,
                MIN(timestamp) as first_timestamp,
                MAX(timestamp) as last_timestamp,
                COUNT(CASE WHEN close_price > 0 THEN 1 END) as valid_prices
            FROM five_minute_data
            WHERE timestamp >= ?
            GROUP BY symbol
        '''
        
        df = pd.read_sql_query(query, conn, params=(cutoff_time,))
        conn.close()
        
        # 完全性スコア計算
        df['completeness_score'] = (df['actual_points'] / expected_points) * 100
        df['price_validity_score'] = (df['valid_prices'] / df['actual_points']) * 100
        
        # 問題のある銘柄を特定
        problematic_symbols = df[
            (df['completeness_score'] < 80) | 
            (df['price_validity_score'] < 95)
        ].copy()
        
        result = {
            'total_symbols': len(df),
            'avg_completeness': df['completeness_score'].mean(),
            'avg_price_validity': df['price_validity_score'].mean(),
            'problematic_symbols': len(problematic_symbols),
            'symbol_details': df.to_dict('records'),
            'issues': problematic_symbols.to_dict('records')
        }
        
        return result
    
    def detect_price_anomalies(self, hours_back: int = 24) -> List[Dict]:
        """価格異常値検出"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                symbol,
                timestamp,
                close_price,
                volume,
                LAG(close_price, 1) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_price
            FROM five_minute_data
            WHERE timestamp >= ?
            ORDER BY symbol, timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=(cutoff_time,))
        conn.close()
        
        if df.empty:
            return []
        
        # 価格変動率計算
        df['price_change_rate'] = (df['close_price'] - df['prev_price']) / df['prev_price']
        
        # 異常値検出
        anomalies = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if len(symbol_data) < 10:
                continue
            
            # 統計的異常値検出
            price_changes = symbol_data['price_change_rate'].dropna()
            
            if len(price_changes) == 0:
                continue
            
            # IQR方法での異常値検出
            q1 = price_changes.quantile(0.25)
            q3 = price_changes.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = symbol_data[
                (symbol_data['price_change_rate'] < lower_bound) |
                (symbol_data['price_change_rate'] > upper_bound)
            ]
            
            for _, row in outliers.iterrows():
                anomalies.append({
                    'symbol': row['symbol'],
                    'timestamp': row['timestamp'],
                    'price': row['close_price'],
                    'prev_price': row['prev_price'],
                    'change_rate': row['price_change_rate'],
                    'volume': row['volume'],
                    'anomaly_type': 'price_outlier'
                })
        
        return anomalies
    
    def check_volume_anomalies(self, hours_back: int = 24) -> List[Dict]:
        """出来高異常値検出"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                symbol,
                timestamp,
                volume,
                close_price
            FROM five_minute_data
            WHERE timestamp >= ? AND volume > 0
            ORDER BY symbol, timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=(cutoff_time,))
        conn.close()
        
        if df.empty:
            return []
        
        anomalies = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if len(symbol_data) < 10:
                continue
            
            # 出来高の統計的異常値検出
            volumes = symbol_data['volume']
            
            # 対数変換（出来高は正規分布に従わないため）
            log_volumes = np.log(volumes + 1)
            
            mean_log_vol = log_volumes.mean()
            std_log_vol = log_volumes.std()
            
            # 3シグマルールで異常値検出
            threshold = 3.0
            
            outliers = symbol_data[
                (log_volumes < mean_log_vol - threshold * std_log_vol) |
                (log_volumes > mean_log_vol + threshold * std_log_vol)
            ]
            
            for _, row in outliers.iterrows():
                anomalies.append({
                    'symbol': row['symbol'],
                    'timestamp': row['timestamp'],
                    'volume': row['volume'],
                    'price': row['close_price'],
                    'avg_volume': np.exp(mean_log_vol),
                    'anomaly_type': 'volume_outlier'
                })
        
        return anomalies
    
    def check_spread_anomalies(self, hours_back: int = 24) -> List[Dict]:
        """スプレッド異常値検出"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                o.symbol,
                o.timestamp,
                o.bid_price_1,
                o.ask_price_1,
                o.bid_ask_spread,
                f.close_price
            FROM order_book o
            JOIN five_minute_data f ON o.symbol = f.symbol AND o.timestamp = f.timestamp
            WHERE o.timestamp >= ? 
            AND o.bid_price_1 > 0 AND o.ask_price_1 > 0
            ORDER BY o.symbol, o.timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=(cutoff_time,))
        conn.close()
        
        if df.empty:
            return []
        
        # スプレッド率計算
        df['spread_rate'] = df['bid_ask_spread'] / df['close_price']
        
        anomalies = []
        
        # 異常なスプレッドを検出
        high_spread = df[df['spread_rate'] > self.quality_thresholds['max_spread_rate']]
        
        for _, row in high_spread.iterrows():
            anomalies.append({
                'symbol': row['symbol'],
                'timestamp': row['timestamp'],
                'bid_price': row['bid_price_1'],
                'ask_price': row['ask_price_1'],
                'spread': row['bid_ask_spread'],
                'spread_rate': row['spread_rate'],
                'price': row['close_price'],
                'anomaly_type': 'high_spread'
            })
        
        return anomalies
    
    def check_data_freshness(self) -> Dict:
        """データ新鮮度チェック"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                symbol,
                MAX(timestamp) as last_update,
                COUNT(*) as total_records
            FROM five_minute_data
            GROUP BY symbol
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {'status': 'no_data', 'symbols': []}
        
        current_time = datetime.now()
        freshness_threshold = timedelta(minutes=self.quality_thresholds['data_freshness_minutes'])
        
        df['last_update'] = pd.to_datetime(df['last_update'])
        df['minutes_since_update'] = (current_time - df['last_update']).dt.total_seconds() / 60
        df['is_fresh'] = df['minutes_since_update'] <= self.quality_thresholds['data_freshness_minutes']
        
        stale_symbols = df[~df['is_fresh']].copy()
        
        result = {
            'status': 'ok' if len(stale_symbols) == 0 else 'stale_data',
            'total_symbols': len(df),
            'fresh_symbols': len(df[df['is_fresh']]),
            'stale_symbols': len(stale_symbols),
            'avg_minutes_since_update': df['minutes_since_update'].mean(),
            'stale_symbol_details': stale_symbols.to_dict('records')
        }
        
        return result
    
    def generate_quality_report(self) -> Dict:
        """品質レポート生成"""
        logging.info("データ品質レポート生成開始")
        
        # 各種チェック実行
        completeness = self.check_data_completeness()
        price_anomalies = self.detect_price_anomalies()
        volume_anomalies = self.check_volume_anomalies()
        spread_anomalies = self.check_spread_anomalies()
        freshness = self.check_data_freshness()
        
        # 総合スコア計算
        quality_score = (
            completeness['avg_completeness'] * 0.3 +
            completeness['avg_price_validity'] * 0.2 +
            (100 - len(price_anomalies)) * 0.2 +
            (100 - len(volume_anomalies)) * 0.1 +
            (100 - len(spread_anomalies)) * 0.1 +
            (freshness['fresh_symbols'] / max(freshness['total_symbols'], 1) * 100) * 0.1
        )
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'quality_score': round(quality_score, 2),
            'status': 'good' if quality_score >= 80 else 'warning' if quality_score >= 60 else 'poor',
            'completeness': completeness,
            'freshness': freshness,
            'anomalies': {
                'price_anomalies': len(price_anomalies),
                'volume_anomalies': len(volume_anomalies),
                'spread_anomalies': len(spread_anomalies),
                'total_anomalies': len(price_anomalies) + len(volume_anomalies) + len(spread_anomalies)
            },
            'detailed_anomalies': {
                'price': price_anomalies[:10],  # 最新10件
                'volume': volume_anomalies[:10],
                'spread': spread_anomalies[:10]
            }
        }
        
        logging.info(f"品質レポート生成完了 - スコア: {quality_score:.2f}")
        return report
    
    def save_quality_report(self, report: Dict, file_path: str = None):
        """品質レポート保存"""
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"quality_report_{timestamp}.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            logging.info(f"品質レポート保存完了: {file_path}")
            
        except Exception as e:
            logging.error(f"品質レポート保存エラー: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """パフォーマンス指標取得"""
        conn = sqlite3.connect(self.db_path)
        
        queries = {
            'total_records': 'SELECT COUNT(*) FROM five_minute_data',
            'unique_symbols': 'SELECT COUNT(DISTINCT symbol) FROM five_minute_data',
            'data_size_mb': '''
                SELECT 
                    (COUNT(*) * 
                     (LENGTH(symbol) + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8)) / 1024.0 / 1024.0 as size_mb
                FROM five_minute_data
            ''',
            'avg_records_per_symbol': '''
                SELECT AVG(record_count) FROM (
                    SELECT COUNT(*) as record_count 
                    FROM five_minute_data 
                    GROUP BY symbol
                )
            '''
        }
        
        metrics = {}
        for metric_name, query in queries.items():
            try:
                result = conn.execute(query).fetchone()
                metrics[metric_name] = result[0] if result else 0
            except Exception as e:
                logging.error(f"パフォーマンス指標取得エラー {metric_name}: {e}")
                metrics[metric_name] = 0
        
        conn.close()
        
        return metrics

# 使用例
if __name__ == "__main__":
    monitor = DataQualityMonitor()
    
    # 品質レポート生成
    report = monitor.generate_quality_report()
    
    # レポート保存
    monitor.save_quality_report(report)
    
    # パフォーマンス指標表示
    metrics = monitor.get_performance_metrics()
    
    print("=" * 50)
    print("データ品質レポート")
    print("=" * 50)
    print(f"品質スコア: {report['quality_score']}/100 ({report['status']})")
    print(f"データ完全性: {report['completeness']['avg_completeness']:.1f}%")
    print(f"価格有効性: {report['completeness']['avg_price_validity']:.1f}%")
    print(f"異常値総数: {report['anomalies']['total_anomalies']}件")
    print(f"データ新鮮度: {report['freshness']['fresh_symbols']}/{report['freshness']['total_symbols']} 銘柄")
    print("\nパフォーマンス指標:")
    print(f"総レコード数: {metrics['total_records']:,}")
    print(f"銘柄数: {metrics['unique_symbols']}")
    print(f"データサイズ: {metrics['data_size_mb']:.2f} MB")
    print(f"銘柄あたり平均レコード数: {metrics['avg_records_per_symbol']:.0f}")
    print("=" * 50)