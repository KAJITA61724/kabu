"""
フェーズ1統合システム - データ基盤完成版
- 5分足データ収集
- 移動平均計算
- 板情報収集
- テクニカル指標計算
- データ品質監視
- 統合監視ダッシュボード
"""

import sys
import os
import threading
import time
import logging
import json
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# 独自モジュールのインポート
from enhanced_data_collector import EnhancedDataCollector
from data_quality_monitor import DataQualityMonitor
from technical_indicators import TechnicalIndicators

class Phase1IntegratedSystem:
    """フェーズ1統合システム"""
    
    def __init__(self, config_file: str = "phase1_config.json"):
        self.config = self.load_config(config_file)
        self.running = False
        
        # コンポーネント初期化
        self.data_collector = EnhancedDataCollector(self.config['database']['path'])
        self.quality_monitor = DataQualityMonitor(self.config['database']['path'])
        self.technical_indicators = TechnicalIndicators(self.config['database']['path'])
        
        # ログ設定
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 月別ログディレクトリ
        monthly_log_dir = log_dir / datetime.now().strftime("%Y%m")
        monthly_log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(monthly_log_dir / 'phase1_system.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_file: str) -> Dict:
        """設定ファイル読み込み"""
        default_config = {
            "database": {
                "path": "enhanced_trading.db"
            },
            "data_collection": {
                "interval_minutes": 5,
                "symbols_file": "nikkei225_symbols.csv",
                "min_volume_threshold": 300000
            },
            "quality_monitoring": {
                "check_interval_minutes": 30,
                "quality_threshold": 80,
                "report_interval_hours": 6
            },
            "technical_indicators": {
                "calculation_interval_minutes": 15,
                "required_data_points": 50
            },
            "system": {
                "max_threads": 4,
                "market_hours": {
                    "start": "09:00",
                    "end": "15:00"
                }
            }
        }
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # デフォルト設定をユーザー設定で更新
            default_config.update(user_config)
            return default_config
            
        except FileNotFoundError:
            self.logger.warning(f"設定ファイルが見つかりません: {config_file}")
            self.logger.info("デフォルト設定を使用します")
            return default_config
    
    def is_market_hours(self) -> bool:
        """市場時間判定"""
        current_time = datetime.now().time()
        market_start = datetime.strptime(self.config['system']['market_hours']['start'], '%H:%M').time()
        market_end = datetime.strptime(self.config['system']['market_hours']['end'], '%H:%M').time()
        
        return market_start <= current_time <= market_end
    
    def is_weekday(self) -> bool:
        """平日判定"""
        return datetime.now().weekday() < 5  # 月曜日=0, 日曜日=6
    
    def should_collect_data(self) -> bool:
        """データ収集実行判定"""
        return self.is_market_hours() and self.is_weekday()
    
    def data_collection_job(self):
        """データ収集ジョブ"""
        if not self.should_collect_data():
            self.logger.info("市場時間外のためデータ収集をスキップ")
            return
        
        try:
            self.logger.info("データ収集開始")
            
            # Excel接続確認
            if not self.data_collector.connected:
                if not self.data_collector.initialize_excel_connection():
                    self.logger.error("Excel接続失敗")
                    return
            
            # 銘柄リスト読み込み
            if not self.data_collector.symbols:
                self.data_collector.load_nikkei225_symbols()
            
            # 5分足データ収集
            collected_data = self.data_collector.collect_5min_data()
            
            if collected_data:
                # データ保存
                self.data_collector.save_5min_data(collected_data)
                self.logger.info(f"データ収集完了: {len(collected_data)}銘柄")
            else:
                self.logger.warning("収集データが空です")
                
        except Exception as e:
            self.logger.error(f"データ収集エラー: {e}")
    
    def get_uptime_hours(self) -> float:
        """稼働時間取得"""
        if hasattr(self, 'start_time'):
            return (datetime.now() - self.start_time).total_seconds() / 3600
        return 0.0
    
    def start_system(self):
        """システム開始"""
        self.logger.info("=" * 60)
        self.logger.info("フェーズ1統合システム開始")
        self.logger.info("=" * 60)
        
        self.start_time = datetime.now()
        self.running = True
        
        # 初期化処理
        try:
            # データベース初期化
            self.data_collector.initialize_database()
            
            # Excel接続初期化
            if not self.data_collector.initialize_excel_connection():
                self.logger.error("Excel接続初期化失敗")
                return False
            
            # 銘柄リスト読み込み
            self.data_collector.load_nikkei225_symbols()
            
            # 業界フラグ初期化
            self.initialize_industry_flags()
            
            # スケジューラー設定
            self.setup_scheduler()
            
            # 初回実行
            self.logger.info("初回実行開始")
            self.data_collection_job()
            self.technical_indicators_job()
            self.quality_monitoring_job()
            
            self.logger.info("初期化完了")
            
        except Exception as e:
            self.logger.error(f"初期化エラー: {e}")
            return False
        
        # メインループ
        self.logger.info("メインループ開始")
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # 1分ごとにチェック
                
        except KeyboardInterrupt:
            self.logger.info("キーボード割り込み検出")
            self.stop_system()
            
        except Exception as e:
            self.logger.error(f"メインループエラー: {e}")
            self.stop_system()
        
        return True
    
    def stop_system(self):
        """システム停止"""
        self.logger.info("システム停止処理開始")
        
        self.running = False
        
        # 最終レポート生成
        try:
            self.generate_comprehensive_report()
        except Exception as e:
            self.logger.error(f"最終レポート生成エラー: {e}")
        
        # Excel接続クリーンアップ
        try:
            if self.data_collector.excel_app:
                self.data_collector.excel_app.Quit()
        except Exception as e:
            self.logger.error(f"Excel終了エラー: {e}")
        
        uptime = self.get_uptime_hours()
        self.logger.info(f"システム停止完了 (稼働時間: {uptime:.2f}時間)")
        self.logger.info("=" * 60)
    
    def initialize_industry_flags(self):
        """業界フラグ初期化"""
        try:
            import sqlite3
            
            # 業界マッピング読み込み
            with open('industry_mapping.json', 'r', encoding='utf-8') as f:
                industry_data = json.load(f)
            
            conn = sqlite3.connect(self.config['database']['path'])
            cursor = conn.cursor()
            
            # 業界フラグテーブル更新
            for symbol, info in industry_data['industry_mapping'].items():
                cursor.execute('''
                    INSERT OR REPLACE INTO industry_flags 
                    (symbol, industry_name, industry_code, sector_name, 
                    is_nikkei225, market_cap_category, avg_volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    info['industry_name'],
                    info['industry_code'],
                    info['sector_name'],
                    info['is_nikkei225'],
                    info['market_cap_category'],
                    info['avg_volume']
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info("業界フラグ初期化完了")
            
        except Exception as e:
            self.logger.error(f"業界フラグ初期化エラー: {e}")
    
    def get_system_status(self) -> Dict:
        """システム状態取得"""
        try:
            # 基本状態
            status = {
                'running': self.running,
                'uptime_hours': self.get_uptime_hours(),
                'market_status': 'open' if self.should_collect_data() else 'closed',
                'current_time': datetime.now().isoformat()
            }
            
            # データ状態
            if self.data_collector.symbols:
                status['symbols_count'] = len(self.data_collector.symbols)
                status['excel_connected'] = self.data_collector.connected
            
            # 最新品質情報
            try:
                quality_report = self.quality_monitor.generate_quality_report()
                status['quality_score'] = quality_report['quality_score']
                status['quality_status'] = quality_report['status']
            except:
                status['quality_score'] = 0
                status['quality_status'] = 'unknown'
            
            # パフォーマンス指標
            try:
                metrics = self.quality_monitor.get_performance_metrics()
                status['total_records'] = metrics['total_records']
                status['data_size_mb'] = metrics['data_size_mb']
            except:
                status['total_records'] = 0
                status['data_size_mb'] = 0
            
            return status
            
        except Exception as e:
            self.logger.error(f"システム状態取得エラー: {e}")
            return {'error': str(e)}
    
    def print_status_dashboard(self):
        """ステータスダッシュボード表示"""
        status = self.get_system_status()
        
        print("\n" + "=" * 80)
        print("フェーズ1統合システム - ステータスダッシュボード")
        print("=" * 80)
        
        print(f"システム状態: {'🟢 稼働中' if status['running'] else '🔴 停止中'}")
        print(f"稼働時間: {status['uptime_hours']:.2f}時間")
        print(f"市場状態: {'🟢 開場中' if status['market_status'] == 'open' else '🔴 閉場中'}")
        print(f"現在時刻: {status['current_time']}")
        
        print("\n📊 データ状況:")
        print(f"対象銘柄数: {status.get('symbols_count', 0)}")
        print(f"Excel接続: {'🟢 接続中' if status.get('excel_connected', False) else '🔴 未接続'}")
        print(f"総レコード数: {status.get('total_records', 0):,}")
        print(f"データサイズ: {status.get('data_size_mb', 0):.2f} MB")
        
        print("\n🔍 品質状況:")
        quality_emoji = {'good': '🟢', 'warning': '🟡', 'poor': '🔴'}.get(status.get('quality_status', 'unknown'), '⚪')
        print(f"品質スコア: {quality_emoji} {status.get('quality_score', 0)}/100")
        print(f"品質状態: {status.get('quality_status', 'unknown')}")
        
        print("=" * 80)
    
    def run_interactive_mode(self):
        """インタラクティブモード実行"""
        self.logger.info("インタラクティブモード開始")
        
        while True:
            print("\n📋 フェーズ1統合システム メニュー")
            print("1. システム開始")
            print("2. システム停止")
            print("3. ステータス表示")
            print("4. 手動データ収集")
            print("5. 品質チェック")
            print("6. テクニカル指標計算")
            print("7. 包括的レポート生成")
            print("8. 設定表示")
            print("9. 終了")
            
            choice = input("\n選択してください (1-9): ").strip()
            
            if choice == '1':
                if not self.running:
                    threading.Thread(target=self.start_system, daemon=True).start()
                    print("✅ システムを開始しました")
                else:
                    print("⚠️ システムは既に稼働中です")
            
            elif choice == '2':
                if self.running:
                    self.stop_system()
                    print("✅ システムを停止しました")
                else:
                    print("⚠️ システムは既に停止中です")
            
            elif choice == '3':
                self.print_status_dashboard()
            
            elif choice == '4':
                print("📥 手動データ収集を開始します...")
                self.data_collection_job()
                print("✅ データ収集完了")
            
            elif choice == '5':
                print("🔍 品質チェックを開始します...")
                self.quality_monitoring_job()
                print("✅ 品質チェック完了")
            
            elif choice == '6':
                print("📈 テクニカル指標計算を開始します...")
                self.technical_indicators_job()
                print("✅ テクニカル指標計算完了")
            
            elif choice == '7':
                print("📋 包括的レポートを生成します...")
                self.generate_comprehensive_report()
                print("✅ レポート生成完了")
            
            elif choice == '8':
                print("\n⚙️ 現在の設定:")
                print(json.dumps(self.config, indent=2, ensure_ascii=False))
            
            elif choice == '9':
                if self.running:
                    self.stop_system()
                print("👋 終了します")
                break
            
            else:
                print("❌ 無効な選択です")

def create_default_config():
    """デフォルト設定ファイル作成"""
    config = {
        "database": {
            "path": "enhanced_trading.db"
        },
        "data_collection": {
            "interval_minutes": 5,
            "symbols_file": "nikkei225_symbols.csv",
            "min_volume_threshold": 300000
        },
        "quality_monitoring": {
            "check_interval_minutes": 30,
            "quality_threshold": 80,
            "report_interval_hours": 6
        },
        "technical_indicators": {
            "calculation_interval_minutes": 15,
            "required_data_points": 50
        },
        "system": {
            "max_threads": 4,
            "market_hours": {
                "start": "09:00",
                "end": "15:00"
            }
        }
    }
    
    with open('phase1_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    def quality_monitoring_job(self):
        """品質監視ジョブ"""
        try:
            self.logger.info("品質監視開始")
            
            # 品質レポート生成
            report = self.quality_monitor.generate_quality_report()
            
            # 品質レポート保存
            self.quality_monitor.save_quality_report(report)
            
            # 品質アラート判定
            if report['quality_score'] < self.config['quality_monitoring']['quality_threshold']:
                self.logger.warning(f"品質アラート: スコア {report['quality_score']}")
                self.send_quality_alert(report)
            
            self.logger.info(f"品質監視完了: スコア {report['quality_score']}")
            
        except Exception as e:
            self.logger.error(f"品質監視エラー: {e}")
    
    def technical_indicators_job(self):
        """テクニカル指標計算ジョブ"""
        try:
            self.logger.info("テクニカル指標計算開始")
            
            # 全銘柄の指標計算
            self.technical_indicators.calculate_and_save_all_symbols()
            
            # 指標サマリー取得
            summary = self.technical_indicators.get_indicator_summary()
            
            self.logger.info(f"テクニカル指標計算完了: {summary['total_symbols']}銘柄")
            
        except Exception as e:
            self.logger.error(f"テクニカル指標計算エラー: {e}")
    
    def cleanup_job(self):
        """クリーンアップジョブ"""
        try:
            self.logger.info("データクリーンアップ開始")
            
            # 古いデータ削除
            self.data_collector.cleanup_old_data()
            
            self.logger.info("データクリーンアップ完了")
            
        except Exception as e:
            self.logger.error(f"データクリーンアップエラー: {e}")
    
    def send_quality_alert(self, report: Dict):
        """品質アラート送信"""
        alert_message = f"""
        データ品質アラート
        
        時刻: {report['generated_at']}
        品質スコア: {report['quality_score']}/100
        ステータス: {report['status']}
        
        詳細:
        - データ完全性: {report['completeness']['avg_completeness']:.1f}%
        - 異常値数: {report['anomalies']['total_anomalies']}件
        - 新鮮度: {report['freshness']['fresh_symbols']}/{report['freshness']['total_symbols']}銘柄
        """
        
        # ここで実際のアラート送信処理を実装
        # 例: メール送信、Slack通知など
        self.logger.warning(alert_message)
    
    def setup_scheduler(self):
        """スケジューラー設定"""
        # データ収集（5分ごと）
        schedule.every(self.config['data_collection']['interval_minutes']).minutes.do(
            self.data_collection_job
        )
        
        # 品質監視（30分ごと）
        schedule.every(self.config['quality_monitoring']['check_interval_minutes']).minutes.do(
            self.quality_monitoring_job
        )
        
        # テクニカル指標計算（15分ごと）
        schedule.every(self.config['technical_indicators']['calculation_interval_minutes']).minutes.do(
            self.technical_indicators_job
        )
        
        # クリーンアップ（1時間ごと）
        schedule.every().hour.do(self.cleanup_job)
        
        # 品質レポート生成（6時間ごと）
        schedule.every(self.config['quality_monitoring']['report_interval_hours']).hours.do(
            self.generate_comprehensive_report
        )
        
        self.logger.info("スケジューラー設定完了")
    
    def generate_comprehensive_report(self):
        """包括的レポート生成"""
        try:
            self.logger.info("包括的レポート生成開始")
            
            # 各種データ取得
            quality_report = self.quality_monitor.generate_quality_report()
            performance_metrics = self.quality_monitor.get_performance_metrics()
            indicator_summary = self.technical_indicators.get_indicator_summary()
            
            # 統合レポート作成
            comprehensive_report = {
                'generated_at': datetime.now().isoformat(),
                'system_status': 'running' if self.running else 'stopped',
                'market_status': 'open' if self.should_collect_data() else 'closed',
                'data_quality': quality_report,
                'performance_metrics': performance_metrics,
                'technical_indicators': indicator_summary,
                'system_info': {
                    'config': self.config,
                    'uptime_hours': self.get_uptime_hours(),
                    'total_symbols': len(self.data_collector.symbols) if self.data_collector.symbols else 0
                }
            }
            
            # レポート保存
            reports_dir = Path("reports/quality")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = reports_dir / f"comprehensive_report_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"包括的レポート生成完了: {report_file}")
            
        except Exception as e:
            self.logger.error(f"包括的レポート生成エラー: {e}")