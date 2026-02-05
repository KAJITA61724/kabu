# 機械学習モデル統合システム

## 概要
このシステムは、yfinanceを使用して株価データを収集し、複数の機械学習モデルを比較して最適な予測モデルを選択する統合システムです。

## 主な機能

### 1. データ収集
- yfinanceAPIを使用して日本株の5分足データを自動収集
- SQLiteデータベースに保存
- 重複データを自動的に除去

### 2. 特徴量エンジニアリング
- 24の高度な特徴量を自動生成
- 価格変動、移動平均、ボラティリティ、出来高、RSI、ラグ特徴量
- 時間系特徴量（時間帯による市場特性）

### 3. 複数モデル比較
- **RandomForest**: アンサンブル学習モデル
- **LinearRegression**: 線形回帰モデル
- **SimpleMovingAverage**: 移動平均ベースライン
- **LightGBM**: 勾配ブースティング（オプション）

### 4. 性能評価
- MSE (Mean Squared Error): 平均二乗誤差
- MAE (Mean Absolute Error): 平均絶対誤差
- R² (決定係数): 回帰の説明力
- 各モデルの性能ランキング

### 5. 自動化機能
- データ収集からモデル評価まで一気通貫
- 結果レポートの自動生成
- モデルとスケーラーの自動保存

## 使用方法

### 基本実行
```bash
python ml_complete_system.py
```

### 実行結果
1. **データ収集**: 各銘柄の5分足データを取得
2. **特徴量作成**: 24の特徴量を自動生成
3. **モデル比較**: 複数モデルの性能を比較
4. **レポート生成**: 各銘柄の詳細分析レポート
5. **モデル保存**: 最適化されたモデルを保存

## 出力ファイル

### レポートファイル (`reports/`)
- `{銘柄}_comparison_report.txt`: 各銘柄の詳細比較レポート

### モデルファイル (`models/`)
- `{銘柄}_RandomForest_model.pkl`: RandomForestモデル
- `{銘柄}_LinearRegression_model.pkl`: 線形回帰モデル
- `{銘柄}_scaler.pkl`: データスケーラー

### ログファイル
- `ml_analysis.log`: 実行ログ

## 分析結果の例

### 性能比較（MAE順）
```
1. LinearRegression
   MSE: 69.9866
   MAE: 4.2881
   R²: 0.9073

2. RandomForest
   MSE: 156.8508
   MAE: 10.2415
   R²: 0.7923

3. SimpleMovingAverage
   MSE: 1902.6531
   MAE: 34.7782
   R²: -1.5197
```

### 価格予測
```
現在価格: 3574.00
予測価格: 3573.43
予測変動: -0.57 (-0.02%)
```

## 必要な依存関係
- pandas
- numpy
- scikit-learn
- yfinance
- sqlite3
- lightgbm (オプション)

## インストール
```bash
pip install pandas numpy scikit-learn yfinance lightgbm
```

## 対象銘柄
- 7203: トヨタ自動車
- 6758: ソニーグループ
- 8306: 三菱UFJフィナンシャル・グループ
- 9984: ソフトバンクグループ
- 6861: キーエンス

## 特徴量一覧
1. **価格系**: price_change, high_low_ratio, open_close_ratio
2. **移動平均**: sma_5, sma_10, sma_ratio
3. **ボラティリティ**: volatility_5, volatility_10
4. **出来高**: volume_ratio, volume_price_ratio
5. **テクニカル指標**: rsi
6. **ラグ特徴量**: close_lag_1~5, volume_lag_1~5, change_lag_1~5
7. **時間系**: hour, minute, time_of_day

## システムの特徴
- ✅ 完全自動化（データ収集→分析→レポート生成）
- ✅ 複数モデルの性能比較
- ✅ 詳細なログ記録
- ✅ エラーハンドリング
- ✅ ファイル自動保存
- ✅ 結果の可視化レポート

## 注意事項
- yfinanceAPIの利用制限に注意
- LightGBMは別途インストールが必要
- 市場時間外は最新データが取得できない場合があります

## 今後の拡張可能性
- 他の機械学習アルゴリズムの追加
- リアルタイム予測機能
- Webインターフェースの追加
- バックテスト機能の実装
- アラート機能の追加
