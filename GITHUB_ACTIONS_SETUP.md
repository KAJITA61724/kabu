# 🤖 GitHub Actions自動データ収集セットアップ

PCを閉じていても、GitHub Actionsが自動でデータ収集します。

## ✅ すでに設定済み

`.github/workflows/data-collection.yml` が作成されています。

## 🚀 有効化方法

### 1. GitHubにプッシュ
```bash
cd /workspaces/kabu
git add .github/workflows/data-collection.yml
git add RakutenTradingSystem/data_collection/
git commit -m "GitHub Actionsでデータ収集自動化"
git push
```

### 2. GitHub Actionsを有効化
1. GitHubリポジトリ → **Actions**タブ
2. 「I understand my workflows, go ahead and enable them」をクリック
3. 左メニューから「自動データ収集」を選択
4. 「Enable workflow」をクリック

### 3. 動作確認
- **Actions**タブで実行履歴を確認
- 緑色のチェックマーク = 成功
- 赤いバツマーク = エラー（ログで詳細確認）

## 📅 実行スケジュール

**平日（月～金）の9:00～15:00（日本時間）に5分おきに自動実行**

- 週末・祝日は自動スキップ
- PC不要
- Codespace不要
- 完全自動

## 💾 データの保存場所

実行ごとに：
- Artifactとして自動保存（90日間保存）
- Actions → 実行履歴 → Artifacts からダウンロード可能

## 💰 無料枠

GitHub Actionsの無料枠：
- パブリックリポジトリ: **無制限**
- プライベートリポジトリ: **月2,000分**

このワークフローは1回約1分なので、月2,000回実行可能（十分すぎる）。

## 🔍 手動実行

緊急時やテスト時：
1. Actions → 「自動データ収集」
2. 「Run workflow」ボタン
3. 「Run workflow」を確認

## ⚠️ 注意点

- 初回実行時は空のDBから開始
- データは実行ごとに蓄積
- Artifactから最新DBをダウンロードして使用

## 🎯 利点

✅ PC不要  
✅ Codespace不要  
✅ 完全自動  
✅ 無料  
✅ 信頼性が高い  
✅ ログで確認可能
