import pandas as pd

# prime_symbols.csvの内容確認
df = pd.read_csv('prime_symbols.csv')
print(f'全銘柄数: {len(df)}')
print(f'30万株以上の銘柄: {len(df[df["avg_volume"] >= 300000])}')
print('最初の5銘柄:')
print(df.head())

# 銘柄リストを表示
filtered_df = df[df['avg_volume'] >= 300000]
symbols = filtered_df['symbol'].astype(str).tolist()
print(f'\n30万株以上の銘柄リスト（最初の10個）: {symbols[:10]}')
print(f'total symbols: {len(symbols)}')
