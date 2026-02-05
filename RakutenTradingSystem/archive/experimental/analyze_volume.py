import pandas as pd

# prime_symbols.csvの詳細分析
df = pd.read_csv('prime_symbols.csv')

print("=== avg_volume 列の詳細分析 ===")
print(f"avg_volume の型: {df['avg_volume'].dtype}")
print(f"avg_volume の統計:")
print(df['avg_volume'].describe())

print(f"\n=== 30万株（300,000）以上の銘柄 ===")
high_volume = df[df['avg_volume'] >= 300000]
print(f"30万株以上の銘柄数: {len(high_volume)}")

print(f"\n=== 30万株未満の銘柄 ===")
low_volume = df[df['avg_volume'] < 300000]
print(f"30万株未満の銘柄数: {len(low_volume)}")
if len(low_volume) > 0:
    print("30万株未満の銘柄:")
    print(low_volume[['symbol', 'name', 'avg_volume']].to_string())

print(f"\n=== avg_volume の分布 ===")
print("出来高別の銘柄数:")
print(f"100万株以上: {len(df[df['avg_volume'] >= 1000000])}")
print(f"50万株以上: {len(df[df['avg_volume'] >= 500000])}")
print(f"30万株以上: {len(df[df['avg_volume'] >= 300000])}")
print(f"10万株以上: {len(df[df['avg_volume'] >= 100000])}")

print(f"\n=== 最高・最低出来高 ===")
max_volume = df.loc[df['avg_volume'].idxmax()]
min_volume = df.loc[df['avg_volume'].idxmin()]
print(f"最高出来高: {max_volume['symbol']} {max_volume['name']} - {max_volume['avg_volume']:,}株")
print(f"最低出来高: {min_volume['symbol']} {min_volume['name']} - {min_volume['avg_volume']:,}株")
