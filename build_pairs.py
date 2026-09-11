"""Run with: python build_pairs.py
(Windows CMD can't handle multi-line -c "..." commands, so this is a
plain script instead — does the same thing.)"""
from src.ingest import load_brand_pairs

df = load_brand_pairs('data/real_twcs.csv', 'AppleSupport', top_level_only=True)
df.to_csv('data/real_apple_pairs.csv', index=False)
print(f"Done — {len(df)} pairs saved to data/real_apple_pairs.csv")
