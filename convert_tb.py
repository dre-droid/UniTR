import pandas as pd
from tbparse import SummaryReader

# 1. Load the data
reader = SummaryReader('/home/it4i-andreaam/UniTR/output/nuscenes_models/unitr_ibot_fusion_v5.1/full_dataset_v5.3/tensorboard/events.out.tfevents.1777998844.acn37.karolina.it4i.cz')
df = reader.scalars

# 2. Filter for target tags (train/, train/diag/, meta_data/)
target_prefixes = ('train/', 'meta_data/')
tags = sorted([tag for tag in df['tag'].unique() if tag.startswith(target_prefixes)])

all_downsampled = []
summary_stats = []

for tag in tags:
    df_tag = df[df['tag'] == tag].copy()
    if df_tag.empty:
        continue
    
    # 3. Downsample (Take only ~100 points evenly spaced)
    n = max(1, len(df_tag) // 10)
    df_small = df_tag.iloc[::n].copy()
    all_downsampled.append(df_small)

    # 4. Get high-level stats for the summary
    stats = {
        "tag": tag,
        "start_value": df_tag['value'].iloc[0],
        "end_value": df_tag['value'].iloc[-1],
        "min_value": df_tag['value'].min(),
        "max_value": df_tag['value'].max(),
        "mean": df_tag['value'].mean(),
        "std": df_tag['value'].std(),
        "total_steps": df_tag['step'].max()
    }
    summary_stats.append(stats)

    # Add stats to the downsampled dataframe for CSV export
    df_small['mean'] = stats['mean']
    df_small['std'] = stats['std']
    all_downsampled.append(df_small)

# 5. Export a tiny version for the Agent
if all_downsampled:
    df_final = pd.concat(all_downsampled)
    # Include 'tag', 'mean', and 'std' in CSV
    df_final[['step', 'tag', 'value', 'mean', 'std']].to_csv('tiny_log.csv', index=False)

    print("Summary Stats for Agent:")
    df_stats = pd.DataFrame(summary_stats)
    print(df_stats.to_string(index=False))
    print(f"\nDownsampled data for {len(tags)} metrics saved to tiny_log.csv")
else:
    print("No matching metrics found with prefixes:", target_prefixes)
