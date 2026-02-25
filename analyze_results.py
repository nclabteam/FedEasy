import os
import json
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict

def format_value(mean, std):
    if pd.isna(std):
        return f"{mean:.2f}"
    return f"{mean:.2f}±{std:.2f}"

def analyze_results(paths, strategy_filter=None, dataset_filter=None, alpha_filter=None):
    groups = {
        "strategy": [],
        "dataset": [],
        "dirichlet_alpha": [],
        "seed": [],
        "path": [],
    }

    # Collect data paths from directories
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist.")
            continue
            
        for root, dirs, files in os.walk(path):
            if "config.json" in files:
                config_path = os.path.join(root, "config.json")
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    
                    strategy = config["server"]["strategy"]
                    dataset = config["common"]["dataset"]
                    alpha = float(config["common"]["dirichlet_alpha"])
                    seed = config["common"].get("seed", "unknown")
                    
                    # Filters
                    if strategy_filter and strategy_filter.lower() not in strategy.lower():
                        continue
                    if dataset_filter and dataset_filter.lower() != dataset.lower():
                        continue

                    if alpha_filter is not None and abs(alpha - alpha_filter) > 1e-5:
                        continue
                    
                    # Find CSV matching expected naming convention
                    csv_files = [f for f in files if f.endswith(".csv")]
                    if not csv_files:
                        continue

                    expected_csv = f'{strategy}_{dataset}_{config["common"].get("data_type","")}_{config["client"].get("batch_size",")")}_{config["client"].get("lr","")}_{config["client"].get("epochs","")}.csv'
                    # prefer exact match
                    if expected_csv in csv_files:
                        csv_path = os.path.join(root, expected_csv)
                    else:
                        # try candidates that start with strategy_dataset_
                        prefix = f'{strategy}_{dataset}_'
                        candidates = [os.path.join(root, f) for f in csv_files if f.startswith(prefix)]
                        if candidates:
                            # pick the most recently modified candidate
                            csv_path = max(candidates, key=os.path.getmtime)
                            print(f"Using candidate CSV {csv_path} (expected {expected_csv} not found)")
                        else:
                            # fallback to first CSV and warn
                            csv_path = os.path.join(root, csv_files[0])
                            print(f"Warning: expected csv {expected_csv} not found; falling back to {csv_path}")
                    
                    groups["strategy"].append(strategy)
                    groups["dataset"].append(dataset)
                    groups["dirichlet_alpha"].append(alpha)
                    groups["seed"].append(seed)
                    groups["path"].append(csv_path)
                except Exception as e:
                    print(f"Failed to read config {config_path}: {e}")
                    continue

    df_meta = pd.DataFrame(groups)
  

    if df_meta.empty:
        print("No valid results found matching the filters.")
        return

    report = []

    # Group by experimental settings
    for (strategy, dataset, alpha), gdf in df_meta.groupby(["strategy", "dataset", "dirichlet_alpha"]):
        all_dfs = []
        seeds_found = sorted(gdf["seed"].unique())
        
        for csv_path in gdf["path"]:
            try:
                temp_df = pd.read_csv(csv_path)
                if temp_df.empty:
                    print(f"CSV is empty: {csv_path}")
                elif "accuracy" not in temp_df.columns:
                    print(f"Missing 'accuracy' column in {csv_path}; columns: {temp_df.columns.tolist()}")
                else:
                    all_dfs.append(temp_df)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

        if not all_dfs:
            print(f"No valid CSVs with accuracy found for strategy={strategy}, dataset={dataset}, alpha={alpha}")
            print(f"Tried paths: {gdf['path'].tolist()}")
            continue

        # Aggregate across seeds
        seed_metrics = []
        for df in all_dfs:
            metrics = {
                "accuracy": df["accuracy"].iloc[-1],
                "time": df["time"].sum() if "time" in df.columns else 0,
                "round": df["round"].max()
            }
            seed_metrics.append(metrics)
        
        sm_df = pd.DataFrame(seed_metrics)
        
        acc_m, acc_s = sm_df["accuracy"].mean() * 100, sm_df["accuracy"].std() * 100
        time_m, time_s = sm_df["time"].mean(), sm_df["time"].std()
        max_round = sm_df["round"].max()

        report.append({
            "Algorithm": strategy,
            "Dataset": dataset,
            "Alpha": alpha,
            "Last Acc": format_value(acc_m, acc_s),
            "Time (s)": format_value(time_m, time_s),
            "Round": int(max_round),
            "Seeds": ", ".join(map(str, seeds_found)),
            "Num_Seeds": len(all_dfs),
            "Mean_Val": acc_m
        })

    df_report = pd.DataFrame(report)

    if df_report.empty:
        print("No aggregated results to report.")
        return

    # Calculate column widths
    max_algo_len = max(df_report['Algorithm'].map(len).max(), len('Algorithm'))
    
    # Print the requested columns
    header_line = f"{'Algorithm':<{max_algo_len}} | {'Dataset':<10} | {'Alpha':<6} | {'Acc (%)':<12} | {'Time (s)':<12} | {'R'}"
    separator = f"{'-'*max_algo_len}-|-{'-'*10}-|-{'-'*6}-|-{'-'*12}-|-{'-'*12}-|-{'-'*2}"
    
    print(f"\n{'='*len(separator)}")
    print(header_line)
    print(separator)
    
    # Sort by Dataset, Alpha, then Accuracy
    df_sorted = df_report.sort_values(by=["Dataset", "Alpha", "Mean_Val"], ascending=[True, True, False])
    
    for _, row in df_sorted.iterrows():
        print(f"{row['Algorithm']:<{max_algo_len}} | {row['Dataset']:<10} | {row['Alpha']:<6} | {row['Last Acc']:<12} | {row['Time (s)']:<12} | {row['Round']}")

    # Save to CSV
    df_sorted.drop(columns=['Mean_Val']).to_csv("summary_results.csv", index=False)
    print(f"\nDetailed summary saved to summary_results.csv")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze FL experiment results.')
    parser.add_argument('--paths', type=str, nargs='+', help='Path(s) to the results directory (e.g., out/2026-02-09)')
    parser.add_argument('--strategy', type=str, help='Filter by strategy name')
    parser.add_argument('--dataset', type=str, help='Filter by dataset name')
    parser.add_argument('--alpha', type=float, help='Filter by Dirichlet alpha value')
    args = parser.parse_args()
    
    analyze_results(args.paths, strategy_filter=args.strategy, dataset_filter=args.dataset, alpha_filter=args.alpha)

