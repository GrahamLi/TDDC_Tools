import argparse
import os
import re
from typing import List, Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt

def is_line_flat(series: pd.Series, threshold: float = 0.01) -> bool:
    """Determines if a line is 'flat' based on its standard deviation relative to its mean."""
    if series.dropna().empty:
        return False
    
    mean_val = series.mean()
    if pd.isna(mean_val):
        return False
        
    std_val = series.std()
    if pd.isna(std_val):
        return False
    
    # If the line is essentially a constant zero, it's flat.
    if abs(mean_val) < 1e-9 and std_val < 1e-9:
        return True
        
    # Main condition: standard deviation is a small fraction of the mean value.
    # Avoid division by zero if mean is very small.
    if abs(mean_val) > 1e-9 and (std_val / abs(mean_val)) < threshold:
        return True
        
    return False

def parse_custom_bins(bins_str: str) -> List[Tuple[float, float]]:
    """Parses a custom bin string like '0-30,30-100,>1000' into a list of tuples."""
    bins: List[Tuple[float, float]] = []
    for part in bins_str.split(","):
        part = part.strip()
        if not part: continue
        if part.startswith(">"):
            lo = float(part[1:])
            bins.append((lo, float("inf")))
        else:
            lo, hi = part.split("-")
            bins.append((float(lo), float(hi)))
    return bins

def get_label_to_cols_map(scheme: str, price: float, custom_bins_str: str, all_cols: List[str]) -> Tuple[Dict[str, List[str]], str]:
    """Determines which original columns map to which new aggregated group label."""
    
    def parse_col_range(label: str) -> Tuple[float, float]:
        """Helper to parse share range from a column name string."""
        nums = [float(s.replace(",", "")) for s in re.findall(r"[\d,]+", label)]
        if len(nums) == 2: return nums[0], nums[1]
        if len(nums) == 1:
            if ">" in label or "以上" in label: return nums[0], float("inf")
            return nums[0], nums[0]
        return 0.0, 0.0

    label_to_cols: Dict[str, List[str]] = {}
    scheme_name = scheme

    if scheme == "shares":
        scheme_name = "General_Definition"
        bins = {
            "散戶 (1-400K)": (1, 400000),
            "中實戶 (400K-1M)": (400001, 1000000),
            "大戶 (>1M)": (1000001, float("inf"))
        }
        label_to_cols = {label: [] for label in bins}
        for col in all_cols:
            lo, hi = parse_col_range(col)
            if lo == 0: continue
            # Find which bin the majority of the column range falls into
            col_mid = (lo + hi) / 2 if hi != float("inf") else lo * 1.5
            for label, (bin_lo, bin_hi) in bins.items():
                if bin_lo <= col_mid < bin_hi:
                    label_to_cols[label].append(col)
                    break
    
    elif scheme == "amount":
        scheme_name = "Amount_Definition"
        if price <= 0: raise ValueError("--price is required for amount scheme")
        bins = {
            "< 5M": (0, 5_000_000),
            "5M-10M": (5_000_000, 10_000_000),
            "10M-30M": (10_000_000, 30_000_000),
            "> 30M": (30_000_000, float("inf"))
        }
        label_to_cols = {label: [] for label in bins}
        for col in all_cols:
            lo, hi = parse_col_range(col)
            mid_shares = (lo + hi) / 2 if hi != float("inf") else lo * 1.5
            amount = mid_shares * price
            for label, (bin_lo, bin_hi) in bins.items():
                if bin_lo <= amount < bin_hi:
                    label_to_cols[label].append(col)
                    break

    elif scheme == "custom":
        scheme_name = "Custom_Definition"
        custom_bins = parse_custom_bins(custom_bins_str)
        if not custom_bins: raise ValueError("--custom-bins required for custom scheme")
        labels = [f"{int(lo)}-{int(hi) if hi != float('inf') else 'inf'}" for lo, hi in custom_bins]
        label_to_cols = {label: [] for label in labels}
        for col in all_cols:
            lo, hi = parse_col_range(col)
            mid_shares = (lo + hi) / 2 if hi != float("inf") else lo * 1.5
            for (bin_lo, bin_hi), label in zip(custom_bins, labels):
                if bin_lo <= mid_shares < bin_hi:
                    label_to_cols[label].append(col)
                    break

    return label_to_cols, scheme_name

def aggregate_data(df: pd.DataFrame, label_to_cols: Dict[str, List[str]]) -> pd.DataFrame:
    """Aggregates a DataFrame based on the provided column mapping."""
    frames = []
    for label, cols in label_to_cols.items():
        if not cols or not all(c in df.columns for c in cols):
            continue
        sub = df[cols].sum(axis=1)
        frames.append(sub.rename(label))
    if not frames: return pd.DataFrame(index=df.index)
    return pd.concat(frames, axis=1)

def plot_and_save(df: pd.DataFrame, price_data: pd.Series, volume_data: pd.Series, title: str, path: str, ylabel: str):
    """Plots the aggregated data and saves it to a file, creating a detail plot for flat lines."""
    if df.empty: return

    # --- Main Plot (with Price and Volume) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1}
    )

    # Top plot: Aggregated data + Price
    for col in df.columns:
        ax1.plot(df.index, df[col], label=str(col), marker='.', markersize=5, linestyle='-')
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.grid(True)

    handles1, labels1 = ax1.get_legend_handles_labels()

    if not price_data.empty:
        ax_price = ax1.twinx()
        ax_price.plot(price_data.index, price_data, color='darkgray', alpha=0.7, linestyle='--', label='Weekly Close Price')
        ax_price.set_ylabel("Close Price", fontsize=12)
        ax_price.grid(False)
        handles2, labels2 = ax_price.get_legend_handles_labels()
        # Combine legends from both y-axes
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=9)
    else:
        ax1.legend(handles1, labels1, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=9)

    # Bottom plot: Volume
    if not volume_data.empty:
        ax2.bar(volume_data.index, volume_data, color='lightgray', width=1.0, label='Volume')
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.tick_params(axis='x', labelrotation=45)
    ax2.grid(True)

    fig.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout to make space for legend
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved main chart to: {path}")

    # --- Detail Plot (for flat lines only, without price/volume overlay) ---
    flat_cols = [col for col in df.columns if is_line_flat(df[col])]
    if flat_cols:
        print(f"Detected {len(flat_cols)} near-flat lines. Creating detail chart...")
        fig_detail, ax_detail = plt.subplots(figsize=(15, 8))
        for col in flat_cols:
            ax_detail.plot(df.index, df[col], label=str(col), marker='.', markersize=4, linestyle='-')
        
        ax_detail.set_title(f"{title} (Detail View)", fontsize=16)
        ax_detail.set_ylabel(ylabel, fontsize=12)
        ax_detail.set_xlabel("Date", fontsize=12)
        ax_detail.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
        ax_detail.tick_params(axis='x', labelrotation=45)
        ax_detail.grid(True)
        fig_detail.tight_layout()
        
        detail_path = path.replace('.png', '_screenIn.png')
        plt.savefig(detail_path, dpi=150)
        plt.close(fig_detail)
        print(f"Saved detail chart to: {detail_path}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate TDCC data and create trend charts with price/volume overlay.")
    parser.add_argument("--base", required=True, help="Base directory containing the 'data' folder.")
    parser.add_argument("--input", required=True, help="Path to the Excel file from Program 2.")
    parser.add_argument("--scheme", choices=["shares", "amount", "custom"], default="shares", help="Aggregation scheme.")
    parser.add_argument("--price", type=float, default=0.0, help="Stock price, required for 'amount' scheme.")
    parser.add_argument("--custom-bins", type=str, default="", help="Custom bins for 'custom' scheme, e.g., '0-1000,1001-10000,>10000'")
    args = parser.parse_args()

    out_dir = os.path.join(args.base, "output", "trends")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(args.base, "data", "trends_aggregated")
    os.makedirs(data_dir, exist_ok=True)

    try:
        xls = pd.ExcelFile(args.input)
        people = pd.read_excel(xls, sheet_name="People", index_col=0)
        shares = pd.read_excel(xls, sheet_name="Shares", index_col=0)
        ratio = pd.read_excel(xls, sheet_name="RatioPct", index_col=0)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input}")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Extract price and volume data before modifications
    if '周收盤價' in people.index and '周成交量' in people.index:
        price_data = people.loc['周收盤價']
        volume_data = people.loc['周成交量']
        # The column names are dates as strings, convert them to datetime
        price_data.index = pd.to_datetime(price_data.index, format='%Y%m%d')
        volume_data.index = pd.to_datetime(volume_data.index, format='%Y%m%d')
        print("Found weekly price and volume data.")
    else:
        price_data = pd.Series(dtype=float)
        volume_data = pd.Series(dtype=float)
        print("Weekly price and volume data not found in input file.")

    dfs = {"People": people, "Shares": shares, "RatioPct": ratio}
    for name, df in dfs.items():
        # Drop non-data rows before transposing
        rows_to_drop = ['周收盤價', '周成交量']
        df.drop(rows_to_drop, errors='ignore', inplace=True)
        
        # Convert column headers (dates) to datetime objects
        df.columns = pd.to_datetime(df.columns, format='%Y%m%d')
        
        # Transpose so that dates become the index
        dfs[name] = df.transpose()

    people, shares, ratio = dfs["People"], dfs["Shares"], dfs["RatioPct"]

    label_to_cols, scheme_name = get_label_to_cols_map(args.scheme, args.price, args.custom_bins, people.columns.astype(str))

    agg_people = aggregate_data(people, label_to_cols)
    agg_shares = aggregate_data(shares, label_to_cols)
    agg_ratio = aggregate_data(ratio, label_to_cols)

    ticker_stub = os.path.splitext(os.path.basename(args.input))[0]
    out_xlsx = os.path.join(data_dir, f"{ticker_stub}_{scheme_name}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        agg_people.to_excel(writer, sheet_name="Agg_People")
        agg_shares.to_excel(writer, sheet_name="Agg_Shares")
        agg_ratio.to_excel(writer, sheet_name="Agg_RatioPct")
    print(f"Saved aggregated Excel to: {out_xlsx}")

    # Plotting with price and volume overlays
    plot_and_save(agg_people, price_data, volume_data, f"{ticker_stub} People Trend ({scheme_name})", os.path.join(out_dir, f"{ticker_stub}_{scheme_name}_people.png"), "People")
    plot_and_save(agg_shares, price_data, volume_data, f"{ticker_stub} Shares Trend ({scheme_name})", os.path.join(out_dir, f"{ticker_stub}_{scheme_name}_shares.png"), "Shares")
    plot_and_save(agg_ratio, price_data, volume_data, f"{ticker_stub} Ratio Trend ({scheme_name})", os.path.join(out_dir, f"{ticker_stub}_{scheme_name}_ratio.png"), "Ratio (%)")

    print(f"\nProcessing complete.")

if __name__ == "__main__":
    main()
