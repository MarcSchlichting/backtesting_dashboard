import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import hashlib
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

class IndexPerformanceAnalyzer:
    """
    Analyze constituent performance vs index with data caching and sliding window analysis
    """
    
    def __init__(self, cache_dir='data_cache'):
        """
        Initialize analyzer with cache directory
        
        Parameters:
        -----------
        cache_dir : str
            Directory to store cached data
        """
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_key(self, symbols, start_date, end_date):
        """Generate unique cache key based on symbols and date range"""
        symbols_str = ','.join(sorted(symbols))
        key_str = f"{symbols_str}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key):
        """Get full path to cache file"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _save_to_cache(self, data, cache_key):
        """Save data to cache"""
        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to cache: {cache_path}")
    
    def _load_from_cache(self, cache_key):
        """Load data from cache if available"""
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Data loaded from cache: {cache_path}")
            return data
        return None
    
    def load_symbols_from_file(self, filename):
        """
        Load stock symbols from a file (one symbol per line or comma-separated)
        """
        # with open(filename, 'r') as f:
        #     content = f.read()
        #     # Handle both newline-separated and comma-separated formats
        #     if ',' in content:
        #         symbols = [s.strip() for s in content.split(',')]
        #     else:
        #         symbols = [s.strip() for s in content.split('\n')]
        #     # Remove empty strings
        #     symbols = [s for s in symbols if s]
        # return symbols
        return pd.read_csv(filename, index_col='symbol').T.to_dict()
    
    def download_all_data(self, symbols_file, index_symbol, start_date, end_date, progress_bar, force_download=False):
        """
        Download all data upfront and cache it
        
        Parameters:
        -----------
        symbols_file : str
            Path to file containing constituent symbols
        index_symbol : str
            Symbol for the index (e.g., '^GSPC' for S&P 500)
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        force_download : bool
            If True, force re-download even if cache exists
            
        Returns:
        --------
        dict : Dictionary with 'index' and 'constituents' data
        """
        # Load symbols
        print(f"Loading symbols from {symbols_file}...")
        symbols = self.load_symbols_from_file(symbols_file)
        index_info = {"index":Path(symbols_file).stem, "data":symbols}
        print(f"Found {len(symbols)} symbols")
        
        # Generate cache key
        all_symbols = [index_info["index"]] + list(symbols.keys())
        cache_key = self._get_cache_key(all_symbols, start_date, end_date)
        
        # Check cache unless force download
        if not force_download:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        print(f"\nDownloading data from {start_date} to {end_date}...")
        
        # Download index data
        print(f"Downloading index data for {index_symbol}...")
        index_data = yf.download(index_symbol, start=start_date, end=end_date, progress=False)
        
        if index_data.empty:
            raise ValueError(f"No data found for index {index_symbol}")
        
        # Download constituent data
        print(f"\nDownloading constituent data...")
        constituents_data = {}
        failed_symbols = []
        
        for i, symbol in enumerate(tqdm(symbols, desc="Downloading stocks")):
            try:
                stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if stock_data.empty or len(stock_data) < 2:
                    print(f"\nWarning: Insufficient data for {symbol}, skipping...")
                    failed_symbols.append(symbol)
                    continue
                
                constituents_data[symbol] = stock_data
                
            except Exception as e:
                print(f"\nError downloading {symbol}: {str(e)}")
                failed_symbols.append(symbol)
                continue
            progress_bar.progress((i+1)/len(symbols))
        
        print(f"\n\nDownload complete!")
        print(f"Successfully downloaded: {len(constituents_data)} stocks")
        print(f"Failed: {len(failed_symbols)} stocks")
        if failed_symbols:
            print(f"Failed symbols: {', '.join(failed_symbols)}")
        
        # Package data
        data = {
            'index': index_data,
            'constituents': constituents_data,
            'index_info': index_info,
            'start_date': start_date,
            'end_date': end_date,
            'download_timestamp': datetime.now().isoformat()
        }
        
        # Save to cache
        self._save_to_cache(data, cache_key)
        
        return data
    
    def calculate_period_performance(self, prices, start_idx, end_idx):
        """
        Calculate percentage performance for a specific period
        
        Parameters:
        -----------
        prices : pd.Series or pd.DataFrame column
            Price data
        start_idx : int
            Start index
        end_idx : int
            End index
            
        Returns:
        --------
        float : Percentage return
        """
        # if isinstance(prices, pd.DataFrame):
        #     prices = prices['Close']
        
        start_price = prices.iloc[start_idx]
        end_price = prices.iloc[end_idx]
        
        return ((end_price - start_price) / start_price) * 100
    
    def analyze_sliding_window(self, data, window_days=90, step_days=30):
        """
        Analyze performance over sliding windows
        
        Parameters:
        -----------
        data : dict
            Data dictionary from download_all_data()
        window_days : int
            Window size in calendar days (default 90)
        step_days : int
            Step size in calendar days (default 30)
            
        Returns:
        --------
        list : List of DataFrames, one for each window
        """
        index_data = data['index']
        constituents_data = data['constituents']
        
        # Get date range
        all_dates = index_data.index
        start_date = all_dates[0]
        end_date = all_dates[-1]
        
        print(f"\nAnalyzing sliding windows:")
        print(f"Window size: {window_days} days")
        print(f"Step size: {step_days} days")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        results = []
        current_start = start_date
        
        while True:
            current_end = current_start + timedelta(days=window_days)
            
            # Break if we've gone past the end date
            if current_end > end_date:
                break
            
            # Find closest actual trading dates
            window_start_dates = all_dates[all_dates >= current_start]
            if len(window_start_dates) == 0:
                break
            window_start = window_start_dates[0]
            
            window_end_dates = all_dates[all_dates <= current_end]
            if len(window_end_dates) == 0:
                current_start += timedelta(days=step_days)
                continue
            window_end = window_end_dates[-1]
            
            # Skip if window is too short (less than half the desired window)
            if (window_end - window_start).days < window_days / 2:
                current_start += timedelta(days=step_days)
                continue
            
            # Calculate performance for this window
            try:
                window_result = self._analyze_window(
                    index_data, constituents_data, 
                    window_start, window_end
                )
                window_result['window_start'] = window_start
                window_result['window_end'] = window_end
                window_result['window_days'] = (window_end - window_start).days
                results.append(window_result)
            except Exception as e:
                print(f"Error analyzing window {window_start.date()} to {window_end.date()}: {str(e)}")
            
            # Move to next window
            current_start += timedelta(days=step_days)
        
        print(f"\nCompleted analysis of {len(results)} windows")
        return results
    
    def _analyze_window(self, index_data, constituents_data, start_date, end_date):
        """
        Analyze performance for a specific window
        """
        # Get index data for window
        index_window = index_data.xs(INDEX_SYMBOL, axis=1, level='Ticker')
        index_window = index_window.loc[start_date:end_date]
        if len(index_window) < 2:
            raise ValueError("Insufficient index data for window")
        
        index_performance = self.calculate_period_performance(
            index_window['Close'], 0, -1
        )
        
        # Analyze constituents
        results = []
        for symbol, stock_data in constituents_data.items():
            try:
                # Get stock data for window
                stock_window = stock_data.xs(symbol, axis=1, level='Ticker')
                stock_window = stock_window.loc[start_date:end_date]
                
                if len(stock_window) < 2:
                    continue
                
                stock_performance = self.calculate_period_performance(
                    stock_window['Close'], 0, -1
                )
                
                relative_performance = stock_performance - index_performance
                
                results.append({
                    'Symbol': symbol,
                    'Stock Performance (%)': stock_performance,
                    'Index Performance (%)': index_performance,
                    'Relative Performance (%)': relative_performance,
                    'Start Price': stock_window['Close'].iloc[0],
                    'End Price': stock_window['Close'].iloc[-1]
                })
                
            except Exception as e:
                continue
        
        # Create DataFrame and sort
        df = pd.DataFrame(results)
        df = df.sort_values('Relative Performance (%)', ascending=False)
        df = df.reset_index(drop=True)
        
        return df
    
    def plot_sliding_window_results(self, window_results, top_n=10, bottom_n=10):
        """
        Plot results from sliding window analysis
        """
        if not window_results:
            print("No results to plot")
            return
        
        # Extract data for plotting
        dates = [r['window_end'] for r in window_results]
        
        # Track top and bottom performers over time
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Index performance over time
        index_perf = [r.iloc[0]['Index Performance (%)'] if len(r) > 0 else 0 
                      for r in window_results]
        axes[0].plot(dates, index_perf, marker='o', linewidth=2, color='blue')
        axes[0].set_title(f'Index Performance ({window_results[0]["window_days"]} Day Windows)', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Performance (%)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot 2: Best performer relative performance over time
        best_perf = [r.iloc[0]['Relative Performance (%)'] if len(r) > 0 else 0 
                     for r in window_results]
        best_symbols = [r.iloc[0]['Symbol'] if len(r) > 0 else '' 
                       for r in window_results]
        axes[1].plot(dates, best_perf, marker='o', linewidth=2, color='green')
        axes[1].set_title('Best Performer Relative Performance', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Relative Performance (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot 3: Worst performer relative performance over time
        worst_perf = [r.iloc[-1]['Relative Performance (%)'] if len(r) > 0 else 0 
                      for r in window_results]
        worst_symbols = [r.iloc[-1]['Symbol'] if len(r) > 0 else '' 
                        for r in window_results]
        axes[2].plot(dates, worst_perf, marker='o', linewidth=2, color='red')
        axes[2].set_title('Worst Performer Relative Performance', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Relative Performance (%)', fontsize=12)
        axes[2].set_xlabel('Window End Date', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sliding_window_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSliding window chart saved as 'sliding_window_analysis.png'")
        plt.show()
    
    def export_sliding_window_results(self, window_results, filename='sliding_window_results.xlsx'):
        """
        Export sliding window results to Excel with multiple sheets
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for i, result in enumerate(window_results):
                if len(result) > 0:
                    summary_data.append({
                        'Window': i + 1,
                        'Start Date': result['window_start'],
                        'End Date': result['window_end'],
                        'Days': result['window_days'],
                        'Index Performance (%)': result.iloc[0]['Index Performance (%)'],
                        'Best Symbol': result.iloc[0]['Symbol'],
                        'Best Rel Perf (%)': result.iloc[0]['Relative Performance (%)'],
                        'Worst Symbol': result.iloc[-1]['Symbol'],
                        'Worst Rel Perf (%)': result.iloc[-1]['Relative Performance (%)'],
                        'Avg Rel Perf (%)': result['Relative Performance (%)'].mean(),
                        'Stocks Analyzed': len(result)
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual window sheets (limit to first 20 windows to avoid Excel limit)
            for i, result in enumerate(window_results[:20]):
                sheet_name = f"Window_{i+1}"
                result_export = result.drop(columns=['window_start', 'window_end', 'window_days'], errors='ignore')
                result_export.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\nResults exported to {filename}")
    
    def get_consistency_rankings(self, window_results, top_n=20):
        """
        Analyze which stocks consistently outperform across windows
        
        Parameters:
        -----------
        window_results : list
            List of window analysis results
        top_n : int
            Number of top consistent performers to return
            
        Returns:
        --------
        pd.DataFrame : Rankings based on consistency
        """
        # Track performance across all windows
        symbol_stats = {}
        
        for result in window_results:
            for _, row in result.iterrows():
                symbol = row['Symbol']
                rel_perf = row['Relative Performance (%)']
                
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {
                        'appearances': 0,
                        'outperformance_count': 0,
                        'total_rel_perf': 0,
                        'rel_perf_values': []
                    }
                
                symbol_stats[symbol]['appearances'] += 1
                symbol_stats[symbol]['total_rel_perf'] += rel_perf
                symbol_stats[symbol]['rel_perf_values'].append(rel_perf)
                
                if rel_perf > 0:
                    symbol_stats[symbol]['outperformance_count'] += 1
        
        # Calculate consistency metrics
        consistency_data = []
        for symbol, stats in symbol_stats.items():
            avg_rel_perf = stats['total_rel_perf'] / stats['appearances']
            outperform_rate = (stats['outperformance_count'] / stats['appearances']) * 100
            std_dev = np.std(stats['rel_perf_values'])
            
            consistency_data.append({
                'Symbol': symbol,
                'Avg Relative Perf (%)': avg_rel_perf,
                'Outperformance Rate (%)': outperform_rate,
                'Std Dev (%)': std_dev,
                'Windows Analyzed': stats['appearances'],
                'Times Outperformed': stats['outperformance_count'],
                'Consistency Score': avg_rel_perf * (outperform_rate / 100) - std_dev * 0.1
            })
        
        consistency_df = pd.DataFrame(consistency_data)
        consistency_df = consistency_df.sort_values('Consistency Score', ascending=False)
        consistency_df = consistency_df.reset_index(drop=True)
        
        return consistency_df

    def plot_single_period(self, df_results, top_n=10, bottom_n=10):
        """
        Plot top and bottom performers for a single period
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Top performers
        top = df_results.head(top_n)
        ax1.barh(range(len(top)), top['Relative Performance (%)'], color='green', alpha=0.7)
        ax1.set_yticks(range(len(top)))
        ax1.set_yticklabels(top['Symbol'])
        ax1.set_xlabel('Relative Performance (%)')
        ax1.set_title(f'Top {top_n} Performers vs Index')
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        
        # Bottom performers
        bottom = df_results.tail(bottom_n)
        ax2.barh(range(len(bottom)), bottom['Relative Performance (%)'], color='red', alpha=0.7)
        ax2.set_yticks(range(len(bottom)))
        ax2.set_yticklabels(bottom['Symbol'])
        ax2.set_xlabel('Relative Performance (%)')
        ax2.set_title(f'Bottom {bottom_n} Performers vs Index')
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        print("\nChart saved as 'performance_analysis.png'")
        plt.show()

    def print_summary(self, df_results):
        """
        Print summary statistics
        """
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print(f"\nTotal stocks analyzed: {len(df_results)}")
        print(f"Stocks outperforming index: {len(df_results[df_results['Relative Performance (%)'] > 0])}")
        print(f"Stocks underperforming index: {len(df_results[df_results['Relative Performance (%)'] < 0])}")
        print(f"\nAverage relative performance: {df_results['Relative Performance (%)'].mean():.2f}%")
        print(f"Median relative performance: {df_results['Relative Performance (%)'].median():.2f}%")
        print(f"Std dev of relative performance: {df_results['Relative Performance (%)'].std():.2f}%")
        print(f"\nBest performer: {df_results.iloc[0]['Symbol']} ({df_results.iloc[0]['Relative Performance (%)']:.2f}%)")
        print(f"Worst performer: {df_results.iloc[-1]['Symbol']} ({df_results.iloc[-1]['Relative Performance (%)']:.2f}%)")
        print("\n" + "="*80)


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = IndexPerformanceAnalyzer(cache_dir='data_cache')
    

    for months_pre in range(1,13):
        # Configuration
        SYMBOLS_FILE = "index_data/^GSPC.txt"
        INDEX_SYMBOL = "^GSPC"
        START_DATE = "2022-01-01"
        END_DATE = "2026-01-02"
        DELTA_MONTHS_PRE = months_pre
        DELTA_MONTHS_POST = 6
        n_top = 10
        
        # Download/load data (will use cache if available)
        print("="*80)
        print("STEP 1: DATA DOWNLOAD/LOADING")
        print("="*80)
        data = analyzer.download_all_data(
            symbols_file=SYMBOLS_FILE,
            index_symbol=INDEX_SYMBOL,
            start_date=START_DATE,
            end_date=END_DATE,
            force_download=False  # Set to True to force re-download
        )
        
        pre_rel_performances = []
        post_rel_performances = []

        
        ANALYSIS_START = datetime.strptime(START_DATE, "%Y-%m-%d").date()
        ANALYSIS_MIDDLE = ANALYSIS_START + relativedelta(months=DELTA_MONTHS_PRE)
        ANALYSIS_END = ANALYSIS_MIDDLE + relativedelta(months=DELTA_MONTHS_POST)
        
        while ANALYSIS_END <= datetime.strptime(END_DATE, "%Y-%m-%d").date():
            # # Print dates
            # print("ANALYSIS START: ", ANALYSIS_START)
            # print("ANALYSIS MIDDLE: ", ANALYSIS_MIDDLE)
            # print("ANALYSIS END: ", ANALYSIS_END)


            # Analysis from start to middle
            results_start_to_middle = analyzer._analyze_window(
                data['index'],
                data['constituents'],
                pd.Timestamp(ANALYSIS_START),
                pd.Timestamp(ANALYSIS_MIDDLE)
            )

            # Analysis from middle to end
            results_middle_to_end = analyzer._analyze_window(
                data['index'],
                data['constituents'],
                pd.Timestamp(ANALYSIS_MIDDLE),
                pd.Timestamp(ANALYSIS_END)
            )
            
            # Save single period results
            output_file = f"performance_analysis_{ANALYSIS_START}_to_{ANALYSIS_MIDDLE}.csv"
            results_start_to_middle.to_csv(output_file, index=False)

            output_file = f"performance_analysis_{ANALYSIS_MIDDLE}_to_{ANALYSIS_END}.csv"
            results_middle_to_end.to_csv(output_file, index=False)

            # Trends
            top_symbols = list(results_start_to_middle.head(n_top)["Symbol"])
            top_rel_per = list(results_start_to_middle.head(n_top)["Relative Performance (%)"])
            top_post_rel_per = list(results_middle_to_end[results_middle_to_end["Symbol"].isin(top_symbols)]["Relative Performance (%)"])

            pre_rel_performances.extend(top_rel_per)
            post_rel_performances.extend(top_post_rel_per)

            # Increment dates
            ANALYSIS_START = ANALYSIS_END - relativedelta(months=DELTA_MONTHS_PRE)
            ANALYSIS_MIDDLE = ANALYSIS_START + relativedelta(months=DELTA_MONTHS_PRE)
            ANALYSIS_END = ANALYSIS_MIDDLE + relativedelta(months=DELTA_MONTHS_POST)
        
        print(months_pre, np.mean(post_rel_performances))
    print("stop")
    
