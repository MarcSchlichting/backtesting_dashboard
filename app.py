import streamlit as st
import bt
import pandas as pd
import numpy as np
from utils import IndexPerformanceAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Portfolio Backtesting Tool",
    page_icon="",
    layout="wide"
)

# ===========================
# CUSTOM WEIGHT FUNCTION
# ===========================
def custom_weight_function(prices, lookback_months, index_info):
    """
    Your custom function to compute portfolio weights.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Historical price data (exactly lookback_months of history)
        
    Returns:
    --------
    pd.Series
        Weights for each asset (should sum to 1.0)
    """
    n = st.session_state.get('top_n_stocks', 10)

    end_date = prices.index[-1]
    added_dates = (pd.DataFrame(index_info["data"]).T)["date_added"]
    added_dates = pd.to_datetime(added_dates)
    filter = added_dates <= end_date
    filtered_prices = prices[filter[filter].index]
    start_prices = filtered_prices.iloc[0]
    end_prices = filtered_prices.iloc[-1]
    performances_perc = ((end_prices-start_prices) / start_prices) * 100
    sorted_performances = performances_perc.sort_values(ascending=False)

    top_n_symbols = sorted_performances.index[:n]
    per_index_weight = 1/n
    
    weights = pd.Series(0., index=prices.columns)
    weights[top_n_symbols] = per_index_weight
    
    return weights


# ===========================
# CUSTOM ALGO CLASS
# ===========================
class CustomWeightingAlgo(bt.Algo):
    """
    Algorithm that computes custom weights based on historical data.
    Only runs if sufficient history is available.
    """
    def __init__(self, weight_function, lookback_months, start_date, index_info):
        super(CustomWeightingAlgo, self).__init__()
        self.weight_function = weight_function
        self.lookback_months = lookback_months
        self.start_date = start_date
        self.index_info = index_info
    
    def __call__(self, target):
        all_prices = target.universe.loc[:target.now]
        lookback_date = target.now - pd.DateOffset(months=self.lookback_months)

        if target.now < pd.Timestamp(self.start_date):
            return False
        
        if lookback_date < target.universe.index[0]:
            return False
        
        prices = all_prices.loc[lookback_date:]
        weights = self.weight_function(prices, self.lookback_months, self.index_info)
        target.temp['weights'] = weights
        
        return True


# ===========================
# BACKTEST FUNCTION
# ===========================
# @st.cache_data
def run_backtest(start_date,
                price_data, 
                index_info,
                 rebalance_frequency='quarterly', 
                 lookback_months=12,
                 transaction_cost=0.001):
    """
    Run backtest with custom rebalancing strategy.
    """
    index_symbol = index_info["index"]
    non_index_tickers = [t for t in list(price_data.keys()) if t != index_symbol]
    
    # Select rebalancing algorithm
    if rebalance_frequency == 'monthly':
        rebalance_algo = bt.algos.RunMonthly(run_on_first_date=True)
    elif rebalance_frequency == 'quarterly':
        rebalance_algo = bt.algos.RunQuarterly(run_on_first_date=True)
    else:  # yearly
        rebalance_algo = bt.algos.RunYearly(run_on_first_date=True)
    
    # Define the strategy
    strategy = bt.Strategy(
        'CustomRebalanceStrategy',
        [
            rebalance_algo,
            bt.algos.SelectThese(non_index_tickers),
            CustomWeightingAlgo(custom_weight_function, lookback_months, start_date, index_info),
            bt.algos.Rebalance(),
        ]
    )
    
    index_baseline = bt.Strategy(
        'IndexBaseline',
        [bt.algos.RunAfterDate(start_date),
         bt.algos.SelectThese([index_symbol]),
         bt.algos.WeighEqually(),
         bt.algos.Rebalance(),]
    )
    
    def commission_fun(q,p):
        return abs(p*q)*transaction_cost

    # Create backtest with transaction costs
    backtest = bt.Backtest(
        strategy,
        price_data,
        commissions=lambda q, p: abs(p*q)*transaction_cost,
        integer_positions=False,
        additional_data={"index_info":index_info}
    )

    backtest_index = bt.Backtest(
        index_baseline,
        price_data,
        commissions=lambda q, p: 0,
        integer_positions=False,
        additional_data={"index_info":index_info}
    )
    
    # Run the backtest
    result = bt.run(backtest, backtest_index)
    
    return result


# ===========================
# STREAMLIT UI
# ===========================
def main():
    st.title("Portfolio Backtesting Tool")
    st.markdown("---")
    
    # Sidebar for parameters
    st.sidebar.header("Backtest Parameters")
    
    # Index selection
    index_file = st.sidebar.text_input(
        "Index File Path",
        value="./index_data/^GSPC.txt",
        help="Path to the index data file"
    )
    
    index_symbol = st.sidebar.text_input(
        "Index Symbol",
        value="^GSPC",
        help="Symbol of the index to backtest against"
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        data_start_date = st.date_input(
            "Data Start Date",
            value=pd.to_datetime("2021-01-01"),
            help="Start date for downloading historical data"
        )
    with col2:
        data_end_date = st.date_input(
            "Data End Date",
            value=pd.to_datetime("2026-01-02"),
            help="End date for downloading historical data"
        )
    
    backtest_start_date = st.sidebar.date_input(
        "Backtest Start Date",
        value=pd.to_datetime("2022-01-01"),
        help="Start date for the backtest (must be after data start date)"
    )
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Settings")
    
    top_n_stocks = st.sidebar.slider(
        "Top N Stocks",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Number of top-performing stocks to hold"
    )
    st.session_state['top_n_stocks'] = top_n_stocks
    
    lookback_months = st.sidebar.slider(
        "Lookback Period (months)",
        min_value=1,
        max_value=24,
        value=6,
        help="Number of months to look back for performance calculation"
    )
    
    rebalance_frequency = st.sidebar.selectbox(
        "Rebalance Frequency",
        options=['monthly', 'quarterly', 'yearly'],
        index=1,
        help="How often to rebalance the portfolio"
    )
    
    transaction_cost = st.sidebar.number_input(
        "Transaction Cost (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.35,
        step=0.01,
        help="Transaction cost as a percentage"
    ) / 100
    
    # Run backtest button
    run_button = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)
    
    # Main content area
    if run_button:
        # try:
            # with st.spinner("Loading data..."):
            #     # Load data
            #     analyzer = IndexPerformanceAnalyzer()
            #     raw_data = analyzer.download_all_data(
            #         index_file,
            #         index_symbol,
            #         str(data_start_date),
            #         str(data_end_date)
            #     )
                
            #     # Format data for bt
            #     prices_dict = {}
            #     for t in raw_data["constituents"]:
            #         prices_dict[t] = raw_data["constituents"][t].xs(t, axis=1, level='Ticker')["Close"]
            #     prices_dict[raw_data["index_info"]["index"]] = raw_data["index"].xs(
            #         raw_data["index_info"]["index"],
            #         axis=1,
            #         level='Ticker'
            #     )["Close"]
            #     prices = pd.DataFrame(prices_dict)
                
            #     st.success("Data loaded successfully!")
        progress_bar = st.progress(0)

        try:
            # Load data
            progress_bar.progress(0)
            analyzer = IndexPerformanceAnalyzer()
            
            progress_bar.progress(0)
            raw_data = analyzer.download_all_data(
                index_file,
                index_symbol,
                str(data_start_date),
                str(data_end_date),
                progress_bar=progress_bar
            )
            
            # Format data for bt
            prices_dict = {}
            for t in raw_data["constituents"]:
                prices_dict[t] = raw_data["constituents"][t].xs(t, axis=1, level='Ticker')["Close"]
            
            prices_dict[raw_data["index_info"]["index"]] = raw_data["index"].xs(
                raw_data["index_info"]["index"],
                axis=1,
                level='Ticker'
            )["Close"]
            prices = pd.DataFrame(prices_dict)
            
            st.success("Data loaded successfully!")
            
            import time
            time.sleep(0.5)
            progress_bar.empty()
                

            
            with st.spinner("Running backtest..."):
                # Run backtest
                result = run_backtest(
                    start_date=str(backtest_start_date),
                    price_data=prices,
                    index_info=raw_data["index_info"],
                    rebalance_frequency=rebalance_frequency,
                    lookback_months=lookback_months,
                    transaction_cost=transaction_cost
                )
                
                st.success("Backtest completed!")
            
            # Display results
            st.markdown("---")
            st.header("Backtest Results")
            
            # Performance chart
            st.subheader("Portfolio Performance")
            fig = go.Figure()
            
            for strategy_name in result.prices.columns:
                fig.add_trace(go.Scatter(
                    x=result.prices.index,
                    y=result.prices[strategy_name],
                    mode='lines',
                    name=strategy_name,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Strategy Performance Comparison",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                hovermode='x unified',
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance statistics
            st.subheader("Performance Statistics")
            
            stats_df = result.stats
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Custom Strategy**")
                custom_stats = stats_df['CustomRebalanceStrategy']
                st.metric("Total Return", f"{custom_stats['total_return']:.2%}")
                st.metric("CAGR", f"{custom_stats['cagr']:.2%}")
                st.metric("Sharpe Ratio", f"{custom_stats['daily_sharpe']:.2f}")
                st.metric("Max Drawdown", f"{custom_stats['max_drawdown']:.2%}")
                st.metric("Volatility", f"{custom_stats['daily_vol']:.2%}")
            
            with col2:
                st.markdown("**Index Baseline**")
                index_stats = stats_df['IndexBaseline']
                st.metric("Total Return", f"{index_stats['total_return']:.2%}")
                st.metric("CAGR", f"{index_stats['cagr']:.2%}")
                st.metric("Sharpe Ratio", f"{index_stats['daily_sharpe']:.2f}")
                st.metric("Max Drawdown", f"{index_stats['max_drawdown']:.2%}")
                st.metric("Volatility", f"{index_stats['daily_vol']:.2%}")
            
            # Full statistics table
            with st.expander("View Full Statistics Table"):
                st.dataframe(stats_df, use_container_width=True)
            
            # Transactions
            st.subheader("Rebalancing Transactions")
            
            transactions = result.get_transactions()
            
            if not transactions.empty:
                transactions_dict = {
                    str(pd.Timestamp(key).date()): group 
                    for key, group in transactions.groupby(level=0)
                }
                
                st.write(f"**Total Rebalancing Events:** {len(transactions_dict)}")
                
                # Create tabs for each rebalancing date
                if len(transactions_dict) > 0:
                    # Get sorted list of dates
                    rebalance_dates = sorted(transactions_dict.keys())
                    
                    # Create tabs
                    tabs = st.tabs([f"{date}" for date in rebalance_dates])
                    
                    # Display transactions for each date in its respective tab
                    for tab, date in zip(tabs, rebalance_dates):
                        with tab:
                            trans_df = transactions_dict[date].copy()
                            
                            # Summary metrics for this rebalancing event
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                num_trades = len(trans_df)
                                st.metric("Number of Trades", num_trades)
                            
                            with col2:
                                total_value = abs(trans_df['quantity'] * trans_df['price']).sum()
                                st.metric("Total Trade Value", f"${total_value:,.2f}")
                            
                            with col3:
                                buys = trans_df[trans_df['quantity'] > 0]
                                num_buys = len(buys)
                                st.metric("Buys", num_buys)
                            
                            with col4:
                                sells = trans_df[trans_df['quantity'] < 0]
                                num_sells = len(sells)
                                st.metric("Sells", num_sells)
                            
                            st.markdown("---")
                            
                            # Separate buys and sells
                            if num_buys > 0:
                                st.markdown("##### Buy Orders")
                                buy_df = buys.copy()
                                buy_df['Value'] = buy_df['quantity'] * buy_df['price']
                                buy_df = buy_df.sort_values('Value', ascending=False)
                                
                                # Format the dataframe
                                buy_display = buy_df.copy()
                                buy_display['quantity'] = buy_display['quantity'].apply(lambda x: f"{x:.2f}")
                                buy_display['price'] = buy_display['price'].apply(lambda x: f"${x:.2f}")
                                buy_display['Value'] = buy_display['Value'].apply(lambda x: f"${x:,.2f}")
                                
                                st.dataframe(
                                    buy_display[['quantity', 'price', 'Value']],
                                    use_container_width=True
                                )
                            
                            if num_sells > 0:
                                st.markdown("##### Sell Orders")
                                sell_df = sells.copy()
                                sell_df['Value'] = abs(sell_df['quantity'] * sell_df['price'])
                                sell_df = sell_df.sort_values('Value', ascending=False)
                                
                                # Format the dataframe
                                sell_display = sell_df.copy()
                                sell_display['quantity'] = sell_display['quantity'].apply(lambda x: f"{abs(x):.2f}")
                                sell_display['price'] = sell_display['price'].apply(lambda x: f"${x:.2f}")
                                sell_display['Value'] = sell_display['Value'].apply(lambda x: f"${x:,.2f}")
                                
                                st.dataframe(
                                    sell_display[['quantity', 'price', 'Value']],
                                    use_container_width=True
                                )
                            
        
            
                    
            else:
                st.info("No transactions recorded during the backtest period.")
            
            # Drawdown chart
            st.subheader("Drawdown Analysis")
            
            drawdown_fig = go.Figure()
            
            for strategy_name in result.prices.columns:
                # Calculate drawdown
                prices = result.prices[strategy_name]
                cummax = prices.cummax()
                drawdown = (prices - cummax) / cummax
                
                drawdown_fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown * 100,
                    mode='lines',
                    name=strategy_name,
                    fill='tozeroy',
                    line=dict(width=2)
                ))
            
            drawdown_fig.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode='x unified',
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(drawdown_fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            st.exception(e)
    
    else:
        # Welcome message
        st.info("Configure your backtest parameters in the sidebar and click 'Run Backtest' to begin.")
        


if __name__ == "__main__":
    main()