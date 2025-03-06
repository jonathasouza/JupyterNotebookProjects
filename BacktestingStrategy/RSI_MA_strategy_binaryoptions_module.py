from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class TradeConfig:
    """Configuration parameters for the trading strategy"""
    initial_capital: float
    base_bet_size: float
    rsi_period: int
    rsi_ma_period: int
    risk_per_trade_pct: float
    option_return_pct: float

class TradingStrategy:
    def __init__(self, config: TradeConfig):
        self.config = config
        self.current_capital = config.initial_capital
        self.trades_history = []
        
    def calculate_position_size(self, price: float) -> float:
        """Calculate dynamic position size based on risk per trade"""
        risk_amount = self.current_capital * (self.config.risk_per_trade_pct / 100) # amount I can lose
        position_size_in_dollars = risk_amount  # Define your position size in dollars
        return position_size_in_dollars

    def process_month(self, df: pd.DataFrame) -> dict:
        """Process a month of trading data"""
        if df.empty:
            logging.warning("Empty dataframe received for processing")
            return None

        # No need to calculate RSI here since BinanceTrader already has that function
        
        # Initialize position tracking
        position = 0
        entry_price = 0
        monthly_trades = []
        
        # Add shifted columns for crossover detection
        df['RSI_prev'] = df['RSI'].shift(1)
        df['RSI_MA_prev'] = df['RSI_MA'].shift(1)
        
        # Process signals
        for idx, row in df.iterrows():
            position_size = self.calculate_position_size(row['close']) # Basically not doing anything

            # So the trades df range doesn't get out of df range
            if idx == df.index.values[-1:]:
                break
            # Process new signals
            if position == 0 and (row['RSI'] > row['RSI_MA'] and row['RSI_prev'] < row['RSI_MA_prev']):
                
                monthly_trades.append({
                    'timestamp': idx,
                    'price': row['close'],
                    'quantity': position_size,
                    'type': 'entrada_comprado'
                })
                monthly_trades.append({
                    'timestamp': idx + timedelta(minutes=5),
                    'price': 0.0,
                    'quantity': position_size,
                    'type': 'saida_comprado'
                })
                position = 1
                
            elif position > 0 and (row['RSI'] < row['RSI_MA'] and row['RSI_prev'] > row['RSI_MA_prev']):
                monthly_trades.append({
                    'timestamp': idx,
                    'price': row['close'],
                    'quantity': position_size,
                    'type': 'entrada_vendido'
                })
                monthly_trades.append({
                    'timestamp': idx + timedelta(minutes=5),
                    'price': 0.0,
                    'quantity': position_size,
                    'type': 'saida_vendido'
                })
                position = 0

        return self.calculate_monthly_metrics(monthly_trades, df)

    def calculate_monthly_metrics(self, trades, df) -> dict:
        """Calculate performance metrics for the month"""
        if not trades:
            return {
                'return': 0,
                'trades': 0,
                'win_rate': 0,
                'avg_trade_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }

        trades_df = pd.DataFrame(trades)        

        # Fixing Exit Prices
        # Temporarily set 'timestamp' as the index for trades_df
        trades_df.set_index('timestamp', inplace=True)
        # Identify rows in trades_df where price == 0
        rows_to_update = trades_df['price'] == 0.0

        # Replace these rows with the corresponding values from df
        trades_df.loc[rows_to_update, 'price'] = df.loc[trades_df.index[rows_to_update], 'close']
        # Reset the index back to 'timestamp'
        trades_df.reset_index(inplace=True)

        
        # Shift prices for calculating trade PnL
        trades_df['prev_price'] = trades_df['price'].shift(1)
        trades_df['price_change'] = trades_df['price']-trades_df['prev_price']
        
        # PnL calculation (using dollar-based positions)
        # Ensure required columns exist
        trades_df['pnl'] = 0.0  # Initialize PnL column
    
        # Handle 'saida_comprado' type trades
        trades_df.loc[
            (trades_df['type'] == 'saida_comprado') & (trades_df['price_change'] > 0), # Given these conditions
            'pnl' # Populate this column
        ] = trades_df['quantity'] * (self.config.option_return_pct / 100)
    
        trades_df.loc[
            (trades_df['type'] == 'saida_comprado') & (trades_df['price_change'] < 0),
            'pnl'
        ] = -trades_df['quantity']
    
        # Handle 'saida_vendido' type trades
        trades_df.loc[
            (trades_df['type'] == 'saida_vendido') & (trades_df['price_change'] < 0),
            'pnl'
        ] = trades_df['quantity'] * (self.config.option_return_pct / 100)
    
        trades_df.loc[
            (trades_df['type'] == 'saida_vendido') & (trades_df['price_change'] > 0),
            'pnl'
        ] = -trades_df['quantity']

        # Verify pnl validity
        if 'pnl' not in trades_df.columns or trades_df['pnl'].isna().all():
            raise ValueError("Invalid 'pnl' column in trades_df.")
        
        # Total return
        total_return = trades_df['pnl'].sum()
    
        # Number of trades
        num_trades = len(trades_df[trades_df['type'].isin(['saida_comprado', 'saida_vendido'])])
    
        # Winning trades (PnL > 0)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
    
        # Updating the current capital after trades
        self.current_capital += total_return
    
        # Win rate (percentage of trades with positive PnL)
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
    
        # Average return per trade
        avg_trade_return = total_return / num_trades if num_trades > 0 else 0
    
        # Calculate Sharpe ratio based on daily PnL
        # Resample PnL to daily intervals and calculate the Sharpe ratio
        daily_returns = trades_df.set_index('timestamp')['pnl'].resample('D').sum()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 1 else 0
    
        # Calculate maximum drawdown
        cumulative_pnl = trades_df['pnl'].cumsum()
        max_drawdown = self.calculate_max_drawdown(cumulative_pnl)
    
        return {
            'return': total_return,
            'trades': num_trades,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
        """Calculate the maximum drawdown from peak"""
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown
        # drawdowns = cumulative_returns - rolling_max
        # return abs(drawdowns.min()) if len(drawdowns) > 0 else 0
