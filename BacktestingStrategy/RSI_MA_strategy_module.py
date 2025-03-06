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
    leverage: float
    base_bet_size: float
    rsi_period: int
    rsi_ma_period: int
    stop_loss_pct: float
    take_profit_pct: float
    risk_per_trade_pct: float

class TradingStrategy:
    def __init__(self, config: TradeConfig):
        self.config = config
        self.current_capital = config.initial_capital
        self.trades_history = []
        
    def calculate_position_size(self, price: float) -> float:
        """Calculate dynamic position size based on risk per trade"""
        risk_amount = self.current_capital * (self.config.risk_per_trade_pct / 100) # amount I can lose
         # total buy amount I can send by calculateing the stop loss an leverage as base. Example if I can lose 10 dolar and my stop loos is 10% then my trade will be of 100 dolars. That change according with the leverage.
        position_size_in_dollars = risk_amount  # Define your position size in dollars
        
        # If using leverage, position size will be multiplied by leverage
        position_size_in_dollars *= self.config.leverage
    
        # Ensure that the position does not exceed available capital
        if position_size_in_dollars > self.current_capital:
            position_size_in_dollars = self.current_capital  # Prevent position from exceeding available capital
        
        return position_size_in_dollars
        # The commented out logicbellow  was using the position as quantity of stocks, not dolar value:
        #position_size = (risk_amount / self.config.stop_loss_pct) * self.config.leverage
        #return min(position_size, self.config.base_bet_size * self.current_capital)

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
#            # Check stop loss and take profit if in position
#            if position != 0:
#                price_change_pct = (row['close'] - entry_price) / entry_price * 100
#                
#                if (position > 0 and price_change_pct <= -self.config.stop_loss_pct) or \
#                   (position < 0 and price_change_pct >= self.config.stop_loss_pct):
#                    # Stop loss hit
#                    monthly_trades.append({
#                        'timestamp': idx,
#                        'price': row['close'],
#                        'quantity': -position,
#                        'type': 'stop_loss'
#                    })
#                    position = 0
#                    
#                elif (position > 0 and price_change_pct >= self.config.take_profit_pct) or \
#                     (position < 0 and price_change_pct <= -self.config.take_profit_pct):
#                    # Take profit hit
#                    monthly_trades.append({
#                        'timestamp': idx,
#                        'price': row['close'],
#                        'quantity': -position,
#                        'type': 'take_profit'
#                    })
#                    position = 0
            
            # Process new signals
            if position == 0 and (row['RSI'] > row['RSI_MA'] and row['RSI_prev'] < row['RSI_MA_prev']):
                position_size = self.calculate_position_size(row['close'])
                monthly_trades.append({
                    'timestamp': idx,
                    'price': row['close'],
                    'quantity': position_size,
                    'type': 'buy'
                })
                position = position_size
                entry_price = row['close']
                
            elif position > 0 and (row['RSI'] < row['RSI_MA'] and row['RSI_prev'] > row['RSI_MA_prev']):
                monthly_trades.append({
                    'timestamp': idx,
                    'price': row['close'],
                    'quantity': -position,
                    'type': 'sell'
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
        
        # Calculate PnL (Profit and Loss)
#Claude        trades_df['pnl'] = trades_df.apply(
#Claude            lambda x: x['quantity'] * (x['price'] - trades_df['price'].shift(1)).fillna(0)
#Claude            if x['type'] in ['sell', 'stop_loss', 'take_profit'] else 0,
#Claude            axis=1
#Claude        ).fillna(0)
#Claude        GPT \/

        # Shift prices for calculating trade PnL
        trades_df['prev_price'] = trades_df['price'].shift(1)
        
       # PnL calculation (using dollar-based positions)
        trades_df['pnl'] = np.where(
            trades_df['type'].isin(['sell', 'stop_loss', 'take_profit']),
            trades_df['quantity'] * (1 - (trades_df['prev_price'] / trades_df['price'])),
            0
        )

        # Fill NaN values in PnL (e.g., for the first row where prev_price is NaN)
        trades_df['pnl'] =  trades_df['pnl'].fillna(0)

        # Verify pnl validity
        if 'pnl' not in trades_df.columns or trades_df['pnl'].isna().all():
            raise ValueError("Invalid 'pnl' column in trades_df.")
        
        # Total return
        total_return = trades_df['pnl'].sum()
    
        # Number of trades
        num_trades = len(trades_df[trades_df['type'].isin(['sell', 'stop_loss', 'take_profit'])])
    
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
