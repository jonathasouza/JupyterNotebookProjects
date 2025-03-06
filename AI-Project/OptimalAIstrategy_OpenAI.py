from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import json
import os
import time
import traceback
from typing import List, Dict, Optional, Any
import pyarrow as pa
import pyarrow.parquet as pq
from binance.client import Client
from openai import OpenAI
from scipy.stats import linregress
import importlib.util

# Configure logging once at the module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimization.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradeConfig:
    initial_capital: float = 2000
    leverage: float = 5
    base_bet_size: float = 10
    rsi_period: int = 14
    rsi_ma_period: int = 14
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    risk_per_trade_pct: float = 1.0
    volatility_lookback: int = 20
    trend_lookback: int = 50
    min_volatility_threshold: float = 0.5
    max_volatility_threshold: float = 3.0
    position_scaling: bool = True

class BinanceTrader:
    def __init__(self, cred_file_path=None):
        self.setup_credentials(cred_file_path)
        self.initialize_timeframes()
        self.setup_directories()
        
    def setup_credentials(self, cred_file_path):
        """Set up API credentials from file."""
        if not cred_file_path:
            raise ValueError("File path for credentials is required")
            
        try:
            with open(cred_file_path) as f:
                env = json.load(f)
            self.binance_api_key = env['BINANCE_API_KEY']
            self.binance_api_secret = env['BINANCE_SECRET_KEY']
            self.binance_client = Client(self.binance_api_key, self.binance_api_secret)
            
            # Verify connection
            status = self.binance_client.ping()
            if not status:
                logger.info("Binance connection successful")
            else:
                raise ConnectionError(f"Binance connection failed: {status}")
                
        except Exception as e:
            logger.error(f"Error setting up credentials: {e}")
            raise

    def initialize_timeframes(self):
        """Initialize timeframe dictionary."""
        self.timeframe_dict = {
            'M1': [Client.KLINE_INTERVAL_1MINUTE, 60],
            'M3': [Client.KLINE_INTERVAL_3MINUTE, 180],
            'M5': [Client.KLINE_INTERVAL_5MINUTE, 300],
            'M15': [Client.KLINE_INTERVAL_15MINUTE, 900],
            'M30': [Client.KLINE_INTERVAL_30MINUTE, 1800],
            'H1': [Client.KLINE_INTERVAL_1HOUR, 3600],
            'H2': [Client.KLINE_INTERVAL_2HOUR, 7200],
            'H4': [Client.KLINE_INTERVAL_4HOUR, 14400],
            'H6': [Client.KLINE_INTERVAL_6HOUR, 21600],
            'H8': [Client.KLINE_INTERVAL_8HOUR, 28800],
            'H12': [Client.KLINE_INTERVAL_12HOUR, 43200],
            'D1': [Client.KLINE_INTERVAL_1DAY, 86400],
            'D3': [Client.KLINE_INTERVAL_3DAY, 259200],
            'W1': [Client.KLINE_INTERVAL_1WEEK, 604800],
            'MN1': [Client.KLINE_INTERVAL_1MONTH, 2592000]
        }

    def setup_directories(self):
        """Set up required directories for data storage."""
        base_dir = 'binancedata'
        ohlc_dir = os.path.join(base_dir, 'ohlc')
        
        os.makedirs(ohlc_dir, exist_ok=True)
        
        for timeframe in self.timeframe_dict.keys():
            os.makedirs(os.path.join(ohlc_dir, timeframe), exist_ok=True)

    def binance_update_ohlc(self, symbol, timeframe, verbose=False):
        """Update OHLC data for a symbol and timeframe."""
        file_path = f'binancedata/ohlc/{timeframe}/{symbol}_{timeframe}.parquet'
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                  'close_time', 'quote_asset_volume', 'number_of_trades', 
                  'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']

        # Initialize or load existing data
        if os.path.exists(file_path):
            df = pq.read_table(file_path).to_pandas()
            start_time = int(pd.Timestamp(df['timestamp'].max(), tz='UTC').timestamp() * 1000)
        else:
            df = pd.DataFrame(columns=columns)
            start_time = int(pd.Timestamp(datetime(2012, 1, 1), tz='UTC').timestamp() * 1000)

        end_time = int(pd.Timestamp(datetime.now(), tz='UTC').timestamp() * 1000)
        
        # Fetch new data
        new_data = []
        while start_time < end_time:
            klines = self.binance_client.get_klines(
                symbol=symbol,
                interval=self.timeframe_dict[timeframe][0],
                startTime=start_time
            )
            if not klines:
                break
            new_data.extend(klines)
            start_time = klines[-1][6]

        # Process and save data
        if new_data:
            new_df = pd.DataFrame(new_data, columns=columns)
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
            df = pd.concat([df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_asset_volume', 'taker_buy_base_volume', 
                             'taker_buy_quote_volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            table = pa.Table.from_pandas(df)
            pq.write_table(table, file_path)

    def read_ohlc(self, symbol, timeframe, initial_date=datetime(2012, 1, 1), 
                 final_date=datetime.now()):
        """Read OHLC data for a symbol and timeframe."""
        file_path = f'binancedata/ohlc/{timeframe}/{symbol}_{timeframe}.parquet'
        if not os.path.exists(file_path):
            logger.error(f"No data found for {symbol} at {timeframe} timeframe")
            return pd.DataFrame()
            
        df = pq.read_table(file_path).to_pandas()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                         'quote_asset_volume', 'taker_buy_base_volume', 
                         'taker_buy_quote_volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        return df.loc[(df.index >= initial_date) & (df.index < final_date)]

class TradingStrategy:
    def __init__(self, cred_file_path: str, config: TradeConfig):
        self.config = config
        self.current_capital = config.initial_capital
        self.trades_history = []
        self.current_drawdown = 0
        self.peak_capital = config.initial_capital
        
        # Initialize OpenAI client
        try:
            with open(cred_file_path) as f:
                env = json.load(f)
            self.openai_client = OpenAI(api_key=env['OPENAI_API_KEY'])
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for trading decisions."""
        df = df.copy()
        
        # Volatility indicators
        df['atr'] = TradingStrategy.calculate_atr(df)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        # Trend indicators
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['trend_strength'] = (df['ema_20'] - df['ema_50']) / df['ema_50'] * 100
        
        # RSI and divergence
        df['RSI'] = TradingStrategy.calculate_rsi(df['close'])
        df['RSI_MA'] = df['RSI'].rolling(window=14).mean()
        df['rsi_slope'] = TradingStrategy.calculate_slope(df['RSI'])
        df['price_slope'] = TradingStrategy.calculate_slope(df['close'])
        
        return df

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr = pd.DataFrame({
            'tr1': df['high'] - df['low'],
            'tr2': abs(df['high'] - df['close'].shift()),
            'tr3': abs(df['low'] - df['close'].shift())
        }).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_slope(series: pd.Series, period: int = 10) -> pd.Series:
        """Calculate slope using linear regression."""
        return series.rolling(window=period).apply(
            lambda x: linregress(range(len(x)), x)[0] if len(x) > 1 else 0
        )

    def process_trades(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process trades and calculate metrics."""
        if df.empty:
            return self.create_empty_metrics()

        df = self.calculate_technical_indicators(df)
        trades = self.execute_trades(df)
        return self.calculate_metrics(trades)

    def execute_trades(self, df: pd.DataFrame) -> List[Dict]:
        """Execute trading strategy."""
        trades = []
        position = 0
        entry_price = 0

        for i in range(1, len(df)):
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if position != 0:
                price_change_pct = (current_row['close'] - entry_price) / entry_price * 100
                if self.check_exit_conditions(current_row, prev_row, price_change_pct):
                    trades.append({
                        'timestamp': current_row.name,
                        'price': current_row['close'],
                        'quantity': -position,
                        'type': 'sell'
                    })
                    position = 0
            
            elif self.check_entry_conditions(current_row, prev_row):
                position_size = self.calculate_position_size(
                    current_row['close'],
                    current_row['volatility']
                )
                trades.append({
                    'timestamp': current_row.name,
                    'price': current_row['close'],
                    'quantity': position_size,
                    'type': 'buy'
                })
                position = position_size
                entry_price = current_row['close']

        return trades
    
    def optimize_strategy(self, cred_file_path: str,code_file_path iterations: int = 5):
        """Main optimization loop with AI code optimization."""
        try:
            # Initialize components
            trader = BinanceTrader(cred_file_path)
            config = self.config  # Use the existing config
            
            # Get initial data
            df = trader.read_ohlc('BTCUSDT', 'M5', 
                                 initial_date=datetime.now() - timedelta(days=365))
            
            if df.empty:
                logger.error("No historical data available")
                return
                
            # Run optimization loop
            best_metrics = None
            best_code = None
            last_metrics = None
            
            for i in range(iterations):
                logger.info(f"Starting optimization iteration {i+1}/{iterations}")
                
                # Get current code and metrics
                with open(code_file_path, 'r') as f:
                    current_code = f.read()
                
                current_metrics = self.process_trades(df)
                
                # Optimize code using AI
                optimized_code = self.optimize_code(
                    current_code, 
                    last_metrics or current_metrics,
                    self.openai_client
                )
                
                if optimized_code == current_code:
                    logger.info("No code optimizations suggested in this iteration")
                else:
                    # Test optimized code
                    with open(code_file_path, 'w') as f:
                        f.write(optimized_code)
                    
                    # Reload and test the strategy
                    importlib.invalidate_caches()
                    new_strategy = TradingStrategy(code_file_path, self.config)
                    new_metrics = new_strategy.process_trades(df)
                    
                    # Compare performance
                    if not best_metrics or new_metrics['sharpe_ratio'] > best_metrics['sharpe_ratio']:
                        logger.info("New best performance achieved!")
                        best_metrics = new_metrics
                        best_code = optimized_code
                    else:
                        logger.info("No improvement over best performance")
                    
                    last_metrics = new_metrics
                
                # Add delay to avoid rate limiting
                time.sleep(1)
                
            # Save best version
            if best_code:
                with open(code_file_path, 'w') as f:
                    f.write(best_code)
                    
            logger.info(f"Optimization completed. Best metrics: {best_metrics}")
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            logger.error(traceback.format_exc())


    def check_entry_conditions(self, current_row: pd.Series, prev_row: pd.Series) -> bool:
        """
        Check if conditions are met for entering a trade.
        
        Parameters:
        current_row (pd.Series): Current candle data with indicators
        prev_row (pd.Series): Previous candle data with indicators
        
        Returns:
        bool: True if entry conditions are met, False otherwise
        """
        # Volatility check
        if not (self.config.min_volatility_threshold <= current_row['volatility'] <= self.config.max_volatility_threshold):
            return False
        
        # Trend strength check
        trend_is_strong = abs(current_row['trend_strength']) > 0.5
        
        # RSI conditions
        rsi_oversold = current_row['RSI'] < 30
        rsi_overbought = current_row['RSI'] > 70
        
        # Price and RSI divergence
        bullish_divergence = (
            current_row['price_slope'] > 0 and 
            current_row['rsi_slope'] > 0 and
            current_row['RSI'] > prev_row['RSI']
        )
        
        # EMA crossover
        ema_crossover = (
            prev_row['ema_20'] <= prev_row['ema_50'] and
            current_row['ema_20'] > current_row['ema_50']
        )
        
        # Combined entry conditions
        entry_signal = (
            trend_is_strong and
            ((rsi_oversold and bullish_divergence) or ema_crossover)
        )
        
        return entry_signal

    def check_exit_conditions(self, current_row: pd.Series, prev_row: pd.Series, price_change_pct: float) -> bool:
        """
        Check if conditions are met for exiting a trade.
        
        Parameters:
        current_row (pd.Series): Current candle data with indicators
        prev_row (pd.Series): Previous candle data with indicators
        price_change_pct (float): Percentage change in price since entry
        
        Returns:
        bool: True if exit conditions are met, False otherwise
        """
        # Stop loss check
        if price_change_pct <= -self.config.stop_loss_pct:
            return True
        
        # Take profit check
        if price_change_pct >= self.config.take_profit_pct:
            return True
        
        # RSI overbought condition
        if current_row['RSI'] > 70:
            return True
        
        # Trend reversal check
        ema_crossover_down = (
            prev_row['ema_20'] >= prev_row['ema_50'] and
            current_row['ema_20'] < current_row['ema_50']
        )
        
        # Trend weakness check
        trend_weakening = (
            abs(current_row['trend_strength']) < abs(prev_row['trend_strength']) and
            abs(current_row['trend_strength']) < 0.3
        )
        
        # Volatility expansion check
        volatility_too_high = current_row['volatility'] > self.config.max_volatility_threshold
        
        return ema_crossover_down or trend_weakening or volatility_too_high

    def calculate_position_size(self, current_price: float, volatility: float) -> float:
        """
        Calculate the position size based on current conditions.
        
        Parameters:
        current_price (float): Current asset price
        volatility (float): Current volatility metric
        
        Returns:
        float: Position size in base currency units
        """
        # Calculate base position size from risk percentage
        risk_amount = self.current_capital * (self.config.risk_per_trade_pct / 100)
        base_position = (risk_amount * self.config.leverage) / current_price
        
        if not self.config.position_scaling:
            return base_position
        
        # Scale position size based on volatility
        volatility_scaling = 1.0
        if volatility > self.config.max_volatility_threshold * 0.8:
            volatility_scaling = 0.5
        elif volatility < self.config.min_volatility_threshold * 1.2:
            volatility_scaling = 1.5
            
        return base_position * volatility_scaling

    def calculate_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """
        Calculate performance metrics from trades.
        
        Parameters:
        trades (List[Dict]): List of executed trades
        
        Returns:
        Dict[str, Any]: Dictionary containing performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'return_on_capital': 0.0
            }
        
        profits = []
        running_balance = self.config.initial_capital
        peak_balance = running_balance
        max_drawdown = 0.0
        
        for trade in trades:
            if trade['type'] == 'sell':
                profit = trade['price'] * trade['quantity']
                profits.append(profit)
                running_balance += profit
                
                # Update drawdown calculations
                peak_balance = max(peak_balance, running_balance)
                drawdown = (peak_balance - running_balance) / peak_balance * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        winning_trades = len([p for p in profits if p > 0])
        total_trades = len(trades) // 2  # Divide by 2 because each complete trade has an entry and exit
        
        return {
            'total_trades': total_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'profit_factor': abs(sum([p for p in profits if p > 0]) / sum([p for p in profits if p < 0])) if any(p < 0 for p in profits) else 0,
            'sharpe_ratio': np.mean(profits) / np.std(profits) if len(profits) > 1 else 0,
            'max_drawdown': max_drawdown,
            'return_on_capital': ((running_balance - self.config.initial_capital) / self.config.initial_capital * 100)
        }

    def create_empty_metrics(self) -> Dict[str, Any]:
        """Create empty metrics dictionary when no trades are executed."""
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'return_on_capital': 0.0
        }


    @staticmethod
    def optimize_code(code: str, performance_metrics: Dict[str, Any], openai_client: OpenAI) -> str:
        """Optimize trading strategy code using OpenAI's API."""
        try:
            metrics_str = json.dumps(performance_metrics, indent=2) if performance_metrics else "No prior performance data"
            
            prompt = f"""
            You are tasked with optimizing a Python trading strategy script based on its current performance metrics:
            
            CURRENT PERFORMANCE:
            {metrics_str}
    
            OPTIMIZATION OBJECTIVES:
            1. Address any performance bottlenecks indicated by the metrics
            2. Improve win rate and risk-adjusted returns
            3. Reduce drawdown if it's high
            4. Enhance signal generation based on performance data
            5. If necessary, adapt the strategy to the most recent public known effective quantitative strategy
            
            CRITICAL REQUIREMENTS:
            1. PRESERVE THE ENTIRE FILE STRUCTURE AND ALL IMPORTS
            2. DO NOT REMOVE OR MODIFY CLASS CONSTRUCTORS
            3. IMPROVE ONLY THE INTERNAL LOGIC OF METHODS
            4. ENSURE ALL ORIGINAL CLASSES AND METHODS REMAIN INTACT
            5. FOCUS ON IMPROVING AREAS INDICATED BY THE METRICS
    
            ORIGINAL CODE:
            ```python
            {code}
            ```
    
            RETURN THE COMPLETE, OPTIMIZED PYTHON SCRIPT EXACTLY AS IT WOULD BE SAVED IN THE FILE.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a precise Python trading strategy optimizer focused on performance and reliability."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            
            optimized_code = response.choices[0].message.content.strip()
            
            # Remove any markdown code blocks if present
            optimized_code = optimized_code.replace('```python', '').replace('```', '').strip()
            
            return optimized_code
            
        except Exception as e:
            logger.error(f"Error in code optimization: {e}")
            logger.error(traceback.format_exc())
            return code