from dataclasses import dataclass
from datetime import date, datetime, timedelta, time
import pandas as pd
import numpy as np
import logging
import json
from typing import List, Dict, Optional
from scipy.stats import linregress
import os
import time as time_module  # rename time to avoid conflict
import subprocess
import importlib.util
import re
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
# library pra interagir com formato Parquet
import pyarrow as pa
import pyarrow.parquet as pq
# Library API
from binance.client import Client
import openai
from openai import OpenAI


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


## Section to execute binance related and data retrival commands:

class BinanceTrader():
    def __init__(self, file_path=None, api_key=None, api_secret=None):
        if file_path:
            try:
                with open(file_path) as f:
                    env = json.load(f)
                self.binance_api_key  = env['BINANCE_API_KEY']
                self.binance_api_secret = env['BINANCE_SECRET_KEY']
            except: print("Erro ao ler as credenciais"); quit()
        else:
            print("File is not available to read.")
            quit()
        # Setting connection
        self.binance_client = Client(binance_api_key,binance_api_secret) 
        # Getting server status. Empty means no errors.
        status = self.binance_client.ping()
        if not status:
            # The JSON response is empty meaning no errors. 
            print("Connection successful.")
        else: 
            print(status)
            quit()
        # inicializando variaveis necessarias para as funções
        self.timeframe_dict = {                         
            'M1': [self.binance_client.KLINE_INTERVAL_1MINUTE, 60],  # 1 minute # takes an average of 35 min to update starting from 2017.
            'M3': [self.binance_client.KLINE_INTERVAL_3MINUTE, 180],  # 3 minutes
            'M5': [self.binance_client.KLINE_INTERVAL_5MINUTE, 300],  # 5 minutes
            'M15': [self.binance_client.KLINE_INTERVAL_15MINUTE, 900],  # 15 minutes
            'M30': [self.binance_client.KLINE_INTERVAL_30MINUTE, 1800],  # 30 minutes
            'H1': [self.binance_client.KLINE_INTERVAL_1HOUR, 3600],  # 1 hour
            'H2': [self.binance_client.KLINE_INTERVAL_2HOUR, 7200],  # 2 hours
            'H4': [self.binance_client.KLINE_INTERVAL_4HOUR, 14400],  # 4 hours
            'H6': [self.binance_client.KLINE_INTERVAL_6HOUR, 21600],  # 6 hours
            'H8': [self.binance_client.KLINE_INTERVAL_8HOUR, 28800],  # 8 hours
            'H12': [self.binance_client.KLINE_INTERVAL_12HOUR, 43200],  # 12 hours
            'D1': [self.binance_client.KLINE_INTERVAL_1DAY, 86400],  # 1 day
            'D3': [self.binance_client.KLINE_INTERVAL_3DAY, 259200],  # 3 day
            'W1': [self.binance_client.KLINE_INTERVAL_1WEEK, 604800],  # 1 week
            'MN1': [self.binance_client.KLINE_INTERVAL_1MONTH, 2592000],  # 1 month
        }
        
        # Se nao tivermos as pastas (primeira inicialização), devemos cria-las
        if not os.path.isdir('binancedata\\ohlc'):
                os.mkdir('binancedata')
                os.mkdir('binancedata\\ohlc')
                print("Creating timeframe folders.")
                for timeframe_dir in self.timeframe_dict.keys():
                    try: 
                        os.mkdir(f'binancedata\\ohlc\\{timeframe_dir}')
                        print(f'binancedata\\ohlc\\{timeframe_dir} created.')
                    except FileExistsError: 
                        print(FileExistsError)
                        pass
        elif not os.path.isdir('binancedata\\ohlc'): 
            os.mkdir('binancedata\\ohlc')

##############################################################binance_update_ohlc###########################################################

    # FUNCTIONS ================================================   
    def binance_update_ohlc(self, symbol, timeframe, verbose=False):
        start_time = int(pd.Timestamp(datetime(2012, 1, 1), tz='UTC').timestamp() * 1000)
        end_time = int(pd.Timestamp(datetime.now(), tz='UTC').timestamp() * 1000)
        klines_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']
        print("Dados de tempo e colunas iniciadas. Iniciando funcao binance_update_ohlc.")
        
        # Checar se o ativo ja tem a file necessária
        if not os.path.exists(f'binancedata\\ohlc\\{timeframe}\\{symbol}_{timeframe}.parquet'): 
            print("Arquivo ainda nao existe. Iniciando o dataframe.")
            df = pd.DataFrame(columns=klines_columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Se a file ja existe, checar a data e repor a data inicial de pesquisa    
        else: 
            print("Arquivo ja existe. Lendo o arquivo parquet e iniciando o dataframe.")
            df = pq.ParquetFile(f'binancedata\\ohlc\\{timeframe}\\{symbol}_{timeframe}.parquet').read().to_pandas()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['timestamp'].max() < datetime.now() - timedelta(seconds=self.timeframe_dict[timeframe][1]):
                start_time = int(pd.Timestamp(df['timestamp'].max(), tz='UTC').timestamp() * 1000)
                print("Diferenca do ultimo registro pra hoje e anterior ao timeframe declarado. Atualizando a partir do ultimo registro")
            else: 
                print("Diferenca do ultimo registro pra hoje menor que timeframe declarado. Sem updates.")
                return
                
        # Definindo timeframes        
        timeframe_name = timeframe
        interval = self.timeframe_dict[timeframe][0]

    
        all_klines = []
        
        while start_time < end_time:
            if verbose: print(f" Getting the first 1000 after the start_time: {pd.to_datetime(start_time, unit='ms', utc=True)}.")
            klines = self.binance_client.get_klines(symbol=symbol, interval=interval, startTime=start_time)
            
            if not klines:
                print(f"No data for start time: {pd.to_datetime(start_time, unit='ms', utc=True)}.")
                break
            
            all_klines.extend(klines)
            start_time = klines[-1][6]  # Use the last close_time for pagination
            if verbose: print(f" New start_time: {pd.to_datetime(start_time, unit='ms', utc=True)}.")
        
        # Convert to DataFrame
        
        df_aux = pd.DataFrame(all_klines, columns=klines_columns)
        df_aux['timestamp'] = pd.to_datetime(df_aux['timestamp'], unit='ms')
        print(f"Total de linhas importadas: {len(df_aux)}")

        # Appending data to dataframe
        print('Concatenando dados.')
        df = pd.concat([df_aux, df], ignore_index=True)
        # Fixing format
        df.sort_values(by='timestamp', ascending=True, inplace=True)
        df = df.drop_duplicates(subset=['timestamp'])
        df[['open', 'high','low', 'close','volume', 'quote_asset_volume','taker_buy_base_volume', 'taker_buy_quote_volume']] = df[['open', 'high','low', 'close','volume', 'quote_asset_volume','taker_buy_base_volume', 'taker_buy_quote_volume']].astype(float)
        # Save df to file
        filename = f'binancedata\\ohlc\\{timeframe_name}\\{symbol}_{timeframe_name}.parquet'
        print(f'Salvando arquivo final em {filename}. \n')
        table = pa.Table.from_pandas(df)
        with pq.ParquetWriter(filename, table.schema) as writer:
            writer.write_table(table)

################################################slice###############################################################################

    def slice(self, type, symbol, initial_date, final_date, timeframe=None):
        path = f'binancedata\\ohlc\\{timeframe}\\{symbol}_{timeframe}.parquet' #if type=='ohlc' else f'ticks\\{symbol}_ticksrange.parquet'
        if not os.path.exists(path):
            print(f"O ativo {symbol} não está registrado, favor criá-lo utilizando a função .update_{type}()")
        else:
            df = pq.ParquetFile(path).read().to_pandas()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df[['open', 'high','low', 'close','volume', 'quote_asset_volume','taker_buy_base_volume', 'taker_buy_quote_volume']] = df[['open', 'high','low', 'close','volume', 'quote_asset_volume','taker_buy_base_volume', 'taker_buy_quote_volume']].astype(float)
            return df.loc[(df.index >= initial_date) & (df.index < final_date)]
        pass
        
#################################################read_ohlc##############################################################################    
    
    def read_ohlc(self, symbol, timeframe, initial_date=datetime(2012, 1, 1), final_date=datetime.now()):
        return self.slice('ohlc', symbol, initial_date, final_date, timeframe)

#    def read_ticks(self, symbol, initial_date=datetime(2012, 1, 1), final_date=datetime.now()):
#        return self.slice('ticks', symbol, initial_date, final_date)
#
    def update_ohlc_alltimeframes(self, symbol):
        results = {}  # To store results for each timeframe
        for timeframe_key in reversed(list(self.timeframe_dict.keys())):
            try:
                print(f'Updating values for symbol: {symbol} and timeframe: {timeframe_key}.')
                # Call the update function and store the result
                results[timeframe_key] = self.binance_update_ohlc(symbol, timeframe_key)
            except Exception as iterationError:
                print(f"Error updating {timeframe_key}: {iterationError}")
                results[timeframe_key] = None  # Optionally record the error
                pass
        return print('Timeframe iterations completed.')


## Section to execute the trading strategy

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


class TradingStrategy:
    def __init__(self, file_path=None, config: TradeConfig):
        self.config = config
        self.current_capital = config.initial_capital
        self.trades_history: List[Dict] = []
        self.current_drawdown = 0
        self.peak_capital = config.initial_capital
        # Initialize OpenAI client
        if file_path:
            try:
                with open(file_path) as f:
                        env = json.load(f)
                self.openAI_api_key  = env['OPENAI_API_KEY']
                openAI_client = OpenAI(
                    api_key=openAI_api_key
                )
            except: print("OpenAI API key not found in environment variables"); quit()
        else:
            print("File is not available to read.")
            quit()
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators for trading decisions."""
        df = df.copy()
        
        # Volatility indicators
        df['atr'] = self.calculate_atr(df, period=14)
        df['volatility'] = df['close'].pct_change().rolling(window=self.config.volatility_lookback).std()
        
        # Trend indicators
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['trend_strength'] = (df['ema_20'] - df['ema_50']) / df['ema_50'] * 100
        
        # Calculate RSI divergence
        df['rsi_slope'] = self.calculate_slope(df['RSI'], period=10)
        df['price_slope'] = self.calculate_slope(df['close'], period=10)
        df['divergence'] = np.where((df['rsi_slope'] < 0) & (df['price_slope'] > 0), -1,
                                  np.where((df['rsi_slope'] > 0) & (df['price_slope'] < 0), 1, 0))
        
        return df


    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()  # Use exponential moving average for more recent bias
        return atr

    @staticmethod
    def calculate_slope(series: pd.Series, period: int = 10) -> pd.Series:
        """Calculate the slope of a series using linear regression."""
        slopes = series.rolling(window=period).apply(lambda x: linregress(np.arange(period), x)[0], raw=True)
        return slopes

    def calculate_position_size(self, price: float, volatility: float) -> float:
        """Dynamic position sizing based on volatility and account risk."""
        risk_amount = self.current_capital * (self.config.risk_per_trade_pct / 100)
        
        # Adjust position size based on volatility
        volatility_scalar = 1.0
        if self.config.position_scaling:
            volatility_scalar = max(0.5, min(1.5, 1 / volatility))
        
        position_size = (risk_amount * volatility_scalar) / (self.config.stop_loss_pct / 100) * self.config.leverage
        return min(position_size, self.current_capital * 0.5)  # Max 50% of capital per trade

    def process_month(self, df: pd.DataFrame) -> dict:
        """Process monthly data with enhanced strategy logic."""
        if df.empty:
            logging.warning("Empty dataframe received for processing.")
            return self.create_empty_metrics()

        df = self.calculate_technical_indicators(df)
        position = 0
        entry_price = 0
        monthly_trades = []

        for i in range(1, len(df)):
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if position != 0:
                price_change_pct = (current_row['close'] - entry_price) / entry_price * 100
                exit_trades = self.check_exit_conditions(
                    monthly_trades, current_row, prev_row, price_change_pct, position
                )
                if exit_trades:
                    monthly_trades.extend(exit_trades)
                    position = 0

            if position == 0:
                entry_signal = self.check_entry_conditions(current_row, prev_row)
                if entry_signal:
                    position_size = self.calculate_position_size(
                        current_row['close'], current_row['volatility']
                    )
                    entry_trade = {
                        'timestamp': current_row.name,
                        'price': current_row['close'],
                        'quantity': position_size,
                        'type': 'buy',
                        'indicators': {
                            'rsi': current_row['RSI'],
                            'trend_strength': current_row['trend_strength'],
                            'volatility': current_row['volatility']
                        }
                    }
                    monthly_trades.append(entry_trade)
                    position = position_size
                    entry_price = current_row['close']

        return self.calculate_monthly_metrics(monthly_trades)

    def check_entry_conditions(
            self, row: pd.Series, prev_row: pd.Series
    ) -> bool:
        """Enhanced entry conditions with multiple confirmations."""
        # Get the volume series from the original DataFrame
        volume_ma = self.volume_series.rolling(20).mean()
        current_volume = row['volume']
    
        # Basic RSI conditions
        rsi_signal = (row['RSI'] > row['RSI_MA'] and prev_row['RSI'] <= prev_row['RSI_MA'])
    
        # Trend confirmation
        trend_signal = row['trend_strength'] > 0
    
        # Volatility filter
        volatility_ok = (
                row['volatility'] >= self.config.min_volatility_threshold and
                row['volatility'] <= self.config.max_volatility_threshold
        )
        
        # Volume confirmation using the current index
        volume_signal = current_volume > volume_ma.loc[row.name]
    
        return (rsi_signal and trend_signal and volatility_ok and volume_signal)   
        
    def check_exit_conditions(
        self, trades: List[Dict], row: pd.Series, prev_row: pd.Series, 
        price_change_pct: float, position: float
        ) -> List[Dict]:
        """Enhanced exit conditions with trailing stops and multiple signals."""
        exit_trades = []
        
        # Dynamic stop loss based on ATR
        dynamic_stop = self.config.stop_loss_pct * (1 + row['atr'] / row['close'])
        
        # Exit conditions
        stop_loss_hit = price_change_pct <= -dynamic_stop
        trend_reversal = (row['trend_strength'] < 0 and prev_row['trend_strength'] > 0)
        rsi_reversal = (row['RSI'] < row['RSI_MA'] and prev_row['RSI'] > prev_row['RSI_MA'])
        
        if stop_loss_hit or trend_reversal or rsi_reversal:
            exit_trades.append({
                'timestamp': row.name,
                'price': row['close'],
                'quantity': -position,
                'type': 'stop_loss' if stop_loss_hit else 'sell',
                'reason': 'stop_loss' if stop_loss_hit else 'signal_reversal'
            })
            
        return exit_trades

    def calculate_monthly_metrics(self, trades: List[Dict]) -> dict:
        """Calculate comprehensive trading metrics."""
        if not trades:
            return self.create_empty_metrics()

        df_trades = pd.DataFrame(trades)
        df_trades['pnl'] = pd.Series(dtype=float)  # Initialize empty pnl column
        
        # Calculate PnL for each trade
        for i in range(1, len(df_trades)):
            if df_trades.iloc[i]['type'] in ['sell', 'stop_loss']:
                entry_price = df_trades.iloc[i-1]['price']
                exit_price = df_trades.iloc[i]['price']
                quantity = abs(df_trades.iloc[i]['quantity'])
                df_trades.iloc[i, df_trades.columns.get_loc('pnl')] = (
                    (exit_price - entry_price) * quantity
                )

        total_pnl = df_trades['pnl'].sum()
        num_trades = len(df_trades[df_trades['type'].isin(['sell', 'stop_loss'])])
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        
        # Update capital and track drawdown
        self.current_capital += total_pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100

        metrics = {
            'return': total_pnl,
            'trades': num_trades,
            'win_rate': (winning_trades / num_trades * 100) if num_trades > 0 else 0,
            'avg_trade_return': total_pnl / num_trades if num_trades > 0 else 0,
            'sharpe_ratio': self.calculate_sharpe_ratio(df_trades),
            'max_drawdown': self.current_drawdown,
            'current_capital': self.current_capital
        }
        
        return metrics

    def create_empty_metrics(self) -> dict:
        """Create empty metrics dictionary."""
        return {
            'return': 0,
            'trades': 0,
            'win_rate': 0,
            'avg_trade_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'current_capital': self.current_capital
        }

    def calculate_sharpe_ratio(self, df_trades: pd.DataFrame) -> float:
        """Calculate Sharpe ratio from trade data."""
        if df_trades.empty or 'pnl' not in df_trades.columns:
            return 0
            
        daily_returns = df_trades.set_index('timestamp')['pnl'].resample('D').sum()
        if len(daily_returns) < 2:
            return 0
            
        return np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0

#### Framework Code Enhancement 


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimization.log')
    ]
)
logger = logging.getLogger(__name__)



# Declaring Functions
def get_historical_data(trader, symbol: str = 'BTCUSDT', timeframe: str = 'M5') -> pd.DataFrame:
    """Get historical data for the last year."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        df = trader.read_ohlc(
            symbol=symbol,
            timeframe=timeframe,
            initial_date=start_date,
            final_date=end_date
        )
        
        if df.empty:
            raise ValueError("Retrieved empty DataFrame from Binance")
            
        logger.info(f"Retrieved {len(df)} data points from Binance")
        return df
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise

def execute_strategy(file_path: str, config: dict) -> pd.DataFrame:
    """Execute trading strategy and return performance metrics."""
    try:
        # Dynamically import the strategy module
        spec = importlib.util.spec_from_file_location("optimized_strategy", file_path)
        if spec is None:
            raise ImportError(f"Could not load module from {file_path}")
        
        optimized_strategy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(optimized_strategy)

        # Initialize trader and get historical data
        trader = optimized_strategy.BinanceTrader('env.json')
        df = get_historical_data(trader)
        
        logger.info("Executing strategy with historical data...")
        results = trader.TradingStrategy.process_month(df)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([results])
        return results_df
    
    except Exception as e:
        logger.error(f"Error executing strategy: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def optimize_code(code: str, performance_metrics: Dict[str, Any]) -> str:
    """Optimize trading strategy code using OpenAI's API."""
    try:
        # Create a more detailed prompt with performance context
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
        
        response = openAI_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise Python trading strategy optimizer focused on performance and reliability."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        
        optimized_code = response.choices[0].message.content.strip()
        return optimized_code
    
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API Error: {e}")
        return code
    except Exception as e:
        logger.error(f"Unexpected error in code optimization: {e}")
        return code

def evaluate_performance(results_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Calculate comprehensive performance metrics."""
    try:
        if results_df.empty:
            logger.warning("Empty results DataFrame")
            return None
        
        metrics = {
            "total_return": float(results_df['return'].sum()),
            "win_rate": float(results_df['win_rate'].mean()),
            "sharpe_ratio": float(results_df['sharpe_ratio'].mean()),
            "max_drawdown": float(results_df['max_drawdown'].min()),
            "avg_trade_return": float(results_df['avg_trade_return'].mean()),
            "total_trades": int(results_df['trades'].sum())
        }
        
        logger.info("Performance metrics calculated successfully")
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating performance: {e}")
        return None

def optimization_loop(iterations: int = 5, config: Optional[dict] = None) -> None:
    """Improved optimization loop with baseline performance establishment."""
    if config is None:
        config = {
            "initial_capital": 2000,
            "leverage": 5,
            "base_bet_size": 10,
            "rsi_period": 14,
            "rsi_ma_period": 14,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0,
            "risk_per_trade_pct": 1.0
        }
    
    file_path = "OptimalAIstrategy.py"
    
    try:
        # First, establish baseline performance
        logger.info("Establishing baseline performance...")
        baseline_results = execute_strategy(file_path, config)
        
        if baseline_results.empty:
            logger.error("Failed to establish baseline performance")
            return
            
        baseline_metrics = evaluate_performance(baseline_results)
        if not baseline_metrics:
            logger.error("Failed to calculate baseline metrics")
            return
            
        logger.info(f"Baseline Metrics:\n{json.dumps(baseline_metrics, indent=2)}")
        
        last_metrics = baseline_metrics
        best_metrics = baseline_metrics
        best_code = read_file(file_path)
        
        # Start optimization iterations
        for iteration in range(iterations):
            logger.info(f"\n=== Starting Iteration {iteration + 1}/{iterations} ===")
            
            try:
                # Read current code
                current_code = read_file(file_path)
                
                # Optimize code based on last performance metrics
                logger.info("Optimizing strategy based on performance metrics...")
                optimized_code = optimize_code(current_code, last_metrics)
                
                if optimized_code == current_code:
                    logger.warning("No optimization changes made in this iteration")
                    continue
                
                # Write and test optimized code
                write_file(file_path, optimized_code)
                results_df = execute_strategy(file_path, config)
                
                if results_df.empty:
                    logger.warning("Strategy execution produced no results")
                    continue
                
                # Evaluate new performance
                new_metrics = evaluate_performance(results_df)
                if not new_metrics:
                    continue
                    
                logger.info(f"New Metrics:\n{json.dumps(new_metrics, indent=2)}")
                
                # Compare with best performance
                if new_metrics['sharpe_ratio'] > best_metrics['sharpe_ratio']:
                    best_metrics = new_metrics
                    best_code = optimized_code
                    logger.info("New best performance achieved!")
                else:
                    logger.info("No improvement over best performance")
                
                last_metrics = new_metrics
                
                # Add delay between iterations
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration + 1}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Restore best performing code
        write_file(file_path, best_code)
        logger.info(f"Optimization completed. Best metrics:\n{json.dumps(best_metrics, indent=2)}")
        
    except Exception as e:
        logger.error(f"Critical error in optimization loop: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        logger.info("Starting trading strategy optimization")
        optimization_loop(iterations=5)
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        logger.error(traceback.format_exc())