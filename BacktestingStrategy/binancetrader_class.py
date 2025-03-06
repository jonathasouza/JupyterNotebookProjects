import json
from datetime import date, datetime, timedelta, time
import pandas as pd
import numpy as np
import os
import time as time_module  # rename time to avoid conflict
# library pra interagir com formato Parquet
import pyarrow as pa
import pyarrow.parquet as pq
# Library Binance
from binance.client import Client

class BinanceTrader():
    def __init__(self, file_path=None, api_key=None, api_secret=None):
        if file_path:
            try:
                with open(file_path) as f:
                    envBinance = json.load(f)
                self.api_key  = envBinance['APIKey']
                self.api_secret = envBinance['SecretKey']
            except: print("Erro ao ler as credenciais"); quit()
        else:
            self.api_key  = envBinance['APIKey']
            self.api_secret = envBinance['SecretKey']
            if (api_key and api_secret ) == None: 
                print("Erro ao ler as credenciais")
                quit()
        # Setting connection
        self.client = Client(api_key,api_secret) 
        # Getting server status. Empty means no errors.
        status = self.client.ping()
        if not status:
            # The JSON response is empty meaning no errors. 
            print("Connection successful.")
        else: 
            print(status)
            quit()
            
        # inicializando variaveis necessarias para as funções
        self.timeframe_dict = {                         
            'M1': [self.client.KLINE_INTERVAL_1MINUTE, 60],  # 1 minute # takes an average of 35 min to update starting from 2017.
            'M3': [self.client.KLINE_INTERVAL_3MINUTE, 180],  # 3 minutes
            'M5': [self.client.KLINE_INTERVAL_5MINUTE, 300],  # 5 minutes
            'M15': [self.client.KLINE_INTERVAL_15MINUTE, 900],  # 15 minutes
            'M30': [self.client.KLINE_INTERVAL_30MINUTE, 1800],  # 30 minutes
            'H1': [self.client.KLINE_INTERVAL_1HOUR, 3600],  # 1 hour
            'H2': [self.client.KLINE_INTERVAL_2HOUR, 7200],  # 2 hours
            'H4': [self.client.KLINE_INTERVAL_4HOUR, 14400],  # 4 hours
            'H6': [self.client.KLINE_INTERVAL_6HOUR, 21600],  # 6 hours
            'H8': [self.client.KLINE_INTERVAL_8HOUR, 28800],  # 8 hours
            'H12': [self.client.KLINE_INTERVAL_12HOUR, 43200],  # 12 hours
            'D1': [self.client.KLINE_INTERVAL_1DAY, 86400],  # 1 day
            'D3': [self.client.KLINE_INTERVAL_3DAY, 259200],  # 3 day
            'W1': [self.client.KLINE_INTERVAL_1WEEK, 604800],  # 1 week
            'MN1': [self.client.KLINE_INTERVAL_1MONTH, 2592000],  # 1 month
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
            klines = self.client.get_klines(symbol=symbol, interval=interval, startTime=start_time)
            
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
####################################################get_rsi###########################################################################
    
    def calculate_rsi(df_input, rsi_period=14, ma_type="SMA", ma_period=14):
        """
        Calculate RSI and RSI Moving Average for a given DataFrame with candlestick data.
    
        Parameters:
            df_input (pd.DataFrame): Input DataFrame with candlestick data. Must include a 'close' column.
            rsi_period (int): Period for the RSI calculation. Default is 14.
            ma_type (str): Type of moving average for smoothing the RSI. Options: 'SMA', 'EMA', 'RMA'.
            ma_period (int): Period for the moving average. Default is 14.
    
        Returns:
            pd.DataFrame: Input DataFrame with added 'RSI' and 'RSI_MA' columns.
        """
        # Ensure 'close' column exists
        if 'close' not in df_input.columns:
            raise ValueError("Input DataFrame must contain a 'close' column.")
    
        # Calculate RSI
        close = df_input['close']
        change = close.diff()
        gain = np.where(change > 0, change, 0)
        loss = np.where(change < 0, -change, 0)
    
        avg_gain = pd.Series(gain).rolling(window=rsi_period).mean()
        avg_loss = pd.Series(loss).rolling(window=rsi_period).mean()
    
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        df_input['RSI'] = rsi
    
        # Calculate Moving Average of RSI
        if ma_type == "SMA":
            df_input['RSI_MA'] = df_input['RSI'].rolling(window=ma_period).mean()
        elif ma_type == "EMA":
            df_input['RSI_MA'] = df_input['RSI'].ewm(span=ma_period, adjust=False).mean()
        elif ma_type == "RMA":  # RMA is equivalent to Pine Script's ta.rma
            df_input['RSI_MA'] = df_input['RSI'].ewm(alpha=1/ma_period, adjust=False).mean()
        else:
            raise ValueError("Invalid ma_type. Options are: 'SMA', 'EMA', 'RMA'.")
    
        return df_input
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
            return df.loc[(df['timestamp'] >= initial_date) & (df['timestamp'] < final_date)]
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