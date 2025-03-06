import logging
import MetaTrader5 as mt5
import json
from datetime import date, datetime, timedelta, time
import pandas as pd
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
import time as time_module  # rename time to avoid conflict
# library pra interagir com formato Parquet
import pyarrow as pa
import pyarrow.parquet as pq


class AsimoTrader():
    def __init__(self, file_path=None, login=None, password=None, server=None):
        if file_path:
            try:
                with open(file_path) as f:
                    credentials = json.load(f)
                self.login = credentials['loginJson']
                self.password = credentials['passwordJson']
                self.server = credentials['serverJson']
            except: print("Erro ao ler as credenciais"); quit()
        else:
            self.login = login
            self.password = password
            self.server = server
            if (login and password and server) == None: print("Erro ao ler as credenciais"); quit()

        # Caso o mt5 nao inicialize, quit()
        # if not mt5.initialize(login=self.login, password=self.password, server=self.server, path="C:\\Program Files\\MetaTrader 5\\terminal64.exe"):
        #    print("initialize() failed, error code = ", mt5.last_error())
        #    mt5.shutdown(); quit()
            
        # Set a retry limit and delay
        retry_limit = 3
        retry_delay = 30  # seconds
        
        # Opening and loging in
        
        mt5.initialize(Login=self.login, password=self.password, server=self.server)
        
        time_module.sleep(5)
        
        for attempt in range(retry_limit):
            try:
                # Attempt to initialize with a delay
                if mt5.initialize(Login=self.login):
                    print("MetaTrader 5 initialized successfully.")
                    break  # Exit loop if successful
                else:
                    raise Exception(f"initialize() failed. Error code: {mt5.last_error()}")
            except Exception as e:
                print(e)
                if attempt < retry_limit - 1:
                    mt5.shutdown()
                    mt5.initialize(Login=self.login, password=self.password, server=self.server)
                    print(f"Retrying in {retry_delay} seconds...")
                    time_module.sleep(retry_delay)
                else:
                    print("Initialization failed after multiple attempts.")
                    mt5.shutdown()
                    sys.exit("Exiting program due to repeated initialization failure.")
                    ##quit()
        
        # If we exit the loop successfully, the code will continue here
        print("Initialization complete, proceeding with other operations.")
        # inicializando variaveis necessarias para as funções
        self.timeframe_dict = {                         # o quanto cada timeframe pode representar em dias? Ver estudos na função self.read_ohlc()
            'TIMEFRAME_M1': [mt5.TIMEFRAME_M1, 60],
            'TIMEFRAME_M2': [mt5.TIMEFRAME_M2, 120],
            'TIMEFRAME_M3': [mt5.TIMEFRAME_M3, 180],
            'TIMEFRAME_M4': [mt5.TIMEFRAME_M4, 240],
            'TIMEFRAME_M5': [mt5.TIMEFRAME_M5, 300],
            'TIMEFRAME_M6': [mt5.TIMEFRAME_M6, 360],
            'TIMEFRAME_M10': [mt5.TIMEFRAME_M10, 600],
            'TIMEFRAME_M12': [mt5.TIMEFRAME_M12, 720],
            'TIMEFRAME_M15': [mt5.TIMEFRAME_M15, 900],
            'TIMEFRAME_M20': [mt5.TIMEFRAME_M20, 1200],
            'TIMEFRAME_M30': [mt5.TIMEFRAME_M30, 1800],
            'TIMEFRAME_H1': [mt5.TIMEFRAME_H1, 3600],
            'TIMEFRAME_H2': [mt5.TIMEFRAME_H2, 7200],
            'TIMEFRAME_H3': [mt5.TIMEFRAME_H3, 10800],
            'TIMEFRAME_H4': [mt5.TIMEFRAME_H4, 14400],
            'TIMEFRAME_H6': [mt5.TIMEFRAME_H6, 21600],
            'TIMEFRAME_H8': [mt5.TIMEFRAME_H8, 28800],
            'TIMEFRAME_H12': [mt5.TIMEFRAME_H12, 43200],
            'TIMEFRAME_D1': [mt5.TIMEFRAME_D1, 86400],
            'TIMEFRAME_W1': [mt5.TIMEFRAME_W1, 604800],
            'TIMEFRAME_MN1' : [mt5.TIMEFRAME_MN1, 2592000],
        }

        # Se nao tivermos as pastas (primeira inicialização), devemos cria-las
        if not os.path.isdir('ohlc'):
                os.mkdir('ohlc')
                for timeframe_dir in self.timeframe_dict.keys():
                    try: os.mkdir(f'ohlc\\{timeframe_dir}')
                    except FileExistsError: pass
        elif not os.path.isdir('ticks'): os.mkdir('ticks')

        # Prólogo
        '''
        Aqui devemos criar casos específicos para TIMEFRAMES diferentes.
        Um TIMEFRAME pequeno [1min] carrega mais informação, mais velas e consequentemente mais bytes;
        Logo temos que entender o limite de retorno do mt5 e carregar os dados em chunks;
        Estudo realizado no notebook timeframe_tests.ipynb
        '''
        # RESULTADO DOS TESTES
        '''
        Pelo resultado dos testes feitos, é possivel puxar até 66 dias anteriores por minuto de timeframe, ou seja:
        M1 -> datetime.now() - timedelta(days=66)
        M2 -> datetime.now() - timedelta(days=132)
        M3 -> datetime.now() - timedelta(days=198) [...]
        Mas por questões de segurança do código iremos utilizar 60 dias por timeframe
        Logo, ao receber um timeframe, teremos que verificar, talvez via um dict, quantos minutos aquele timeframe representa 
        '''
        # Como aumentar a capacidade do MetaTrader
        '''
        Ferramentas -> Configs -> Graficos -> Max_Range: Unlimited
        '''

    # FUNCTIONS ================================================   
    def update_ohlc(self, symbol, timeframe):
        initial_date = datetime(2012, 1, 1)
        final_date = datetime.now()
        print("Iniciando funcao update_ohlc.")
        # Checar se o ativo ja tem a file necessária
        if not os.path.exists(f'ohlc\\{timeframe}\\{symbol}_{timeframe}.parquet'): 
            print("Arquivo ainda nao existe. Iniciando o dataframe.")
            df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
        # Se a file ja existe, checar a data e repor a data inicial de pesquisa    
        else: 
            ## df = pd.read_csv(f'ohlc\\{timeframe}\\{symbol}_{timeframe}.csv')
            print("Arquivo ja existe. Lendo o arquivo parquet e iniciando o dataframe.")
            df = pq.ParquetFile(f'ohlc\\{timeframe}\\{symbol}_{timeframe}.parquet').read().to_pandas()
            df['time'] = pd.to_datetime(df['time'])
            if df['time'].max() < datetime.now() - timedelta(days=7): 
                initial_date = df['time'].max()
                print("Diferenca do ultimo registro pra hoje e menor que 7 dias. Atualizando a partir do ultimo registro")
            else: return
        timedelta_default = timedelta(days=self.timeframe_dict[timeframe][1])
        final_date_aux = initial_date + timedelta_default
        timeframe_name = timeframe
        timeframe = self.timeframe_dict[timeframe][0]
        # Appending data to dataframe
        while True:
            data_aux = mt5.copy_rates_range(symbol, timeframe, initial_date, min(final_date_aux, final_date)) 
            df_aux = pd.DataFrame(data_aux)
            df_aux['time'] = pd.to_datetime(df_aux['time']) # Maybe use unit='s' as a additional argument
            print(f'Concatenando dados de {initial_date} ate {min(final_date_aux, final_date)}.')
            df = pd.concat([df_aux, df], ignore_index=True)
            if final_date_aux > final_date: 
                break         
            if df_aux['time'].max() < initial_date:
                df = df.iloc[0:0]
                initial_date = min(final_date_aux, final_date)
                print(f"Ultimo registro do df_aux veio com a data {df_aux['time'].max()}. Limpando os dados e sequindo em frente. Novo initial_date: {initial_date}")
            else:
                initial_date = df_aux['time'].max()
            print(f"Total de linhas importadas: {len(df)}")
            final_date_aux = initial_date + timedelta_default
        # Save df to file
        df.sort_values(by='time', ascending=False, inplace=True)
        ## df.to_csv(f'ohlc\\{timeframe_name}\\{symbol}_{timeframe_name}.csv', index=False)
        filename = f'ohlc\\{timeframe_name}\\{symbol}_{timeframe_name}.parquet'
        print(f'Salvando arquivo final em {filename}.')
        table = pa.Table.from_pandas(df)
        with pq.ParquetWriter(filename, table.schema) as writer:
            writer.write_table(table)

    # Calculando apenas negocios efetivados, não todos. Existem ticks registrados que 
    def update_ticks(self, symbol):
        initial_date = datetime(2012, 1, 1)
        final_date = datetime.now()
        print("Iniciando funcao update_ticks.")
        if not os.path.exists(f'ticks\\{symbol}_ticksrange.parquet'): 
            print("Arquivo ainda nao existe. Iniciando o dataframe.")
            df = pd.DataFrame(columns=['time', 'bid', 'ask', 'last', 'volume', 'time_msc', 'flags', 'volume_real'])
        else:
            ## df = pd.read_csv(f'ticks\\{symbol}_ticksrange.csv')
            print("Arquivo ja existe. Lendo o arquivo parquet e iniciando o dataframe.")
            df = pq.ParquetFile(f'ticks\\{symbol}_ticksrange.parquet').read().to_pandas()
            df['time'] = pd.to_datetime(df['time'])
            if df['time'].max() < datetime.now() - timedelta(days=7): 
                print("Diferenca do ultimo registro pra hoje e menor que 7 dias. Atualizando a partir do ultimo registro")
                initial_date = df['time'].max()        
        ticks_data = mt5.copy_ticks_range(symbol, initial_date, final_date, mt5.COPY_TICKS_TRADE)
        df_aux = pd.DataFrame(ticks_data)
        df_aux['time'] = pd.to_datetime(df_aux['time']) # Maybe use unit='s' as a additional argument
        print("Concatenando df inicial com valores importados do copy_ticks_range.")
        df = pd.concat([df_aux, df], ignore_index=True)
        df['time'] = pd.to_datetime(df['time'])
        # Save df to file
        df.sort_values(by='time', ascending=False, inplace=True)
        ## df.to_csv(f'ticks\\{symbol}_ticksrange.csv', index=False)
        filename = f'ticks\\{symbol}_ticksrange.parquet'
        print(f'Salvando arquivo final em {filename}.')
        table = pa.Table.from_pandas(df)
        with pq.ParquetWriter(filename, table.schema) as writer:
            writer.write_table(table)

    def slice(self, type, symbol, initial_date, final_date, timeframe=None):
        path = f'ohlc\\{timeframe}\\{symbol}_{timeframe}.parquet' if type=='ohlc' else f'ticks\\{symbol}_ticksrange.parquet'
        if not os.path.exists(path):
            print(f"O ativo {symbol} não está registrado, favor criá-lo utilizando a função .update_{type}()")
        else:
            ## df = pd.read_csv(path)
            df = pq.ParquetFile(path).read().to_pandas()
            df['time'] = pd.to_datetime(df['time'])
            return df.loc[(df['time'] >= initial_date) & (df['time'] < final_date)]
        pass
    
    def read_ohlc(self, symbol, timeframe, initial_date=datetime(2012, 1, 1), final_date=datetime.now()):
        return self.slice('ohlc', symbol, initial_date, final_date, timeframe)

    def read_ticks(self, symbol, initial_date=datetime(2012, 1, 1), final_date=datetime.now()):
        return self.slice('ticks', symbol, initial_date, final_date)



# REPOSITORY ================================================================
# # Coleta todos os ativos ordinarios e retorna uma lista com eles
#     def get_symbols(self):
#         # Coletando todos ativos da B3
#         symbols = mt5.symbols_get()

#         # Separando os ativos ordinários (que nos interessam)
#         ativos = []
#         for symbol in symbols:
#             ticker = symbol.name
#             if not ticker[:4].isalpha():
#                 continue
#             if (len(ticker) == 5) and (ticker[-1] == '3'):
#                 ativos.append(ticker)
#         ativos.sort()
#         len(ativos)
#         return ativos