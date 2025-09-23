import pandas as pd
import numpy as np
import os

class TradingStrategy:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None


    def load_data(self): #função para carregar e organizar o dados
        try:
            self.data = pd.read_csv(self.file_path)
            self.data['datetime'] = pd.to_datetime(self.data['date'] + ' ' + self.data['time'])
            self.data.set_index('datetime', inplace=True)
            self.data.drop(columns=['date', 'time'], inplace=True)
            print(f"Data loaded from {self.file_path} successfully.")
        except Exception as e:
            print(f"Error loading data from {self.file_path}: {e}")

    def add_technical_indicators(self): #adicionar os inidicadores de MA
        try:
            self.data = self.data.reset_index()
            self.data['date'] = self.data['datetime'].dt.date  # Adicionar coluna apenas com a data

            # Cálculo de indicadores técnicos por ticker e data
            for window in [3, 5, 7, 9]: #janelas escolhidas 
                #Calcula para SMA E EMA
                self.data[f'SMA_{window}'] = (
                    self.data.groupby(['id_ticker', 'date'])['close']
                    .transform(lambda x: x.rolling(window=window, min_periods=window).mean().round(4))
                )
                
                self.data[f'EMA_{window}'] = (
                    self.data.groupby(['id_ticker', 'date'])['close']
                    .transform(lambda x: x.ewm(span=window, adjust=False).mean().round(4))
                )
            
            print("Technical indicators added successfully.")
        except Exception as e:
            print(f"Error adding technical indicators: {e}")

    def add_std_features(self):
        try:
            for window in [3, 5, 7, 9]: #janelas escolhidas
                #calcula para desvio padrão close e open
                self.data[f'std_close{window}'] = (
                    self.data.groupby(['id_ticker', 'date'])['close']
                    .transform(lambda x: x.rolling(window=window, min_periods=window).std().round(4))
                )
                
                self.data[f'std_open{window}'] = (
                    self.data.groupby(['id_ticker', 'date'])['open']
                    .transform(lambda x: x.rolling(window=window, min_periods=window).std().round(4))
                )
            
            print("std features and standard deviations added successfully.")
        except Exception as e:
            print(f"Error adding temporal features: {e}")

    def save_data_to_csv(self, output_file): #salva o arquivo de saida
        try:
            self.data.dropna(inplace=True)
            self.data.to_csv(output_file, index=False)
            print(f"Data saved successfully to {output_file}")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")


def process_single_file(input_file, output_file):
    print(f"\nProcessing file: {input_file}")
    strategy = TradingStrategy(input_file)
    strategy.load_data()
    strategy.add_technical_indicators()
    strategy.add_std_features()
    strategy.save_data_to_csv(output_file)


if __name__ == '__main__':
    input_file = r"E:\\progamação\\vs code\\Bovdb\\Traiding_Data\\main\\docs\\dados.csv"
    output_file = r"E:\\progamação\\vs code\\Bovdb\\Traiding_Data\\main\\docs\\dados_indicadores.csv"

    # Process the single CSV file
    process_single_file(input_file, output_file)
