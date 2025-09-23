import pandas as pd
import numpy as np
import os

class TradingStrategy:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            self.data['datetime'] = pd.to_datetime(self.data['date'] + ' ' + self.data['time'])
            self.data.set_index('datetime', inplace=True)
            self.data.drop(columns=['date', 'time'], inplace=True)
            print(f"Data loaded from {self.file_path} successfully.")
        except Exception as e:
            print(f"Error loading data from {self.file_path}: {e}")

    def add_technical_indicators(self):
        try:
            self.data = self.data.reset_index()
            self.data['date'] = self.data['datetime'].dt.date  # Adicionar coluna apenas com a data

            for window in [3, 5, 7, 9, 11]:
                self.data[f'SMA_{window}'] = (
                    self.data.groupby(['id_ticker', 'date'])['close']
                    .transform(lambda x: x.rolling(window=window, min_periods=window).mean().round(4))
                )
            print("Technical indicators added successfully.")
        except Exception as e:
            print(f"Error adding technical indicators: {e}")

    def add_bands_features_norm(self):
        try:
            window = 7  # Janela padrão para as Bandas de Bollinger
            self.data['SMA_20'] = self.data.groupby('id_ticker')['close'].transform(lambda x: x.rolling(window=window, min_periods=window).mean())
            self.data['STD_20'] = self.data.groupby('id_ticker')['close'].transform(lambda x: x.rolling(window=window, min_periods=window).std())
            
            self.data['Upper_Band'] = self.data['SMA_20'] + (self.data['STD_20'] * 0.7929549)
            self.data['Lower_Band'] = self.data['SMA_20'] - (self.data['STD_20'] * 0.7929549)
            
            # Normalizando as bandas
            self.data['Bands_Norm'] = (self.data['close'] - self.data['Lower_Band']) / (self.data['Upper_Band'] - self.data['Lower_Band'])
            self.data['Bands_Norm'] = self.data['Bands_Norm'].fillna(0)  # Tratando NaN
            
            # Removendo colunas não normalizadas
            self.data.drop(columns=['SMA_20', 'STD_20', 'Upper_Band', 'Lower_Band'], inplace=True)
            
            print("Bollinger Bands calculated and normalized successfully.")
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")

    def normalize_moving_averages(self):
        try:
            for window in [3, 5, 7, 9, 11]:
                sma_col = f'SMA_{window}'
                self.data[f'NSMA_{window}'] = self.data.apply(
                    lambda row: 0 if row['high'] == row['low'] else (row[sma_col] - row['low']) / (row['high'] - row['low']),
                    axis=1
                )
                self.data.drop(columns=[sma_col], inplace=True)  # Removendo colunas não normalizadas
            print("Moving Averages normalized successfully.")
        except Exception as e:
            print(f"Error normalizing moving averages: {e}")
    
    def save_data_to_csv(self, output_file):
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
    strategy.add_bands_features_norm()
    strategy.normalize_moving_averages()
    strategy.save_data_to_csv(output_file)

if __name__ == '__main__':
    input_file = r"C:\\Users\\othav\\BovDB.v2\\dados.csv"
    output_file = r"C:\\Users\\othav\\BovDB.v2\\dados_indicadores.csv"

    process_single_file(input_file, output_file)
