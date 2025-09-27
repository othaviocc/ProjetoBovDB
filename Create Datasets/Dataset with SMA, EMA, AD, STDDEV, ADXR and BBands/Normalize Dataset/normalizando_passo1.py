import pandas as pd
import numpy as np
import os

class TradingStrategy:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):  # carregar dataset
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded from {self.file_path} successfully.")
        except Exception as e:
            print(f"Error loading data from {self.file_path}: {e}")

    def normalize_indicators_by_prev_close(self):
        try:
            # lista de indicadores que devem ser normalizados
            sma_cols = [f"SMA_{w}" for w in [3, 5, 7, 9, 11]]
            ema_cols = [f"EMA_{w}" for w in [3, 5, 7, 9, 11]]
            std_cols = [f"std_close{w}" for w in [3, 5, 7, 9, 11]] + \
                       [f"std_open{w}" for w in [3, 5, 7, 9, 11]]

            all_cols = sma_cols + ema_cols + std_cols

            # fechar anterior
            prev_close = self.data['close'].shift(1)

            for col in all_cols:
                if col in self.data.columns:
                    self.data[col] = self.data[col] / prev_close

            print("Indicators normalized by previous close successfully.")
        except Exception as e:
            print(f"Error normalizing indicators: {e}")

    def save_data_to_csv(self, output_file):  # salvar no csv
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
    strategy.normalize_indicators_by_prev_close()
    strategy.save_data_to_csv(output_file)


if __name__ == '__main__':
    input_file = r"C:\\Users\\othav\\BovDB.v2\\dados_indicadores.csv"
    output_file = r"C:\\Users\\othav\\BovDB.v2\\normalizados_passo1.csv"

    process_single_file(input_file, output_file)
