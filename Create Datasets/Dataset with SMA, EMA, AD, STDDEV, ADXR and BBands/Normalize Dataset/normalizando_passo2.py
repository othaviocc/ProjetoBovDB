import pandas as pd
import numpy as np

class TradingNormalizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded from {self.file_path} successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def normalize_sma_ema_std(self):
        try:
            sma_cols = [f"SMA_{w}" for w in [3, 5, 7, 9, 11]]
            ema_cols = [f"EMA_{w}" for w in [3, 5, 7, 9, 11]]
            std_cols = [f"std_close{w}" for w in [3, 5, 7, 9, 11]] + \
                       [f"std_open{w}" for w in [3, 5, 7, 9, 11]]
            all_cols = sma_cols + ema_cols + std_cols

            for col in all_cols:
                if col in self.data.columns:
                    col_min = self.data[col].min()
                    col_max = self.data[col].max()
                    self.data[col] = (self.data[col] - col_min) / (col_max - col_min)

            print("SMA, EMA and std normalized successfully.")
        except Exception as e:
            print(f"Error normalizing SMA/EMA/std: {e}")

    def normalize_bollinger(self):
        try:
            if all(c in self.data.columns for c in ["Bollinger_Lower", "Bollinger_Upper"]):
                prev_close = self.data["close"].shift(1)
                self.data["Bollinger_Norm"] = (
                    (prev_close - self.data["Bollinger_Lower"]) /
                    (self.data["Bollinger_Upper"] - self.data["Bollinger_Lower"])
                )
                # remove as colunas antigas
                self.data.drop(columns=["Bollinger_Mid", "Bollinger_Upper", "Bollinger_Lower"], inplace=True)
                print("Bollinger Bands normalized into a single column successfully.")
        except Exception as e:
            print(f"Error normalizing Bollinger Bands: {e}")

    def normalize_adxr(self):
        try:
            if "ADXR" in self.data.columns:
                self.data["ADXR"] = self.data["ADXR"] / 100
                print("ADXR normalized successfully.")
        except Exception as e:
            print(f"Error normalizing ADXR: {e}")

    def add_trend(self):
        try:
            self.data["trend"] = np.where(self.data["close"] > self.data["close"].shift(1), 1, 0)
            print("Trend column added successfully.")
        except Exception as e:
            print(f"Error adding trend: {e}")

    def save_data(self, output_file):
        try:
            self.data.dropna(inplace=True)
            self.data.to_csv(output_file, index=False)
            print(f"Data saved successfully to {output_file}")
        except Exception as e:
            print(f"Error saving data: {e}")


def process_file(input_file, output_file):
    normalizer = TradingNormalizer(input_file)
    normalizer.load_data()
    normalizer.normalize_sma_ema_std()
    normalizer.normalize_bollinger()
    normalizer.normalize_adxr()
    normalizer.add_trend()
    normalizer.save_data(output_file)


if __name__ == "__main__":
    input_file = r"C:\\Users\\othav\\BovDB.v2\\normalizados_passo1.csv"
    output_file = r"C:\\Users\\othav\\BovDB.v2\\normalizados_passo2.csv"

    process_file(input_file, output_file)
