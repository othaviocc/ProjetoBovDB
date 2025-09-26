import pandas as pd
import numpy as np
import os

class TradingStrategy:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):  # função para carregar e organizar o dados
        try:
            self.data = pd.read_csv(self.file_path)
            self.data['datetime'] = pd.to_datetime(self.data['date'] + ' ' + self.data['time'])
            self.data.set_index('datetime', inplace=True)
            self.data.drop(columns=['date', 'time'], inplace=True)
            print(f"Data loaded from {self.file_path} successfully.")
        except Exception as e:
            print(f"Error loading data from {self.file_path}: {e}")

    def add_technical_indicators(self):  # adicionar os indicadores de MA
        try:
            self.data = self.data.reset_index()
            self.data['date'] = self.data['datetime'].dt.date  # Adicionar coluna apenas com a data

            # SMA e EMA com períodos extras
            for window in [3, 5, 7, 9, 11]:
                # Calcula SMA
                self.data[f'SMA_{window}'] = (
                    self.data.groupby(['id_ticker', 'date'])['close']
                    .transform(lambda x: x.rolling(window=window, min_periods=window).mean().round(4))
                )

                # Calcula EMA
                self.data[f'EMA_{window}'] = (
                    self.data.groupby(['id_ticker', 'date'])['close']
                    .transform(lambda x: x.ewm(span=window, adjust=False).mean().round(4))
                )

            print("Technical indicators added successfully.")
        except Exception as e:
            print(f"Error adding technical indicators: {e}")

    def add_std_features(self):
        try:
            for window in [3, 5, 7, 9, 11]:
                # calcula para desvio padrão close e open
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

    def add_bollinger_bands(self, period=7, std_factor=0.7929549):
        try:
            grouped = self.data.groupby(['id_ticker', 'date'])
            rolling_mean = grouped['close'].transform(lambda x: x.rolling(window=period).mean())
            rolling_std = grouped['close'].transform(lambda x: x.rolling(window=period).std())

            self.data['Bollinger_Mid'] = rolling_mean.round(4)
            self.data['Bollinger_Upper'] = (rolling_mean + std_factor * rolling_std).round(4)
            self.data['Bollinger_Lower'] = (rolling_mean - std_factor * rolling_std).round(4)

            print("Bollinger Bands added successfully.")
        except Exception as e:
            print(f"Error adding Bollinger Bands: {e}")

    def add_ad_line(self):
        try:
            # Money Flow Multiplier
            mfm = ((self.data['close'] - self.data['low']) - (self.data['high'] - self.data['close'])) / (
                self.data['high'] - self.data['low']
            )
            mfm = mfm.fillna(0)

            # Money Flow Volume
            mfv = mfm * self.data['volume']

            # Chaikin A/D Line (cumulativo por ticker)
            self.data['AD_Line'] = self.data.groupby('id_ticker').apply(
                lambda g: mfv.loc[g.index].cumsum()
            ).reset_index(level=0, drop=True)

            print("Chaikin A/D Line added successfully.")
        except Exception as e:
            print(f"Error adding A/D Line: {e}")

    def add_adxr(self, period=14):
        try:
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']

            # True Range (TR)
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # +DM e -DM
            plus_dm = high.diff()
            minus_dm = low.diff() * -1

            plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
            minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

            # Smoothed values
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)

            # DX
            dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di))

            # ADX
            adx = dx.rolling(window=period).mean()

            # ADXR
            adxr = ((adx + adx.shift(period)) / 2).round(4)

            self.data['ADXR'] = adxr

            print("ADXR added successfully.")
        except Exception as e:
            print(f"Error adding ADXR: {e}")

    def save_data_to_csv(self, output_file):  # salva o arquivo de saida
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
    strategy.add_bollinger_bands(period=7, std_factor=0.7929549)
    strategy.add_ad_line()
    strategy.add_adxr(period=14)
    strategy.save_data_to_csv(output_file)


if __name__ == '__main__':
    input_file = r"C:\\Users\\othav\\BovDB.v2\\dados.csv"
    output_file = r"C:\\Users\\othav\\BovDB.v2\\dados_indicadores.csv"

    # Process the single CSV file
    process_single_file(input_file, output_file)
