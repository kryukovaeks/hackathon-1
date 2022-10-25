import pandas as pd
from abc import ABC, abstractmethod

from typing import Optional
from keras.models import load_model

class Strategy(ABC):

    @abstractmethod
    def required_rows(self):
        raise NotImplementedError("Specify required_rows!")

    @abstractmethod
    def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:
        assert len(current_data) == self.required_rows  # This much data will be fed to model

        return None  # If None is returned, no action is executed


class MeanReversionStrategy(Strategy):
    required_rows = 2*24*60   # minutes of data to be fed to model.

    def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:
        avg_price = current_data['price'].mean()
        current_price = current_data['price'][-1]

        target_position = current_position + (avg_price - current_price)/1000

        return target_position


class YourStrategy(Strategy):
    required_rows = 7*24*60  # Specify how many minutes of data are required for live prediction

    def __init__(self):
        training_data = pd.read_pickle("data/train_data.pickle")
        ...  # Use historical data to develop strategy, maybe train an ml_model etc.
        pass

    def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:
        current_price = current_data['price'][-1]
        model = load_model('models/two_weeks_lstm_reg.h5')
        num_mins = 60
        col_min = new_image.min(axis=0)
        col_max = new_image.max(axis=0)
        new_image = (new_image - col_min) / (col_max - col_min)
        images = np.zeros([0, num_mins, data.shape[1]])
        images[train_test_idx:, :, :]
        pred_price = model.predict(current_data)
        target_position = current_position + (pred_price - current_price)/1000

        return target_position  # produce inputs to model from datafram, compute predictions and submit new target position