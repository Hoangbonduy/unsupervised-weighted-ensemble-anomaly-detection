import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from statsmodels.tsa.seasonal import seasonal_decompose

class AnomalyInjector:
    """
    Lớp này được viết lại để mô phỏng chính xác logic từ file inject_anomalies.py
    mà bạn đã cung cấp.
    """
    def __init__(self, random_state=None):
        self.rng = np.random.default_rng(random_state)

    def _get_anomaly_location(self, series, length):
        """
        Tìm vị trí "tốt" để chèn bất thường, ưu tiên các vùng ít biến động.
        """
        window_size = max(10, length * 2)
        if len(series) < window_size:
             if len(series) > length:
                return self.rng.integers(0, len(series) - length)
             else:
                return 0

        moving_std = pd.Series(series).rolling(window=window_size, center=True).std().bfill().ffill().to_numpy()
        candidates = np.where(moving_std < np.quantile(moving_std, 0.25))[0]

        if len(candidates) > length and len(series) > length:
            start_idx = self.rng.choice(candidates)
            return min(start_idx, len(series) - length)
        elif len(series) > length:
            return self.rng.integers(0, len(series) - length)
        else:
            return 0

    def inject(self, series, anomaly_type, params):
        """
        Hàm điều phối, gọi đúng phương thức chèn bất thường.
        """
        method_map = {
            "global": self._inject_global, "contextual": self._inject_contextual,
            "trend": self._inject_trend, "seasonal": self._inject_seasonal,
            "noise": self._inject_noise, "cutoff": self._inject_cutoff,
            "average": self._inject_average, "amplitude": self._inject_amplitude,
            "shapelet": self._inject_shapelet, "width": self._inject_width
        }
        
        if anomaly_type not in method_map:
            raise ValueError(f"Loại bất thường không hợp lệ: {anomaly_type}")
            
        return method_map[anomaly_type](series, params)

    def _inject_global(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        indices = self.rng.choice(len(series), params["n_anomalies"], replace=False)
        for idx in indices:
            series[idx] += self.rng.choice([-1, 1]) * params["magnitude_std"] * np.std(series)
            labels[idx] = 1
        return series, labels

    def _inject_contextual(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        indices = self.rng.choice(len(series), params["n_anomalies"], replace=False)
        for idx in indices:
            series[idx] = np.mean(series) + self.rng.choice([-1, 1]) * params["magnitude_std"] * np.std(series)
            labels[idx] = 1
        return series, labels

    def _inject_trend(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params["min_len"], params["max_len"])
        start = self._get_anomaly_location(series, length)
        trend = np.linspace(0, self.rng.choice([-1, 1]) * params["magnitude"], length)
        series[start:start+length] += trend
        labels[start:start+length] = 1
        return series, labels

    def _inject_seasonal(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        period = params["period"]
        if len(series) < period * 2: return series, labels
        try:
            seasonal = seasonal_decompose(series, model='additive', period=period).seasonal
            peaks, _ = find_peaks(seasonal, height=0)
            if len(peaks) == 0: return series, labels
            indices = self.rng.choice(peaks, min(params["n_anomalies"], len(peaks)), replace=False)
            for idx in indices:
                series[idx] += self.rng.choice([-1, 1]) * params["magnitude_std"] * np.std(series)
                labels[idx] = 1
        except Exception:
            return series, labels
        return series, labels

    def _inject_noise(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params["min_len"], params["max_len"])
        start = self._get_anomaly_location(series, length)
        noise = self.rng.normal(0, np.std(series) * params["noise_std"], length)
        series[start:start+length] += noise
        labels[start:start+length] = 1
        return series, labels

    def _inject_cutoff(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params["min_len"], params["max_len"])
        start = self._get_anomaly_location(series, length)
        cutoff_value = np.quantile(series, 0.05)
        series[start:start+length] = cutoff_value
        labels[start:start+length] = 1
        return series, labels

    def _inject_average(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params["min_len"], params["max_len"])
        start = self._get_anomaly_location(series, length)
        window = params.get("smoothing_window", 5)
        segment = series[start:start+length]
        smoothed_segment = pd.Series(segment).rolling(window, center=True, min_periods=1).mean().to_numpy()
        series[start:start+length] = smoothed_segment
        labels[start:start+length] = 1
        return series, labels

    def _inject_amplitude(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params["min_len"], params["max_len"])
        start = self._get_anomaly_location(series, length)
        segment = series[start:start+length]
        segment_mean = np.mean(segment)
        series[start:start+length] = (segment - segment_mean) * params["scale_factor"] + segment_mean
        labels[start:start+length] = 1
        return series, labels

    def _inject_shapelet(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params["min_len"], params["max_len"])
        start = self._get_anomaly_location(series, length)
        segment = series[start:start+length]
        series[start:start+length] = segment[::-1]
        labels[start:start+length] = 1
        return series, labels
        
    def _inject_width(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        peaks, _ = find_peaks(series, prominence=np.std(series) * 0.5)
        if len(peaks) == 0: return series, labels
        peak_idx = self.rng.choice(peaks)
        width = params["width"]
        start = max(0, peak_idx - width)
        end = min(len(series), peak_idx + width)
        segment = series[start:end]
        if len(segment) < 2: return series, labels
        stretch_factor = params["stretch_factor"]
        new_len = int(len(segment) * stretch_factor)
        interp_func = interp1d(np.arange(len(segment)), segment, kind='linear', fill_value="extrapolate")
        stretched_segment = interp_func(np.linspace(0, len(segment) - 1, new_len))
        series[start:end] = stretched_segment[:len(segment)]
        labels[start:end] = 1
        return series, labels