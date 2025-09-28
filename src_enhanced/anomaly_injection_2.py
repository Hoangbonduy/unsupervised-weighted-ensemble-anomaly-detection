import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from statsmodels.tsa.seasonal import seasonal_decompose

class AnomalyInjector:
    """
    Lớp này được cải tiến để tạo ra các bất thường đa dạng hơn bằng cách
    ngẫu nhiên hóa loại, cường độ, và vị trí của chúng.
    """
    def __init__(self, random_state=None):
        self.rng = np.random.default_rng(random_state)
        # Định nghĩa các loại bất thường có sẵn và các hàm tương ứng
        self.injection_methods = {
            'global': self._inject_global,
            'trend': self._inject_trend,
            'seasonal': self._inject_seasonal_shift,
            'cutoff': self._inject_cutoff,
            'scale': self._inject_scale,
            'shape_distortion': self._inject_shape_distortion,
            'gradual_level_shift': self._inject_gradual_level_shift
        }

    def _get_anomaly_location(self, series, length, strategy='low_variance'):
        """
        Tìm vị trí để chèn bất thường với nhiều chiến lược khác nhau.
        - 'low_variance': Ưu tiên vùng ít biến động.
        - 'random': Chọn vị trí ngẫu nhiên.
        """
        if len(series) <= length:
            return 0

        if strategy == 'random':
            return self.rng.integers(0, len(series) - length)

        # Mặc định là 'low_variance'
        window_size = max(10, length * 2)
        if len(series) < window_size:
            return self.rng.integers(0, len(series) - length)

        moving_std = pd.Series(series).rolling(window=window_size, center=True).std().bfill().ffill().to_numpy()
        # Ưu tiên 25% các vùng có độ biến động thấp nhất
        candidates = np.where(moving_std < np.quantile(moving_std, 0.25))[0]

        if len(candidates) > 0:
            # Lọc ra các candidate hợp lệ
            valid_candidates = candidates[candidates <= len(series) - length]
            if len(valid_candidates) > 0:
                start_idx = self.rng.choice(valid_candidates)
                return start_idx
        
        # Fallback to random if no suitable low-variance region is found
        return self.rng.integers(0, max(1, len(series) - length + 1))

    def _sigmoid(self, x):
        """
        Hàm sigmoid để tạo transition mượt mà.
        """
        return 1 / (1 + np.exp(-x))

    # --- CÁC HÀM INJECT ĐÃ ĐƯỢC CẬP NHẬT VỚI THAM SỐ NGẪU NHIÊN ---

    def _inject_global(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        num_anomalies = self.rng.integers(params.get("min_count", 1), params.get("max_count", 5))
        
        data_std = np.std(series)
        data_mean = np.mean(series)

        for _ in range(num_anomalies):
            loc = self.rng.integers(0, len(series))
            direction = self.rng.choice([-1, 1])
            # Ngẫu nhiên hóa cường độ
            magnitude = self.rng.uniform(params.get("min_multiplier", 3), params.get("max_multiplier", 10))
            anomaly_value = data_mean + direction * data_std * magnitude
            series[loc] = anomaly_value
            labels[loc] = 1
        return series, labels

    def _inject_trend(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params.get("min_len", 30), params.get("max_len", 100))
        
        # Đảm bảo length không vượt quá độ dài của series
        length = min(length, len(series))
        if length <= 0:
            return series, labels
        
        start = self._get_anomaly_location(series, length)
        
        # Đảm bảo start + length không vượt quá series
        if start + length > len(series):
            start = len(series) - length
        
        direction = self.rng.choice([-1, 1])
        # Ngẫu nhiên hóa độ dốc của trend
        strength = self.rng.uniform(params.get("min_strength", 0.05), params.get("max_strength", 0.2)) * np.std(series)
        
        trend = np.linspace(0, direction * strength, length)
        
        # Đảm bảo slice có cùng kích thước với trend
        actual_slice_length = len(series[start:start+length])
        if actual_slice_length != length:
            trend = np.linspace(0, direction * strength, actual_slice_length)
        
        series[start:start+length] += trend
        labels[start:start+length] = 1
        return series, labels

    def _inject_seasonal_shift(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        period = params.get("period", 7)
        if len(series) < 3 * period:
            return series, labels

        start = self.rng.integers(period, len(series) - 2 * period)
        segment_to_shift = series[start : start + period]

        # Dịch chuyển đoạn mùa vụ một cách ngẫu nhiên
        shift_amount = self.rng.integers(int(period / 4), int(period / 2))
        direction = self.rng.choice([-1, 1])
        shift = shift_amount * direction

        shifted_segment = np.roll(segment_to_shift, shift)

        series[start : start + period] = shifted_segment
        labels[start : start + period] = 1
        return series, labels

    def _inject_cutoff(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params.get("min_len", 10), params.get("max_len", 50))
        
        # Đảm bảo length không vượt quá độ dài của series
        length = min(length, len(series))
        if length <= 0:
            return series, labels
        
        start = self._get_anomaly_location(series, length)
        
        # Đảm bảo start + length không vượt quá series
        if start + length > len(series):
            start = len(series) - length
        
        # Ngẫu nhiên giá trị bị "kẹt"
        cutoff_value_options = [0, np.min(series), np.quantile(series, 0.05)]
        cutoff_value = self.rng.choice(cutoff_value_options)

        series[start:start+length] = cutoff_value
        labels[start:start+length] = 1
        return series, labels

    def _inject_average(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params.get("min_len", 15), params.get("max_len", 60))
        
        # Đảm bảo length không vượt quá độ dài của series
        length = min(length, len(series))
        if length <= 0:
            return series, labels
        
        start = self._get_anomaly_location(series, length)
        
        # Đảm bảo start + length không vượt quá series
        if start + length > len(series):
            start = len(series) - length
        
        segment = series[start:start+length]
        smoothing_strength = self.rng.uniform(params.get("min_strength", 0.2), params.get("max_strength", 0.8))
        
        smoothed_segment = pd.Series(segment).rolling(window=int(len(segment)*smoothing_strength), center=True, min_periods=1).mean().to_numpy()
        series[start:start+length] = smoothed_segment
        labels[start:start+length] = 1
        return series, labels

    # --- CÁC HÀM INJECT MỚI ĐƯỢC BỔ SUNG ---

    def _inject_scale(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params.get("min_len", 20), params.get("max_len", 80))
        
        # Đảm bảo length không vượt quá độ dài của series
        length = min(length, len(series))
        if length <= 0:
            return series, labels
        
        start = self._get_anomaly_location(series, length, strategy='random')
        
        # Đảm bảo start + length không vượt quá series
        if start + length > len(series):
            start = len(series) - length
        
        segment = series[start:start+length]
        segment_mean = segment.mean()
        # Ngẫu nhiên hóa hệ số co giãn
        scale_factor = self.rng.uniform(params.get("min_scale", 1.5), params.get("max_scale", 3.0))
        
        series[start:start+length] = (segment - segment_mean) * scale_factor + segment_mean
        labels[start:start+length] = 1
        return series, labels

    def _inject_shape_distortion(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)

        # Tìm một đỉnh sóng nổi bật để làm biến dạng
        peaks, _ = find_peaks(series, prominence=np.std(series) * 0.5, width=5)
        if len(peaks) == 0:
            return series, labels # Không có đỉnh nào đủ rõ để làm biến dạng

        peak_idx = self.rng.choice(peaks)

        # Lấy một đoạn xung quanh đỉnh
        width = self.rng.integers(params.get("min_width", 10), params.get("max_width", 30))
        start = max(0, peak_idx - width)
        end = min(len(series), peak_idx + width + 1)
        segment = series[start:end]

        if len(segment) < 3:
            return series, labels

        # Kéo giãn hoặc nén đoạn sóng này
        stretch_factor = self.rng.uniform(params.get("min_stretch", 1.5), params.get("max_stretch", 2.5))

        original_indices = np.linspace(0, 1, len(segment))
        new_length = int(len(segment) * stretch_factor)
        new_indices = np.linspace(0, 1, new_length)

        interp_func = interp1d(original_indices, segment, kind='cubic', fill_value="extrapolate")
        distorted_segment = interp_func(new_indices)

        # Thay thế lại vào chuỗi gốc (chỉ thay thế một phần)
        replace_len = min(len(distorted_segment), len(series) - start)
        series[start : start + replace_len] = distorted_segment[:replace_len]
        labels[start : start + replace_len] = 1

        return series, labels

    def _inject_gradual_level_shift(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params.get("min_len", 50), params.get("max_len", 150))
        
        # Đảm bảo length không vượt quá độ dài của series
        length = min(length, len(series))
        if length <= 0:
            return series, labels
        
        start = self._get_anomaly_location(series, length)

        # Đảm bảo start + length không vượt quá series
        if start + length > len(series):
            start = len(series) - length
            
        # Đảm bảo không vượt quá độ dài series
        actual_length = min(length, len(series) - start)
        if actual_length <= 0:
            return series, labels

        # Cường độ của sự thay đổi mức
        shift_magnitude = self.rng.uniform(1.5, 3.0) * np.std(series)
        direction = self.rng.choice([-1, 1])

        # Tạo ra một đường cong sigmoid mượt mà với độ dài thực tế
        transition_x = np.linspace(-10, 10, actual_length)
        sigmoid_transition = self._sigmoid(transition_x) * shift_magnitude * direction

        series[start:start+actual_length] += sigmoid_transition
        labels[start:start+actual_length] = 1
        return series, labels
        
    # --- HÀM ĐIỀU PHỐI CHÍNH ---

    def inject(self, series, num_anomalies_range=(4, 10), anomaly_type_dist=None):
        """
        Tiêm một số lượng ngẫu nhiên các bất thường ngẫu nhiên vào chuỗi.
        Hàm này đã được cải tiến để kiểm soát số lượng và sự phân bổ của các loại bất thường.

        Args:
            series (np.ndarray): Chuỗi dữ liệu đầu vào.
            num_anomalies_range (tuple): Khoảng (min, max) cho số lượng bất thường sẽ được tiêm.
                                         Đã tăng giới hạn trên mặc định để tạo sự đa dạng.
            anomaly_type_dist (dict or None): Phân phối xác suất để chọn loại bất thường.
                                              Ví dụ: {'global': 0.5, 'trend': 0.2, ...}
                                              Nếu là None, các loại sẽ được chọn ngẫu nhiên đều.
        """
        injected_series, final_labels = series.copy(), np.zeros_like(series, dtype=int)
        
        # Ngẫu nhiên hóa số lượng bất thường trong một khoảng rộng hơn
        num_to_inject = self.rng.integers(num_anomalies_range[0], num_anomalies_range[1] + 1)
        
        # Xác định các loại bất thường và xác suất lựa chọn của chúng
        if anomaly_type_dist:
            types = list(anomaly_type_dist.keys())
            probabilities = list(anomaly_type_dist.values())
            # Chuẩn hóa lại xác suất để tổng bằng 1
            probabilities = np.array(probabilities) / np.sum(probabilities)
        else:
            types = list(self.injection_methods.keys())
            probabilities = None # ngẫu nhiên đều

        for _ in range(num_to_inject):
            # Ngẫu nhiên chọn một loại bất thường để tiêm theo phân phối xác suất
            anomaly_type = self.rng.choice(types, p=probabilities)
            injection_func = self.injection_methods[anomaly_type]
            
            # (Tùy chọn) Định nghĩa các dải tham số ngẫu nhiên cho từng loại
            params = {} 
            
            # Đảm bảo không tiêm lại vào vùng đã có bất thường để tránh chồng chéo
            # Điều này giúp các bất thường trở nên rõ ràng hơn
            temp_series = injected_series.copy().astype(float)  # Chuyển đổi sang float để hỗ trợ NaN
            temp_series[final_labels == 1] = np.nan # Đánh dấu các vùng đã có bất thường

            injected_series, labels = injection_func(temp_series, params)
            
            # Ghi đè lại các giá trị NaN bằng giá trị gốc
            nan_mask = np.isnan(injected_series)
            injected_series[nan_mask] = series[nan_mask]
            
            final_labels = np.logical_or(final_labels, labels).astype(int)
            
        return injected_series, final_labels