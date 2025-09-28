import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

# --- PHẦN 1: TÁI TRIỂN KHAI LỚP ANOMALYINJECTOR (6 LOẠI CỐT LÕI) ---

class AnomalyInjector:
    def __init__(self, random_state=None):
        self.rng = np.random.default_rng(random_state)

    def _get_anomaly_location(self, series, length):
        # Đảm bảo series đủ dài để thực hiện rolling window
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
        # Bỏ 'noise' ra khỏi danh sách
        method_map = {
            "global": self._inject_global, "contextual": self._inject_contextual,
            "trend": self._inject_trend, "seasonal": self._inject_seasonal,
            "cutoff": self._inject_cutoff, "average": self._inject_average
        }
        return method_map[anomaly_type](series, params)

    # --- Các phương pháp được giữ lại ---
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
        """
        Injects seasonal anomalies by explicitly creating both spikes and dips
        that violate the seasonal pattern.
        - Spikes: Injects high values at seasonal troughs.
        - Dips: Injects low values at seasonal peaks.
        """
        series_copy, labels = series.copy(), np.zeros_like(series, dtype=int)
        
        period = params.get("period", 7)
        n_anomalies = params.get("n_anomalies", 4) # Nên là số chẵn để chia đều
        magnitude_std = params.get("magnitude_std", 2.5)

        if len(series_copy) < period * 2:
            return series_copy, labels

        try:
            # 1. Tách thành phần mùa vụ
            decomposition = seasonal_decompose(series_copy, model='additive', period=period)
            seasonal = decomposition.seasonal

            # 2. Tìm tất cả các đỉnh và đáy mùa vụ
            peaks, _ = find_peaks(seasonal)
            troughs, _ = find_peaks(-seasonal)

            # Nếu không có đủ đỉnh hoặc đáy, không thể chèn bất thường
            if len(peaks) == 0 or len(troughs) == 0:
                return series, labels

            # 3. Phân chia số lượng bất thường thành 2 loại: tăng (spikes) và giảm (dips)
            n_dips = n_anomalies // 2
            n_spikes = n_anomalies - n_dips
            
            magnitude = magnitude_std * np.std(series_copy)

            # 4a. Chèn các điểm "sụt giảm" (Dips) tại các đỉnh mùa vụ
            if n_dips > 0 and len(peaks) > 0:
                # Chọn ngẫu nhiên từ các vị trí đỉnh
                dip_indices = self.rng.choice(peaks, size=min(n_dips, len(peaks)), replace=False)
                for idx in dip_indices:
                    # Trừ đi một giá trị lớn để tạo ra một cú sụt giảm
                    series_copy[idx] -= magnitude * (1 + self.rng.random() * 0.5)
                    labels[idx] = 1

            # 4b. Chèn các điểm "tăng vọt" (Spikes) tại các đáy mùa vụ
            if n_spikes > 0 and len(troughs) > 0:
                # Chọn ngẫu nhiên từ các vị trí đáy
                spike_indices = self.rng.choice(troughs, size=min(n_spikes, len(troughs)), replace=False)
                for idx in spike_indices:
                    # Cộng thêm một giá trị lớn để tạo ra một đỉnh nhọn
                    series_copy[idx] += magnitude * (1 + self.rng.random() * 0.5)
                    labels[idx] = 1
                    
        except Exception as e:
            print(f"Error injecting seasonal anomaly: {e}")
            return series, labels
            
        return series_copy, labels

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

# --- PHẦN 2: LOGIC TRỰC QUAN HÓA ---

DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
TARGET_PLACEID = 4612088555142300004
OUTPUT_DIR = 'focused_injection_plots' # Đổi tên thư mục output

def visualize_injection(original_series, injected_series, labels, injection_type, profile_name, save_dir):
    plt.figure(figsize=(18, 7))
    plt.plot(original_series, label='Original Series', color='gray', alpha=0.6)
    plt.plot(injected_series, label='Injected Series', color='royalblue')
    anomaly_indices = np.where(labels == 1)[0]
    if len(anomaly_indices) > 0:
        plt.scatter(anomaly_indices, injected_series[anomaly_indices], 
                    color='red', label='Injected Anomalies', s=50, zorder=5)
    plt.title(f"Injection: '{injection_type}' with Profile: '{profile_name}' for PlaceID: {TARGET_PLACEID}")
    plt.xlabel('Time Step'); plt.ylabel('View'); plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    file_name = f"{injection_type}.png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, bbox_inches='tight'); plt.close()

def main():
    print("Bắt đầu quá trình trực quan hóa (bộ bất thường tập trung)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data_full = pd.read_csv(DATA_FILE_PATH)
    original_series = data_full[data_full['placeId'] == TARGET_PLACEID].sort_values('date')['view'].to_numpy().astype(float)
    n = len(original_series)
    print(f"Đã tải dữ liệu cho PlaceID {TARGET_PLACEID}, độ dài: {n} điểm.")

    injector = AnomalyInjector(random_state=42)
    
    # Danh sách các bất thường đã được tinh gọn
    injection_types = ["global", "trend", "seasonal", "cutoff", "average"]
    
    # Cấu hình tương ứng
    profiles = {
        "obvious": {
            "global": {"n_anomalies": 3, "magnitude_std": 7.0},
            # "contextual": {"n_anomalies": 3, "magnitude_std": 5.0},
            "trend": {"min_len": n//15, "max_len": n//8, "magnitude": np.std(original_series) * 2.0},
            "seasonal": {"n_anomalies": 4, "magnitude_std": 6.0, "period": 7},
            "cutoff": {"min_len": n//20, "max_len": n//10},
            "average": {"min_len": n//15, "max_len": n//8}
        }
    }

    for profile_name, params_config in profiles.items():
        print(f"\n--- Đang xử lý cấu hình: '{profile_name.upper()}' ---")
        profile_dir = os.path.join(OUTPUT_DIR, profile_name)
        os.makedirs(profile_dir, exist_ok=True)
        
        for method_name in tqdm(injection_types, desc=f"Profile '{profile_name}'"):
            params = params_config[method_name]
            injected_series, labels = injector.inject(original_series, method_name, params)
            visualize_injection(original_series, injected_series, labels, method_name, profile_name, profile_dir)
            
    print("\n" + "="*50)
    print("Hoàn thành!")
    print(f"Tất cả đồ thị đã được lưu vào: '{OUTPUT_DIR}'")
    print("="*50)

if __name__ == '__main__':
    main()