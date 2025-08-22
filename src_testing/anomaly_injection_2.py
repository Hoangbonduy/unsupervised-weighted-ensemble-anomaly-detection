import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score

# ==============================================================================
# PHẦN 1: CÁC HÀM CƠ SỞ (GIỮ NGUYÊN)
# ==============================================================================

def find_dominant_period(series, min_period=4, max_period=100):
    if len(series) < 3: return 0
    fft_series = np.fft.fft(series)[1:]
    frequencies = np.fft.fftfreq(len(series))[1:]
    if len(fft_series) == 0: return 0
    peak_coeff = np.argmax(np.abs(fft_series))
    peak_freq = frequencies[peak_coeff]
    period = 1 / abs(peak_freq) if peak_freq != 0 else 0
    return int(np.clip(period, min_period, max_period))

def get_injection_location(series_len, anomaly_len, period):
    if period == 0 or series_len < period * 2 or anomaly_len >= series_len:
        return np.random.randint(0, max(1, series_len - anomaly_len))
    valid_cycle_starts = np.arange(0, series_len - anomaly_len, period)
    if len(valid_cycle_starts) == 0:
        return np.random.randint(0, max(1, series_len - anomaly_len))
    return np.random.choice(valid_cycle_starts)

# ==============================================================================
# PHẦN 2: 9 HÀM TIÊM BẤT THƯỜNG - CÓ THAM SỐ `seed`
# ==============================================================================

def inject_spike_anomaly(time_series, seed=42):
    np.random.seed(seed) # Cố định seed
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    if len(ts_copy) < 5: return ts_copy, pseudo_labels
    num_spikes = np.random.randint(1, 4)

    strength = np.random.uniform(1.5, 3.0)

    injection_points = np.random.choice(len(ts_copy), size=num_spikes, replace=False)
    spike_magnitude = np.std(ts_copy) * strength
    for point in injection_points:
        ts_copy[point] += spike_magnitude * np.random.choice([-1, 1])
        pseudo_labels[point] = 1
    return ts_copy, pseudo_labels

def inject_contextual_anomaly(time_series, seed=43):
    np.random.seed(seed) # Cố định seed
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    anomaly_len = int(series_len * np.random.uniform(0.05, 0.15))
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    a = np.random.choice([0.5, 1.5]) # Co lại 50% hoặc giãn ra 150%
    b = np.random.normal(0, 0.5 * np.std(ts_copy)) # Dịch chuyển mạnh hơn
    ts_copy[start:end] = a * ts_copy[start:end] + b
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

def inject_flip_anomaly(time_series, seed=44):
    np.random.seed(seed) # Cố định seed
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    anomaly_len = int(series_len * np.random.uniform(0.05, 0.1))
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    ts_copy[start:end] = ts_copy[start:end][::-1]
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

def inject_speedup_anomaly(time_series, seed=45):
    """
    Thay thế Speedup: Thêm một sóng sin tần số cao vào một đoạn.
    Hiệu ứng: Tạo ra một đoạn có 'sóng gợn' hoặc 'rung' rất rõ ràng.
    """
    np.random.seed(seed)
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    
    anomaly_len = int(series_len * 0.15)
    # THAY ĐỔI: Các tham số cho sóng sin
    amplitude = 0.5 * np.std(ts_copy) # Biên độ sóng
    frequency = np.random.uniform(5, 10) # Tần số sóng (cao hơn tần số tự nhiên)

    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len

    x = np.arange(anomaly_len)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * x / anomaly_len)
    
    ts_copy[start:end] += sine_wave
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

def inject_noise_anomaly(time_series, seed=46):
    """
    Tạo nhiễu tần số cao, biên độ thấp.
    Hiệu ứng: Làm cho đường tín hiệu trông 'dày' hoặc 'rung' hơn.
    """
    np.random.seed(seed)
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    
    anomaly_len = int(series_len * 0.15) # Cố định độ dài
    # THAY ĐỔI: Strength rất nhỏ để tạo nhiễu biên độ thấp
    strength = 0.1 
    
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    
    noise = np.random.normal(loc=0, scale=np.std(ts_copy) * strength, size=anomaly_len)
    ts_copy[start:end] += noise
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

def inject_cutoff_anomaly(time_series, seed=47):
    np.random.seed(seed) # Cố định seed
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    anomaly_len = int(series_len * np.random.uniform(0.05, 0.15))
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    loc = np.random.choice([0, 1])
    sigma = 0.05 * np.std(ts_copy)
    cutoff_segment = np.random.normal(loc, sigma, anomaly_len)
    ts_copy[start:end] = cutoff_segment
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

def inject_scale_anomaly(time_series, seed=48):
    np.random.seed(seed) # Cố định seed
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    anomaly_len = int(series_len * np.random.uniform(0.1, 0.2))
    scale_factor = np.random.choice([np.random.uniform(1.5, 2.5), np.random.uniform(0.2, 0.5)])
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    ts_copy[start:end] *= scale_factor
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

def inject_wander_anomaly(time_series, seed=49):
    np.random.seed(seed) # Cố định seed
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    anomaly_len = int(series_len * np.random.uniform(0.1, 0.2))
    strength = np.random.uniform(1.0, 2.0)
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    baseline = np.std(ts_copy) * strength * np.random.choice([-1, 1])
    trend = np.linspace(0, baseline, anomaly_len)
    ts_copy[start:end] += trend
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

def inject_average_anomaly(time_series, seed=50):
    np.random.seed(seed) # Cố định seed
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    anomaly_len = int(series_len * np.random.uniform(0.1, 0.2))
    window = max(3, int(anomaly_len * 0.2))
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    segment_series = pd.Series(ts_copy[start:end])
    moving_avg = segment_series.rolling(window=window, min_periods=1, center=True).mean().values
    ts_copy[start:end] = moving_avg
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

# ==============================================================================
# PHẦN 3: HÀM CHÍNH get_synthetic_ranking - CẬP NHẬT CÁCH GỌI HÀM
# ==============================================================================

def get_synthetic_ranking(time_series, models_to_run_scores, base_seed=42):
    injection_tests = {
        'spike': inject_spike_anomaly,
        'contextual': inject_contextual_anomaly,
        'flip': inject_flip_anomaly,
        'speedup': inject_speedup_anomaly,
        'noise': inject_noise_anomaly,
        'cutoff': inject_cutoff_anomaly,
        'scale': inject_scale_anomaly,
        'wander': inject_wander_anomaly,
        'average': inject_average_anomaly
    }
    
    all_auc_scores = {name: [] for name in models_to_run_scores.keys()}
    
    # Dùng enumerate để mỗi bài test có một seed khác nhau nhưng vẫn cố định
    for i, (test_name, inject_func) in enumerate(injection_tests.items()):
        # Mỗi hàm inject sẽ được gọi với một seed duy nhất và không đổi
        ts_injected, pseudo_labels = inject_func(time_series, seed=base_seed + i)
        
        ts_injected_df = pd.DataFrame(ts_injected)
        
        if len(np.unique(pseudo_labels)) < 2:
            for model_name in models_to_run_scores.keys():
                all_auc_scores[model_name].append(0.5)
            continue

        for model_name, run_func in models_to_run_scores.items():
            anomaly_scores = run_func(ts_injected_df)
            anomaly_scores = np.array(anomaly_scores)
            anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e6, neginf=-1e6)
            
            if len(anomaly_scores) == len(pseudo_labels) and np.all(np.isfinite(anomaly_scores)):
                try:
                    auc = roc_auc_score(pseudo_labels, anomaly_scores)
                    all_auc_scores[model_name].append(auc)
                except ValueError as e:
                    print(f"    Warning: AUC calculation failed for {model_name} on {test_name}: {e}")
                    all_auc_scores[model_name].append(0.0)
            else:
                print(f"    Warning: Invalid anomaly_scores for {model_name} on {test_name}")
                all_auc_scores[model_name].append(0.0)

    avg_scores = {name: np.mean(auc_list) for name, auc_list in all_auc_scores.items()}
    ranked_models = sorted(avg_scores, key=avg_scores.get, reverse=True)
    
    print("\n  Điểm AUC chi tiết trên các bài test tổng hợp:")
    print(pd.DataFrame(all_auc_scores, index=injection_tests.keys()).round(3))
    print("\n  Điểm AUC trung bình:")
    for model_name in ranked_models:
        print(f"    - {model_name}: {avg_scores[model_name]:.4f}")

    return ranked_models, avg_scores