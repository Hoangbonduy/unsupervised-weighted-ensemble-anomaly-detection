import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import t
from matplotlib import pyplot
import warnings
import sranodec as anom
import joblib
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from momentfm import MOMENTPipeline
import warnings
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from momentfm import MOMENTPipeline
import warnings

warnings.filterwarnings("ignore")

WEIGHTS_DIR = 'pretrained_weights/'
WINDOW_SIZE = 128

# Tắt TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Chỉ hiện ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tắt oneDNN warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, pd.Series):
        return x.values
    else:
        return np.array(x)

# --- 1. MÃ NGUỒN CỦA SH-ESD (BẠN ĐÃ CUNG CẤP) ---

def seasonal_mean(x, freq):
  return np.array([pd.Series(x[i::freq]).mean(skipna=True) for i in range(freq)])

def ts_S_Md_decomposition(x, freq):
  nobs = len(x)
  period_averages = seasonal_mean(x, freq)
  seasonal = np.tile(period_averages, nobs // freq + 1)[:nobs]
  med = np.tile(pd.Series(x).median(skipna=True), nobs)
  res = np.array(x) - seasonal - med
  return {"observed": np.array(x), "seasonal": seasonal, "median":med, "residual":res}

def esd_test_statistics(x, hybrid=True):
  if hybrid:
    location = pd.Series(x).median(skipna=True)
    dispersion = np.median(np.abs(x - np.median(x)))
  else:  
    location = pd.Series(x).mean(skipna=True)
    dispersion = pd.Series(x).std(skipna=True)
  return location, dispersion    

def esd_test(x, freq, alpha=0.95, ub=0.499, hybrid=True):
  nobs = len(x)
  if ub > 0.4999:
    ub = 0.499
  k = max(int(np.floor(ub * nobs)), 1)
  res_tmp = ts_S_Md_decomposition(x, freq)["residual"]
  res = np.ma.array(res_tmp, mask=False)
  anomalies = []
  for i in range(1, k+1):
    location, dispersion = esd_test_statistics(res, hybrid)
    if dispersion == 0: # Thêm safeguard để tránh chia cho 0
        break
    tmp = np.abs(res - location) / dispersion
    idx = np.argmax(tmp)
    test_statistic = tmp[idx] 
    n = nobs - res.mask.sum()
    if n <= 2: # Thêm safeguard để tránh lỗi với hàm t.ppf
        break
    p = 1 - alpha / (2 * n)
    critical_value = t.ppf(p, n - 2)
    if test_statistic > critical_value:
      anomalies.append(idx)
    res.mask[idx] = True  
  return anomalies

# --- 2. CÁC HÀM BAO BỌC (WRAPPERS) - ĐÃ CẢI TIẾN ---

def find_dominant_period(series):
    if len(series) < 3: return 0
    fft_series = np.fft.fft(series)[1:]
    frequencies = np.fft.fftfreq(len(series))[1:]
    if len(fft_series) == 0: return 0
    peak_coeff = np.argmax(np.abs(fft_series))
    peak_freq = frequencies[peak_coeff]
    return 1 / abs(peak_freq) if peak_freq != 0 else 0

def run_sr_scores(X):
    """
    Chạy Spectral Residual với tham số đã được tối ưu từ tập dữ liệu lớn.
    """
    series = X.iloc[:, 0].to_numpy()
    
    # Kiểm tra điều kiện tối thiểu cho SR
    if len(series) < 100:  # Tăng ngưỡng từ 50 lên 100
        # Nếu chuỗi quá ngắn, trả về điểm số dựa trên fallback
        mean_val = np.mean(series)
        std_val = np.std(series)
        if std_val == 0:
            return np.zeros(len(series))
        return np.abs((series - mean_val) / std_val)
    
    try:
        # Nạp tham số tối ưu từ file
        try:
            sr_params = joblib.load(os.path.join('pretrained_weights', 'sr_params.joblib'))
            amp_window_size = sr_params['amp_window_size'] 
            series_window_size = sr_params['series_window_size']
            score_window_size = sr_params['score_window_size']
        except:
            # Fallback nếu không tìm thấy file
            amp_window_size, series_window_size, score_window_size = 36, 36, 50
        
        # Điều chỉnh tham số cho chuỗi ngắn để tránh lỗi index out of bounds
        safe_score_window = min(score_window_size, len(series) // 3)  # Không vượt quá 1/3 độ dài chuỗi
        safe_amp_window = min(amp_window_size, len(series) // 4)      # Không vượt quá 1/4 độ dài chuỗi
        safe_series_window = min(series_window_size, len(series) // 4) # Không vượt quá 1/4 độ dài chuỗi
        
        # Đảm bảo các tham số tối thiểu
        safe_score_window = max(safe_score_window, 10)
        safe_amp_window = max(safe_amp_window, 5)
        safe_series_window = max(safe_series_window, 5)
        
        # Tạo mô hình SR với tham số an toàn
        spec = anom.Silency(safe_amp_window, safe_series_window, safe_score_window)
        score = spec.generate_anomaly_score(series)
        
        # Trả về điểm số, xử lý NaN và infinity
        score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
        return score
        
    except (IndexError, ValueError, RuntimeError) as e:
        # Nếu SR thất bại, trả về điểm số dựa trên độ lệch so với trung bình
        print(f"Warning: SR failed, using fallback method. Error: {e}")
        mean_val = np.mean(series)
        std_val = np.std(series)
        if std_val == 0:
            return np.zeros(len(series))
        return np.abs((series - mean_val) / std_val)
    

def run_sr_predictions(X, contamination_rate=None):
    """
    Chạy SR với tham số đã được pretrained và trả về nhãn dự đoán 0/1.
    """
    scores = run_sr_scores(X)  # Đã sử dụng tham số pretrained
    
    if len(scores) == 0:
        return np.zeros(len(X)).astype(int)
    
    # Nếu contamination_rate không được cung cấp, sử dụng giá trị tối ưu
    if contamination_rate is None:
        # Sử dụng contamination rate được tối ưu từ quá trình training
        try:
            sr_params = joblib.load(os.path.join('pretrained_weights', 'sr_params.joblib'))
            # Nếu có thêm thông tin về contamination rate tối ưu trong file
            contamination_rate = sr_params.get('optimal_contamination_rate', 0.03)
        except:
            # Fallback: ước tính từ phân phối scores
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            if std_score > 0:
                # Sử dụng statistical method để ước tính contamination rate
                high_anomaly_count = np.sum(scores > mean_score + 2*std_score)
                contamination_rate = max(0.02, min(0.08, high_anomaly_count / len(scores)))
            else:
                contamination_rate = 0.03
    
    if np.std(scores) > 1e-8:
        threshold = np.percentile(scores, 100 * (1 - contamination_rate))
        predictions = (scores > threshold).astype(int)
        
        # Đảm bảo luôn có ít nhất một số anomaly được phát hiện
        if np.sum(predictions) == 0:
            # Giảm threshold xuống 95th percentile
            threshold = np.percentile(scores, 95)
            predictions = (scores > threshold).astype(int)
            
            # Nếu vẫn không có, chọn top 2% điểm cao nhất
            if np.sum(predictions) == 0:
                n_anomalies = max(1, int(0.02 * len(scores)))
                top_indices = np.argsort(scores)[-n_anomalies:]
                predictions = np.zeros(len(scores)).astype(int)
                predictions[top_indices] = 1
        
        return predictions
    
    # Fallback: chọn top 3% điểm cao nhất
    n_anomalies = max(1, int(0.03 * len(X)))
    top_indices = np.argsort(scores)[-n_anomalies:]
    predictions = np.zeros(len(X)).astype(int)
    predictions[top_indices] = 1
    return predictions

# --- TẢI SCALER CHUNG MỘT LẦN ---
# Đoạn code này nên được đặt ở đầu script đánh giá của bạn
try:
    SCALER_PATH = os.path.join(WEIGHTS_DIR, 'main_scaler.gz')
    SCALER = joblib.load(SCALER_PATH)
    print(f"Đã tải Scaler đã được huấn luyện trước từ: {SCALER_PATH}")
except Exception as e:
    print(f"LỖI NGHIÊM TRỌNG: Không thể tải file scaler 'main_scaler.gz'. Lỗi: {e}")
    print("Vui lòng chạy lại script pretrain_models.py để tạo file này.")
    SCALER = None # Gán là None để các hàm sau có thể kiểm tra

# Hàm tiện ích (không đổi)
def _to_numpy(X):
    if isinstance(X, pd.DataFrame): return X.iloc[:, 0].to_numpy()
    if isinstance(X, pd.Series): return X.to_numpy()
    if isinstance(X, np.ndarray): return X.flatten()
    return np.array(X)

def statistical_fallback(series):
    """
    Fallback method: returns standardized absolute deviation scores.
    """
    mean_val = np.mean(series)
    std_val = np.std(series)
    if std_val == 0:
        return np.zeros(len(series))
    return np.abs((series - mean_val) / std_val)

def run_iforest_scores(X):
    """Chạy IsolationForest đã được train với cải thiện."""
    series = _to_numpy(X)
    if len(series) < 2:
        return np.zeros(len(series))
        
    try:
        # Load model và scaler đã train
        iforest = joblib.load(os.path.join('pretrained_weights', 'iforest_pretrained.joblib'))
        scaler = joblib.load(os.path.join('pretrained_weights', 'iforest_scaler.joblib'))
        
        # Scale dữ liệu
        X_scaled = scaler.transform(series.reshape(-1, 1))
        
        # Tính scores
        scores = -iforest.decision_function(X_scaled)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
        
    except Exception as e:
        print(f"Warning: Pre-trained IForest failed, using fallback. Error: {e}")
        # Fallback method
        return statistical_fallback(series)

# --- Các hàm trả về PREDICTIONS 0/1 (để chạy sau khi đã chọn model) ---

def run_iforest_predictions(X, contamination_rate=None):
    """
    Chạy IForest với model đã được pretrained và threshold thích ứng.
    """
    scores = run_iforest_scores(X)
    
    if len(scores) == 0 or np.std(scores) <= 1e-8:
        return np.zeros(len(X)).astype(int)
    
    # Adaptive contamination rate dựa trên data length
    if contamination_rate is None:
        if len(X) < 100:
            contamination_rate = 0.08  # 8% cho chuỗi ngắn
        elif len(X) < 300:
            contamination_rate = 0.05  # 5% cho chuỗi trung bình
        else:
            contamination_rate = 0.03  # 3% cho chuỗi dài
    
    # Load thresholds đã được tính toán từ quá trình training
    try:
        thresholds = joblib.load(os.path.join('pretrained_weights', 'iforest_thresholds.joblib'))
        # Sử dụng threshold linh hoạt
        if contamination_rate >= 0.05:
            threshold = thresholds['threshold_95']
        else:
            threshold = thresholds['threshold_97']
    except:
        # Fallback: tính threshold từ scores hiện tại
        threshold = np.percentile(scores, 100 * (1 - contamination_rate))
    
    # Multiple threshold strategies để tránh F1 = 0
    predictions = (scores > threshold).astype(int)
    
    # Strategy 1: Đảm bảo có ít nhất một số anomaly được phát hiện
    if np.sum(predictions) == 0:
        # Giảm threshold xuống 90th percentile
        backup_threshold = np.percentile(scores, 90)
        predictions = (scores > backup_threshold).astype(int)
        
        # Nếu vẫn không có, chọn top 5% điểm cao nhất
        if np.sum(predictions) == 0:
            n_anomalies = max(1, int(0.05 * len(scores)))
            top_indices = np.argsort(scores)[-n_anomalies:]
            predictions = np.zeros(len(scores), dtype=int)
            predictions[top_indices] = 1
    
    # Strategy 2: Cap maximum anomalies at 20% của data để tránh quá nhiều false positives
    max_anomalies = max(1, int(0.20 * len(scores)))
    if np.sum(predictions) > max_anomalies:
        top_indices = np.argsort(scores)[-max_anomalies:]
        predictions = np.zeros(len(scores), dtype=int)
        predictions[top_indices] = 1
    
    return predictions

# ==============================================================================
# PHẦN 4: MOVING AVERAGE VÀ IQR ANOMALY DETECTION
# ==============================================================================

def run_moving_average_scores(X, window_size=7):
    """
    Phát hiện bất thường bằng phương pháp Moving Average.
    Tính sai số giữa giá trị thực tế và trung bình trượt, sau đó áp dụng quy tắc 3-sigma.
    """
    series = _to_numpy(X)
    if isinstance(X, pd.DataFrame):
        series = X.iloc[:, 0].values
    elif isinstance(X, pd.Series):
        series = X.values
    
    if len(series) < window_size:
        return np.zeros(len(series))
    
    try:
        # Tính trung bình trượt với center=True để giảm lag
        series_df = pd.Series(series)
        moving_avg = series_df.rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Tính sai số so với trung bình trượt
        errors = series - moving_avg.values
        
        # Áp dụng quy tắc 3-sigma để tính điểm bất thường
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        
        if error_std == 0:
            return np.zeros(len(series))
        
        # Điểm số bất thường = |sai số chuẩn hóa|
        anomaly_scores = np.abs((errors - error_mean) / error_std)
        
        return np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e6, neginf=0.0)
        
    except Exception as e:
        print(f"Warning: Moving Average failed, using fallback. Error: {e}")
        return statistical_fallback(series)

def run_moving_average_predictions(X, window_size=7, contamination_rate=None):
    """
    Chạy Moving Average và trả về nhãn dự đoán 0/1.
    """
    scores = run_moving_average_scores(X, window_size)
    
    if len(scores) == 0:
        return np.zeros(len(X)).astype(int)
    
    # Adaptive contamination rate
    if contamination_rate is None:
        if len(X) < 50:
            contamination_rate = 0.1  # 10% cho chuỗi rất ngắn
        elif len(X) < 200:
            contamination_rate = 0.05  # 5% cho chuỗi ngắn
        else:
            contamination_rate = 0.03  # 3% cho chuỗi dài
    
    if np.std(scores) > 1e-8:
        # Sử dụng quy tắc 3-sigma: điểm > 3 được coi là bất thường
        threshold = 3.0
        predictions = (scores > threshold).astype(int)
        
        # Fallback nếu không có anomaly nào được phát hiện
        if np.sum(predictions) == 0:
            threshold = np.percentile(scores, 100 * (1 - contamination_rate))
            predictions = (scores > threshold).astype(int)
        
        return predictions
    
    return np.zeros(len(X)).astype(int)

def run_iqr_scores(X, use_log_transform=False):
    """
    Phát hiện bất thường bằng phương pháp IQR.
    Hỗ trợ cả IQR thường và IQR với log transform cho dữ liệu có phân phối lệch.
    """
    series = _to_numpy(X)
    if isinstance(X, pd.DataFrame):
        series = X.iloc[:, 0].values
    elif isinstance(X, pd.Series):
        series = X.values
    
    if len(series) < 4:  # Cần ít nhất 4 điểm để tính IQR
        return np.zeros(len(series))
    
    try:
        if use_log_transform:
            # Áp dụng log transform cho dữ liệu có phân phối lệch
            log_series = np.log1p(series)  # log1p để tránh log(0)
            Q1 = np.percentile(log_series, 25)
            Q3 = np.percentile(log_series, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Tính điểm bất thường trong không gian log
            anomaly_scores = np.maximum(
                np.maximum(lower_bound - log_series, 0),  # Bất thường thấp
                np.maximum(log_series - upper_bound, 0)   # Bất thường cao
            )
        else:
            # IQR thường
            Q1 = np.percentile(series, 25)
            Q3 = np.percentile(series, 75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                return np.zeros(len(series))
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Tính điểm bất thường
            anomaly_scores = np.maximum(
                np.maximum(lower_bound - series, 0),  # Bất thường thấp
                np.maximum(series - upper_bound, 0)   # Bất thường cao
            )
            
            # Chuẩn hóa điểm số bằng IQR
            anomaly_scores = anomaly_scores / IQR
        
        return np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e6, neginf=0.0)
        
    except Exception as e:
        print(f"Warning: IQR failed, using fallback. Error: {e}")
        return statistical_fallback(series)

def run_iqr_predictions(X, use_log_transform=False, contamination_rate=None):
    """
    Chạy IQR và trả về nhãn dự đoán 0/1.
    """
    scores = run_iqr_scores(X, use_log_transform)
    
    if len(scores) == 0:
        return np.zeros(len(X)).astype(int)
    
    # Adaptive contamination rate
    if contamination_rate is None:
        contamination_rate = 0.05  # Default 5% cho IQR
    
    if np.std(scores) > 1e-8:
        # Bất kỳ điểm nào có score > 0 đều được coi là bất thường theo định nghĩa IQR
        predictions = (scores > 0).astype(int)
        
        # Nếu có quá nhiều anomalies, giới hạn theo contamination_rate
        if np.sum(predictions) > len(X) * contamination_rate * 3:  # Cho phép 3x contamination_rate
            threshold = np.percentile(scores[scores > 0], 100 * (1 - contamination_rate))
            predictions = (scores > threshold).astype(int)
        
        return predictions
    
    return np.zeros(len(X)).astype(int)

def run_iqr_combined_scores(X):
    """
    Kết hợp IQR thường và IQR-Log để phát hiện cả bất thường cao và thấp.
    """
    series = _to_numpy(X)
    if isinstance(X, pd.DataFrame):
        series = X.iloc[:, 0].values
    elif isinstance(X, pd.Series):
        series = X.values
    
    if len(series) < 4:
        return np.zeros(len(series))
    
    try:
        # IQR thường cho bất thường cao
        iqr_normal_scores = run_iqr_scores(X, use_log_transform=False)
        
        # IQR-Log cho bất thường thấp
        iqr_log_scores = run_iqr_scores(X, use_log_transform=True)
        
        # Kết hợp hai điểm số
        combined_scores = np.maximum(iqr_normal_scores, iqr_log_scores)
        
        return combined_scores
        
    except Exception as e:
        print(f"Warning: IQR Combined failed, using fallback. Error: {e}")
        return statistical_fallback(series)

def run_iqr_combined_predictions(X, contamination_rate=None):
    """
    Chạy IQR kết hợp và trả về nhãn dự đoán 0/1.
    """
    scores = run_iqr_combined_scores(X)
    
    if len(scores) == 0:
        return np.zeros(len(X)).astype(int)
    
    # Adaptive contamination rate
    if contamination_rate is None:
        contamination_rate = 0.05  # Default 5% cho IQR combined
    
    if np.std(scores) > 1e-8:
        predictions = (scores > 0).astype(int)
        
        # Giới hạn số lượng anomalies
        if np.sum(predictions) > len(X) * contamination_rate * 3:
            threshold = np.percentile(scores[scores > 0], 100 * (1 - contamination_rate))
            predictions = (scores > threshold).astype(int)
        
        return predictions
    
    return np.zeros(len(X)).astype(int)

def run_ma_inverted_scores(X, window_size=7):
    """
    Chạy Moving Average trên dữ liệu ĐÃ LẬT NGƯỢC để tìm các điểm rơi (dips).
    Về bản chất, đây là một mô hình chuyên phát hiện các điểm có giá trị thấp bất thường.
    """
    # Tạo một bản sao để không làm thay đổi dữ liệu gốc
    X_inverted = X.copy()
    
    # Lật ngược chuỗi thời gian bằng cách nhân với -1
    X_inverted.iloc[:, 0] = X_inverted.iloc[:, 0] * -1
    
    # Chạy mô hình Moving Average gốc trên dữ liệu đã lật ngược
    # Mô hình sẽ tìm "đỉnh nhọn" trên chuỗi bị lật, tương đương với "điểm rơi" trên chuỗi gốc
    inverted_scores = run_moving_average_scores(X_inverted, window_size=window_size)
    
    return inverted_scores

def run_ma_inverted_predictions(X, window_size=7, contamination_rate=None):
    """
    Trả về dự đoán 0/1 cho mô hình MA lật ngược.
    """
    scores = run_ma_inverted_scores(X, window_size)
    if len(scores) == 0 or np.std(scores) <= 1e-8:
        return np.zeros(len(X)).astype(int)

    # Adaptive contamination rate
    if contamination_rate is None:
        if len(X) < 50:
            contamination_rate = 0.1
        elif len(X) < 200:
            contamination_rate = 0.05
        else:
            contamination_rate = 0.03

    threshold = 3.0 # Vẫn giữ ngưỡng 3-sigma
    predictions = (scores > threshold).astype(int)

    if np.sum(predictions) == 0:
        threshold = np.percentile(scores, 100 * (1 - contamination_rate))
        predictions = (scores > threshold).astype(int)
    
    return predictions

def run_knn_scores(X):
    """
    Chạy k-Nearest Neighbors (KNN) để phát hiện bất thường và trả về điểm số.
    Mô hình này rất giỏi trong việc tìm ra các điểm/đoạn dữ liệu 'lạc lõng' so với phần còn lại.
    """
    # Chuyển đổi dữ liệu sang định dạng numpy array 2D
    series = _to_numpy(X).reshape(-1, 1)
    if len(series) < 2:
        return np.zeros(len(series))

    try:
        # Khởi tạo mô hình KNN từ thư viện PyOD
        # n_neighbors=5 là một giá trị khởi đầu tốt, có thể được tinh chỉnh sau này
        clf = KNN(n_neighbors=5, method='mean')
        
        # Huấn luyện mô hình trên dữ liệu
        clf.fit(series)
        
        # Lấy điểm số bất thường. Điểm càng cao, càng bất thường.
        scores = clf.decision_scores_
        
        # Xử lý các giá trị không hợp lệ và trả về
        return np.nan_to_num(scores)

    except Exception as e:
        print(f"Warning: KNN failed, using fallback. Error: {e}")
        # Sử dụng fallback thống kê nếu có lỗi
        return statistical_fallback(series.flatten())

# ==============================================================================
# PHẦN 5: RePAD FORECASTING ANOMALY SCORES
# ==============================================================================

_REPAD_MODEL = None  # cache model sau lần load đầu tiên

def _load_repad_model(model_path=os.path.join(WEIGHTS_DIR, 'repad_forecasting_model.keras')):
    """Lazy load mô hình RePAD forecasting. Trả về None nếu không tồn tại."""
    global _REPAD_MODEL
    if _REPAD_MODEL is not None:
        return _REPAD_MODEL  # đã cache

    # Kiểm tra tồn tại trước
    if not os.path.exists(model_path):
        print(f"[RePAD] Không tìm thấy model tại {model_path}")
        return None

    try:
        from tensorflow import keras
        _REPAD_MODEL = keras.models.load_model(model_path)
        print(f"[RePAD] Đã load model từ {model_path}")
    except Exception as e:
        print(f"[RePAD] Lỗi load model: {e}")
        _REPAD_MODEL = None
    return _REPAD_MODEL

def _create_repad_sequences(data, time_steps):
    """Tạo (X, y) theo đúng logic inference gốc: X =  (time_steps-1) giá trị, y = giá trị kế cuối của cửa sổ."""
    X, y = [], []
    n = len(data)
    if n < time_steps:
        return np.array([]), np.array([])
    # data là array shape (n,1)
    for i in range(n - time_steps + 1):
        X.append(data[i:i + time_steps - 1])        # 0 .. time_steps-2 (length time_steps-1)
        y.append(data[i + time_steps - 1])          # phần tử thứ time_steps-1 (0-index)
    return np.array(X), np.array(y)

def run_repad_scores(X, time_steps=30, scaler=None, smooth=False, debug=False):
    """
    Trả về anomaly scores dựa trên lỗi dự báo của mô hình RePAD.

    Tham số:
        X: DataFrame (1 cột) hoặc Series hoặc ndarray 1D.
        time_steps: cửa sổ thời gian RePAD (mặc định 30 như inference gốc).
        scaler: tuỳ chọn, nếu truyền vào phải có fit_transform / transform (vd MinMaxScaler). Nếu None sẽ fit mới trên chuỗi.
        smooth: nếu True sẽ áp dụng smoothing nhẹ (rolling mean 3) lên lỗi trước khi trả về.

    Trả về:
        np.ndarray length = len(series), trong đó các vị trí đầu (time_steps-1) không có dự báo sẽ gán 0.
    """
    series = _to_numpy(X)
    if len(series) == 0:
        return np.array([])
    if len(series) < time_steps:
        # Không đủ độ dài để tạo sequence -> trả về zeros
        return np.zeros(len(series))

    # Scale chuỗi về [0,1] như logic inference (dùng MinMaxScaler)
    try:
        if scaler is None:
            scaler = MinMaxScaler()
            series_scaled = scaler.fit_transform(series.reshape(-1, 1)).astype(np.float32)
        else:
            series_scaled = scaler.transform(series.reshape(-1, 1)).astype(np.float32)
    except Exception as e:
        print(f"[RePAD] Lỗi scaler: {e} -> dùng series chuẩn hoá min-max thủ công")
        s_min, s_max = np.min(series), np.max(series)
        rng = s_max - s_min if (s_max - s_min) > 0 else 1.0
        series_scaled = ((series - s_min) / rng).reshape(-1, 1).astype(np.float32)

    # Tạo sequences
    X_seq, y_true = _create_repad_sequences(series_scaled, time_steps)
    if X_seq.size == 0:
        return np.zeros(len(series))

    # Load model
    model = _load_repad_model()
    if model is None:
        if debug:
            print("[RePAD] Model load failed -> returning zeros")
        return np.zeros(len(series))

    try:
        y_pred = model.predict(X_seq, verbose=0)
    except Exception as e:
        print(f"[RePAD] Lỗi dự báo: {e}")
        return np.zeros(len(series))

    # Tính lỗi dự báo (absolute error)
    try:
        prediction_error = np.abs(y_pred - y_true).flatten()
    except Exception:
        # fallback nếu shape lệch
        prediction_error = np.linalg.norm(y_pred - y_true, axis=-1)

    # Ánh xạ lỗi về độ dài gốc: các phần đầu chưa có dự báo -> 0
    scores = np.zeros(len(series))
    usable_len = min(len(prediction_error), len(series) - (time_steps - 1))
    scores[time_steps - 1: time_steps - 1 + usable_len] = prediction_error[:usable_len]

    if smooth and usable_len > 3:
        # smoothing nhẹ bằng rolling mean 3
        try:
            import pandas as pd
            s = pd.Series(scores)
            scores = s.rolling(window=3, min_periods=1, center=True).mean().to_numpy()
        except Exception:
            pass

    # Chuẩn hoá scores về [0,1] để đồng nhất với các detector khác (tránh scale quá lớn)
    max_score = np.max(scores)
    if max_score > 0:
        scores = scores / (max_score + 1e-12)

    if debug:
        nonzero = scores[scores > 0]
        if nonzero.size:
            print(f"[RePAD][debug] raw_error_stats: min={nonzero.min():.4g} max={nonzero.max():.4g} mean={nonzero.mean():.4g} std={nonzero.std():.4g} count={nonzero.size}")
        else:
            print("[RePAD][debug] all scores zero (no usable sequences)")

    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

# ==============================================================================
# PHẦN 6: DeepAnT FORECASTING ANOMALY SCORES
# ==============================================================================
_DEEPANT_MODEL = None

def _load_deepant_model(model_path=os.path.join(WEIGHTS_DIR, 'deepant_model.keras')):
    """Lazy load mô hình DeepAnT (CNN forecasting)."""
    global _DEEPANT_MODEL
    if _DEEPANT_MODEL is not None:
        return _DEEPANT_MODEL
    if not os.path.exists(model_path):
        print(f"[DeepAnT] Không tìm thấy model tại {model_path}")
        return None
    try:
        from tensorflow import keras
        _DEEPANT_MODEL = keras.models.load_model(model_path)
        print(f"[DeepAnT] Đã load model từ {model_path}")
    except Exception as e:
        print(f"[DeepAnT] Lỗi load model: {e}")
        _DEEPANT_MODEL = None
    return _DEEPANT_MODEL

def _create_deepant_sequences(data, time_steps):
    """Tạo (X,y) giống logic inference DeepAnT: X length time_steps-1, y là phần tử kế tiếp."""
    X, y = [], []
    n = len(data)
    if n < time_steps:
        return np.array([]), np.array([])
    for i in range(n - time_steps + 1):
        X.append(data[i:i + time_steps - 1])
        y.append(data[i + time_steps - 1])
    return np.array(X), np.array(y)

def run_deepant_scores(X, time_steps=30, scaler=None, smooth=False, debug=False):
    """Trả về anomaly scores dựa trên lỗi dự báo của mô hình DeepAnT.

    Giống RePAD: chuẩn hoá đầu vào MinMax, tạo cửa sổ length time_steps-1 dự đoán điểm kế tiếp.
    scores đầu (time_steps-1) = 0. Trả về mảng length len(series) đã chuẩn hoá 0-1.
    """
    series = _to_numpy(X)
    if len(series) == 0:
        return np.array([])
    if len(series) < time_steps:
        return np.zeros(len(series))

    # Scale
    try:
        if scaler is None:
            scaler = MinMaxScaler()
            series_scaled = scaler.fit_transform(series.reshape(-1, 1)).astype(np.float32)
        else:
            series_scaled = scaler.transform(series.reshape(-1, 1)).astype(np.float32)
    except Exception as e:
        print(f"[DeepAnT] Lỗi scaler: {e} -> dùng min-max thủ công")
        s_min, s_max = np.min(series), np.max(series)
        rng = s_max - s_min if (s_max - s_min) > 0 else 1.0
        series_scaled = ((series - s_min) / rng).reshape(-1, 1).astype(np.float32)

    X_seq, y_true = _create_deepant_sequences(series_scaled, time_steps)
    if X_seq.size == 0:
        return np.zeros(len(series))

    model = _load_deepant_model()
    if model is None:
        if debug:
            print("[DeepAnT] Model load failed -> zeros")
        return np.zeros(len(series))

    try:
        y_pred = model.predict(X_seq, verbose=0)
    except Exception as e:
        print(f"[DeepAnT] Lỗi dự báo: {e}")
        return np.zeros(len(series))

    try:
        prediction_error = np.abs(y_pred - y_true).flatten()
    except Exception:
        prediction_error = np.linalg.norm(y_pred - y_true, axis=-1)

    scores = np.zeros(len(series))
    usable_len = min(len(prediction_error), len(series) - (time_steps - 1))
    scores[time_steps - 1: time_steps - 1 + usable_len] = prediction_error[:usable_len]

    if smooth and usable_len > 3:
        try:
            import pandas as pd
            scores = pd.Series(scores).rolling(window=3, min_periods=1, center=True).mean().to_numpy()
        except Exception:
            pass

    max_score = np.max(scores)
    if max_score > 0:
        scores = scores / (max_score + 1e-12)

    if debug:
        nonzero = scores[scores > 0]
        if nonzero.size:
            print(f"[DeepAnT][debug] raw_error_stats: min={nonzero.min():.4g} max={nonzero.max():.4g} mean={nonzero.mean():.4g} std={nonzero.std():.4g} count={nonzero.size}")
        else:
            print("[DeepAnT][debug] all zero scores")

    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

# Prophet
def _prepare_prophet_df(X):
    """
    Chuẩn bị DataFrame với cột 'ds' và 'y' từ nhiều định dạng đầu vào.
    Hàm này giả định X có index là thời gian hoặc có thể được reset thành dãy số.
    """
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        # Nếu cột đầu tiên không phải là datetime, sử dụng index
        if not np.issubdtype(df.iloc[:, 0].dtype, np.datetime64):
            df = df.reset_index()
        # Đổi tên cột
        df = df.iloc[:, [0, 1]].rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'})

    elif isinstance(X, pd.Series):
        df = X.to_frame().reset_index()
        df.columns = ['ds', 'y']
    else: # Numpy array hoặc list
        series = _to_numpy(X)
        # Tạo chuỗi ngày giả định nếu không có thông tin thời gian
        dates = pd.to_datetime(pd.to_numeric(np.arange(len(series)), downcast='integer'), unit='D')
        df = pd.DataFrame({'ds': dates, 'y': series})

    # Đảm bảo 'ds' là kiểu datetime
    df['ds'] = pd.to_datetime(df['ds'])
    return df

# ==============================================================================
# HÀM MỚI: RUN_PROPHET_SCORES
# ==============================================================================
def run_prophet_scores(X, interval_width=0.97, seasonality_mode='multiplicative', log_transform=True):
    """
    Chạy mô hình Prophet để phát hiện bất thường và trả về điểm số.

    Điểm số được tính dựa trên khoảng cách tương đối của giá trị thực tế so với
    ngưỡng dự báo (khoảng tin cậy).

    Tham số:
        X (DataFrame, Series, ndarray): Chuỗi thời gian đầu vào.
        interval_width (float): Độ rộng khoảng tin cậy của Prophet (ví dụ: 0.97 là 97%).
        seasonality_mode (str): Chế độ mùa vụ ('additive' hoặc 'multiplicative').
        log_transform (bool): Nếu True, áp dụng log transform (log1p) trước khi huấn luyện
                              để ổn định phương sai, hữu ích cho chuỗi có trend mạnh.

    Trả về:
        np.ndarray: Một mảng điểm số bất thường, có cùng độ dài với chuỗi đầu vào.
                    Điểm càng cao, mức độ bất thường càng lớn.
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("Cảnh báo: Thư viện 'prophet' chưa được cài đặt. Prophet sẽ không khả dụng.")
        print("Vui lòng chạy: pip install prophet")
        return np.zeros(len(X))

    series = _to_numpy(X)
    if len(series) < 30: # Prophet cần đủ dữ liệu để xác định mùa vụ
        print("Cảnh báo: Dữ liệu quá ngắn cho Prophet (< 30 điểm), sử dụng fallback.")
        return statistical_fallback(series)

    try:
        # 1. Chuẩn bị dữ liệu
        df_prophet = _prepare_prophet_df(X)

        # 2. Huấn luyện mô hình với weekly seasonality được bật
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)  # Tắt INFO messages
        
        model = Prophet(
            interval_width=interval_width,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            weekly_seasonality=True,  # Bật weekly seasonality
            yearly_seasonality=False,  # Tắt yearly seasonality
            daily_seasonality=False   # Tắt daily seasonality
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        # Sử dụng log transform nếu được bật
        if log_transform:
            df_prophet['y_log'] = np.log1p(df_prophet['y'])
            model.fit(df_prophet[['ds', 'y_log']].rename(columns={'y_log': 'y'}))
        else:
            model.fit(df_prophet)

        # 3. Tạo dự báo
        future = df_prophet[['ds']].copy()
        forecast = model.predict(future)

        # Chuyển đổi ngược nếu đã dùng log
        if log_transform:
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                forecast[col] = np.expm1(forecast[col])

        # Kết hợp dữ liệu thực tế vào forecast
        # Dùng merge để đảm bảo thứ tự các hàng được giữ nguyên
        full_df = pd.merge(df_prophet, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

        # 4. Tính toán điểm bất thường
        # Điểm số = khoảng cách từ điểm thực tế đến ngưỡng / độ rộng của ngưỡng
        # Điều này giúp chuẩn hóa điểm số, làm cho nó có ý nghĩa tương đối.
        upper_bound = full_df['yhat_upper']
        lower_bound = full_df['yhat_lower']
        actual = full_df['y']

        # Tính khoảng cách từ điểm thực tế đến ngưỡng gần nhất
        # Nếu điểm nằm trong ngưỡng, khoảng cách là 0
        distance_from_interval = np.maximum(actual - upper_bound, lower_bound - actual)
        distance_from_interval[distance_from_interval < 0] = 0 # Đảm bảo không có giá trị âm

        # Tính độ rộng của khoảng tin cậy (ngưỡng)
        interval_width_value = upper_bound - lower_bound
        # Tránh chia cho 0
        interval_width_value[interval_width_value <= 0] = 1e-8

        # Điểm số cuối cùng là khoảng cách tương đối
        anomaly_scores = distance_from_interval / interval_width_value

        return np.nan_to_num(anomaly_scores.values, nan=0.0, posinf=1e6, neginf=0.0)

    except Exception as e:
        print(f"Cảnh báo: Prophet thất bại với lỗi '{e}', sử dụng fallback.")
        return statistical_fallback(series)

# --- Helper class để tạo các cửa sổ dữ liệu ---
class _MomentTimeSeriesDataset(Dataset):
    def __init__(self, series, window_size=512, step=1):
        # Handle both pandas Series and numpy arrays
        if hasattr(series, 'values'):
            self.series = series.values.astype(np.float32)
        else:
            self.series = np.asarray(series, dtype=np.float32)
        self.window_size = window_size
        self.step = step

    def __len__(self):
        return max(0, (len(self.series) - self.window_size) // self.step + 1)

    def __getitem__(self, idx):
        start_idx = idx * self.step
        end_idx = start_idx + self.window_size
        window = self.series[start_idx:end_idx]
        return torch.tensor(window).unsqueeze(0)

# --- Biến toàn cục để tải mô hình một lần duy nhất ---
MOMENT_MODEL = None
DEVICE = None

def _load_moment_model():
    """Hàm nội bộ để tải và khởi tạo mô hình MOMENT."""
    global MOMENT_MODEL, DEVICE
    if MOMENT_MODEL is None:
        print("Đang khởi tạo mô hình MOMENT-1-base (chỉ một lần)...")
        try:
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            MOMENT_MODEL = MOMENTPipeline.from_pretrained(
                "AutonLab/MOMENT-1-base",
                model_kwargs={"task_name": "reconstruction"},
            )
            MOMENT_MODEL.init()
            MOMENT_MODEL = MOMENT_MODEL.to(DEVICE).float()
            print(f"Mô hình MOMENT đã được tải lên thiết bị: {DEVICE}")
        except Exception as e:
            MOMENT_MODEL = "failed" # Đánh dấu là đã thử và thất bại
            print(f"Lỗi nghiêm trọng khi tải mô hình MOMENT: {e}")
            print("Vui lòng kiểm tra cài đặt thư viện torch và momentfm.")

# --- Hàm chính để tích hợp vào file main ---
def run_moment_scores(X, window_size: int = 512):
    """
    Chạy mô hình MOMENT trên time series để tính điểm bất thường theo pattern của notebook mẫu.
    Nếu có lỗi hoặc dữ liệu không đủ điều kiện sẽ trả về mảng zeros.

    Args:
        X (pd.DataFrame): DataFrame chứa time series data ở cột đầu tiên.
        window_size (int): Kích thước cửa sổ (mặc định 512 như notebook).

    Returns:
        np.ndarray: Mảng điểm bất thường.
    """
    # Tải mô hình nếu chưa được tải
    _load_moment_model()
    
    if MOMENT_MODEL == "failed" or MOMENT_MODEL is None:
        print("[WARNING] MOMENT model không thể khởi tạo! Bỏ qua model này.")
        return np.zeros(len(X))

    # Lấy time series từ cột đầu tiên và reshape theo format của notebook
    time_series = X.iloc[:, 0].to_numpy()
    original_length = len(time_series)
    
    # Nếu chuỗi ngắn hơn window_size, pad với zeros
    if len(time_series) < window_size:
        print(f"[INFO] Chuỗi thời gian ngắn ({len(time_series)} < {window_size}), padding với zeros.")
        # Pad với zeros để đủ window_size
        time_series = np.pad(time_series, (0, window_size - len(time_series)), mode='constant', constant_values=0)

    try:
        # Chuẩn bị data theo format của notebook: [batch_size, n_channels, context_length]
        # Chia chuỗi thành các windows có độ dài 512
        n_windows = len(time_series) // window_size
        if n_windows == 0:
            print(f"[WARNING] Không thể tạo windows từ chuỗi có độ dài {len(time_series)}. Bỏ qua model này.")
            return np.zeros(len(time_series))
        
        # Pad chuỗi để chia hết cho window_size
        padded_length = n_windows * window_size
        time_series_padded = time_series[:padded_length]
        
        # Reshape thành [n_windows, 1, window_size] theo format notebook
        windowed_data = time_series_padded.reshape(n_windows, 1, window_size)
        windowed_tensor = torch.tensor(windowed_data, dtype=torch.float32).to(DEVICE)
        
        # Chạy model theo pattern notebook
        trues, preds = [], []
        with torch.no_grad():
            # Process theo batch để tránh memory issues
            batch_size = 8
            for i in range(0, len(windowed_tensor), batch_size):
                batch_x = windowed_tensor[i:i+batch_size]
                
                # Forward pass theo notebook
                output = MOMENT_MODEL(x_enc=batch_x)
                
                # Thu thập kết quả
                trues.append(batch_x.detach().squeeze().cpu().numpy())
                preds.append(output.reconstruction.detach().squeeze().cpu().numpy())
        
        # Concatenate results theo notebook
        trues = np.concatenate(trues, axis=0).flatten()
        preds = np.concatenate(preds, axis=0).flatten()
        
        # Tính anomaly scores theo notebook: MSE between observed and predicted
        anomaly_scores_windowed = (trues - preds) ** 2
        
        # Mở rộng scores về độ dài gốc
        anomaly_scores = np.zeros(len(time_series))
        anomaly_scores[:len(anomaly_scores_windowed)] = anomaly_scores_windowed
        
        # Nếu chuỗi dài hơn padded_length, xử lý phần còn lại
        if len(time_series) > padded_length:
            remaining_data = time_series[padded_length:]
            # Pad remaining để đủ 512
            if len(remaining_data) < window_size:
                remaining_padded = np.pad(remaining_data, (0, window_size - len(remaining_data)), mode='constant', constant_values=0)
                remaining_tensor = torch.tensor(remaining_padded.reshape(1, 1, window_size), dtype=torch.float32).to(DEVICE)
                
                with torch.no_grad():
                    output = MOMENT_MODEL(x_enc=remaining_tensor)
                    true_part = remaining_tensor.detach().squeeze().cpu().numpy()[:len(remaining_data)]
                    pred_part = output.reconstruction.detach().squeeze().cpu().numpy()[:len(remaining_data)]
                    
                    remaining_scores = (true_part - pred_part) ** 2
                    anomaly_scores[padded_length:] = remaining_scores
        
        # Trả về chỉ phần scores tương ứng với độ dài gốc
        return anomaly_scores[:original_length]
    
    except Exception as e:
        print(f"[WARNING] MOMENT model gặp lỗi: {e}")
        print("Bỏ qua MOMENT model do lỗi trong quá trình chạy.")
        return np.zeros(original_length)