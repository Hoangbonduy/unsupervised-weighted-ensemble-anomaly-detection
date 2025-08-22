import pandas as pd
from scipy.stats import spearmanr
import numpy as np
import anomaly_injection

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, pd.Series):
        return x.values
    else:
        return np.array(x)

# Cần thêm import mới
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

# Giả sử hàm _to_numpy và find_dominant_period đã có

def get_prediction_error_ranking(time_series, models_to_run_scores):
    """
    Xếp hạng các mô hình dựa trên sự tương quan giữa điểm bất thường của chúng
    và thành phần "phần dư" (residual) từ phân rã chuỗi thời gian.
    """
    ts_series = pd.Series(_to_numpy(time_series))

    # 1. TẠO CHUỖI "LỖI" THÔNG MINH HƠN BẰNG PHÂN RÃ
    
    # Tìm chu kỳ để phân rã cho đúng
    period = anomaly_injection.find_dominant_period(ts_series)
    
    # Yêu cầu tối thiểu để phân rã là 2 chu kỳ đầy đủ
    if len(ts_series) < period * 2 or period == 0:
        print("    Warning (Pred. Error): Chuỗi quá ngắn hoặc không có chu kỳ, dùng fallback Moving Average.")
        # Fallback về phương pháp cũ nếu không phân rã được
        moving_avg = ts_series.rolling(window=7, center=True).mean().bfill().ffill()
        error_series = np.abs(ts_series - moving_avg)
    else:
        # Thực hiện phân rã
        decomposition = seasonal_decompose(
            ts_series, 
            model='additive', # Giả định mô hình cộng (có thể đổi thành 'multiplicative')
            period=period,
            extrapolate_trend='freq' # Giúp xử lý các giá trị NaN ở biên
        )
        
        # Lấy phần dư (residual) làm chuỗi "lỗi"
        # Phần dư chính là những gì không thể giải thích bởi trend và seasonality
        error_series = np.abs(decomposition.resid.fillna(0).values)

    # 2. Lấy điểm số từ tất cả các mô hình (không đổi)
    model_scores = {}
    ts_df = pd.DataFrame(time_series)
    for name, run_func in models_to_run_scores.items():
        model_scores[name] = run_func(ts_df)

    # 3. Tính độ tương quan và xếp hạng (không đổi)
    correlations = {}
    for name, scores in model_scores.items():
        # Xử lý các giá trị vô hạn hoặc NaN trong scores trước khi tính corr
        scores_cleaned = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        if len(scores_cleaned) == len(error_series) and np.std(scores_cleaned) > 1e-8:
            corr, _ = spearmanr(scores_cleaned, error_series)
            correlations[name] = np.nan_to_num(corr)
        else:
            correlations[name] = 0.0

    ranked_models = sorted(correlations, key=correlations.get, reverse=True)
    
    return ranked_models, correlations