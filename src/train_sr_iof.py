import os
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import joblib
import matplotlib.pyplot as plt
import sranodec as anom
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Tắt warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Đường dẫn
DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
MODELS_DIR = 'pretrained_weights'
os.makedirs(MODELS_DIR, exist_ok=True)

def inject_anomalies(series, anomaly_ratio=0.05):
    """Tiêm bất thường vào chuỗi thời gian để đánh giá khả năng phát hiện."""
    series = np.array(series, dtype=np.float64)
    labels = np.zeros(len(series))
    
    num_anomalies = max(1, int(len(series) * anomaly_ratio))
    anomaly_indices = np.random.choice(len(series), num_anomalies, replace=False)
    
    std_dev = np.std(series)
    mean_val = np.mean(series)
    
    for idx in anomaly_indices:
        # Tạo bất thường ngẫu nhiên: spike hoặc dip
        if np.random.random() > 0.5:
            series[idx] = mean_val + std_dev * np.random.uniform(3, 5)  # Spike
        else:
            series[idx] = mean_val - std_dev * np.random.uniform(3, 5)  # Dip
        labels[idx] = 1
    
    return series, labels

def train_sr_model(data_train):
    """
    Tìm tham số tối ưu cho SR dựa trên khả năng phát hiện bất thường.
    """
    print("\n" + "="*50)
    print("ĐANG TÌM THAM SỐ TỐI ƯU CHO MÔ HÌNH SR")
    print("="*50)

    # Các giá trị window size để thử - mở rộng phạm vi
    amp_window_sizes = [12, 24, 36, 48]
    series_window_sizes = [12, 24, 36, 48] 
    score_window_sizes = [50, 100, 150, 200]

    # Lấy mẫu 30 chuỗi thời gian để tìm tham số tối ưu
    sample_series = []
    for pid in tqdm(np.random.choice(list(data_train['placeId'].unique()), 30), 
                    desc="Đang lấy mẫu dữ liệu"):
        series = data_train[data_train['placeId'] == pid]['view'].values
        if len(series) > 200:  # Chỉ lấy các chuỗi đủ dài
            sample_series.append(series)
    
    print(f"Đã lấy {len(sample_series)} chuỗi mẫu để tối ưu tham số")
    
    best_auc = 0
    best_params = None
    
    # Thử các tổ hợp tham số
    for amp_ws in amp_window_sizes:
        for series_ws in series_window_sizes:
            for score_ws in score_window_sizes:
                auc_scores = []
                
                for series in tqdm(sample_series, leave=False, 
                                   desc=f"Thử SR với (a={amp_ws}, s={series_ws}, sc={score_ws})"):
                    try:
                        # Tiêm bất thường để kiểm tra khả năng phát hiện
                        series_with_anomalies, true_labels = inject_anomalies(series, 0.05)
                        
                        # Tạo mô hình SR với tham số hiện tại
                        spec = anom.Silency(amp_ws, series_ws, score_ws)
                        anomaly_scores = spec.generate_anomaly_score(series_with_anomalies)
                        
                        # Tính AUC để đánh giá khả năng phát hiện bất thường
                        if len(np.unique(true_labels)) == 2:  # Có cả normal và anomaly
                            auc = roc_auc_score(true_labels, anomaly_scores)
                            auc_scores.append(auc)
                        else:
                            auc_scores.append(0.5)  # Baseline AUC
                            
                    except Exception as e:
                        auc_scores.append(0.5)  # Baseline AUC cho lỗi
                
                avg_auc = np.mean(auc_scores)
                print(f"Tham số (a={amp_ws}, s={series_ws}, sc={score_ws}): AUC = {avg_auc:.4f}")
                
                if avg_auc > best_auc:
                    best_auc = avg_auc
                    best_params = (amp_ws, series_ws, score_ws)
    
    print(f"\nTham số tối ưu cho SR: amp_window_size={best_params[0]}, "
          f"series_window_size={best_params[1]}, score_window_size={best_params[2]}")
    print(f"AUC trung bình: {best_auc:.4f}")
    
    # Lưu tham số tối ưu
    sr_params = {
        'amp_window_size': best_params[0],
        'series_window_size': best_params[1],
        'score_window_size': best_params[2],
        'validation_auc': best_auc
    }
    
    joblib.dump(sr_params, os.path.join(MODELS_DIR, 'sr_params.joblib'))
    print(f"Đã lưu tham số SR tại: {os.path.join(MODELS_DIR, 'sr_params.joblib')}")
    
    return sr_params

def train_iforest_model(data_train):
    """
    Training IsolationForest với cách tiếp cận cải thiện.
    """
    print("\n" + "="*50)
    print("ĐANG TRAINING ISOLATION FOREST")
    print("="*50)

    # Chuẩn bị dữ liệu với cách tiếp cận tốt hơn
    all_series = []
    place_ids_sample = np.random.choice(data_train['placeId'].unique(), 200, replace=False)
    
    for pid in tqdm(place_ids_sample, desc="Đang xử lý PlaceIDs"):
        series = data_train[data_train['placeId'] == pid]['view'].values
        if len(series) > 50:  # Lọc các chuỗi quá ngắn
            # Lấy mẫu từ mỗi chuỗi thay vì toàn bộ
            sample_size = min(len(series), 500)  # Tối đa 500 điểm mỗi chuỗi
            sampled_indices = np.random.choice(len(series), sample_size, replace=False)
            sampled_series = series[sampled_indices]
            all_series.extend(sampled_series)
    
    # Chuyển thành numpy array
    all_series = np.array(all_series)
    
    # Lấy mẫu cuối cùng để tránh overfit
    if len(all_series) > 50000:
        sample_indices = np.random.choice(len(all_series), 50000, replace=False)
        all_series = all_series[sample_indices]
    
    print(f"Training IsolationForest trên {len(all_series)} điểm dữ liệu...")
    
    # Chuẩn hóa dữ liệu với StandardScaler (ít nhạy cảm với outliers)
    scaler = StandardScaler()
    X = scaler.fit_transform(all_series.reshape(-1, 1))
    
    # Train Isolation Forest với contamination cụ thể
    iforest = IsolationForest(
        n_estimators=100,  # Giảm số cây để tránh overfit
        max_samples=0.8,   # Sử dụng 80% mẫu cho mỗi cây
        contamination=0.05,  # Contamination cụ thể thay vì 'auto'
        random_state=42,
        bootstrap=False,   # Tắt bootstrap để tăng đa dạng
        n_jobs=-1
    )
    
    iforest.fit(X)
    
    # Lưu cả model và scaler
    joblib.dump(iforest, os.path.join(MODELS_DIR, 'iforest_pretrained.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'iforest_scaler.joblib'))
    
    print(f"Đã lưu IsolationForest tại: {os.path.join(MODELS_DIR, 'iforest_pretrained.joblib')}")
    print(f"Đã lưu IsolationForest Scaler tại: {os.path.join(MODELS_DIR, 'iforest_scaler.joblib')}")
    
    # Kiểm tra phân phối điểm số và tìm threshold tốt
    # Trong hàm train_iforest_model, thêm đoạn này sau khi tính scores:
    scores = -iforest.decision_function(X)  # Điểm cao = bất thường

    # Tính và lưu các threshold có ý nghĩa
    thresholds = {
        'threshold_90': np.percentile(scores, 90),
        'threshold_95': np.percentile(scores, 95),
        'threshold_97': np.percentile(scores, 97),
        'threshold_99': np.percentile(scores, 99)
    }

    joblib.dump(thresholds, os.path.join(MODELS_DIR, 'iforest_thresholds.joblib'))
    print(f"Đã lưu thresholds tại: {os.path.join(MODELS_DIR, 'iforest_thresholds.joblib')}")
    
    plt.figure(figsize=(12, 8))
    
    # Lấy các giá trị threshold từ dictionary
    threshold_95 = thresholds['threshold_95']
    threshold_97 = thresholds['threshold_97']
    threshold_99 = thresholds['threshold_99']

    # Subplot 1: Histogram phân phối
    plt.subplot(2, 2, 1)
    plt.hist(scores, bins=50, alpha=0.7, density=True)
    plt.axvline(x=threshold_95, color='r', linestyle='--', label=f'95% ({threshold_95:.3f})')
    plt.axvline(x=threshold_97, color='g', linestyle='--', label=f'97% ({threshold_97:.3f})')
    plt.axvline(x=threshold_99, color='b', linestyle='--', label=f'99% ({threshold_99:.3f})')
    plt.title('Phân phối điểm bất thường từ IsolationForest')
    plt.xlabel('Điểm bất thường')
    plt.ylabel('Mật độ')
    plt.legend()
    
    # Subplot 2: Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(scores)
    plt.title('Box plot của điểm bất thường')
    plt.ylabel('Điểm bất thường')
    
    # Subplot 3: Q-Q plot để kiểm tra phân phối
    from scipy import stats
    plt.subplot(2, 2, 3)
    stats.probplot(scores, dist="norm", plot=plt)
    plt.title('Q-Q Plot vs Normal Distribution')
    
    # Subplot 4: Thống kê các threshold
    plt.subplot(2, 2, 4)
    thresholds = [threshold_95, threshold_97, threshold_99]
    contamination_rates = [0.05, 0.03, 0.01]
    plt.bar(['95%', '97%', '99%'], thresholds, alpha=0.7)
    plt.title('Các threshold khuyến nghị')
    plt.ylabel('Giá trị threshold')
    
    # Thêm text hiển thị contamination rate
    for i, (thresh, cont_rate) in enumerate(zip(thresholds, contamination_rates)):
        plt.text(i, thresh + 0.01, f'{cont_rate*100}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'iforest_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Đã lưu phân tích IForest tại: {os.path.join(MODELS_DIR, 'iforest_analysis.png')}")
    
    # Lưu thống kê threshold để sử dụng sau
    threshold_stats = {
        'threshold_95': threshold_95,
        'threshold_97': threshold_97,
        'threshold_99': threshold_99,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'recommended_threshold': threshold_97  # Sử dụng 97% làm default
    }
    
    joblib.dump(threshold_stats, os.path.join(MODELS_DIR, 'iforest_thresholds.joblib'))
    print(f"Đã lưu thống kê threshold tại: {os.path.join(MODELS_DIR, 'iforest_thresholds.joblib')}")
    
    print(f"\nThống kê IsolationForest:")
    print(f"  - Threshold 95% (5% anomalies): {threshold_95:.4f}")
    print(f"  - Threshold 97% (3% anomalies): {threshold_97:.4f}")
    print(f"  - Threshold 99% (1% anomalies): {threshold_99:.4f}")
    print(f"  - Khuyến nghị sử dụng threshold: {threshold_97:.4f}")
    
    return iforest, scaler, threshold_stats

def main():
    # Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    try:
        data_full = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại '{DATA_FILE_PATH}'")
        return

    # Lấy dữ liệu từ index 30 trở đi (không có nhãn)
    unique_place_ids = data_full['placeId'].unique()
    unlabeled_place_ids = unique_place_ids[30:]
    data_train = data_full[data_full['placeId'].isin(unlabeled_place_ids)]
    
    print(f"Đã đọc {len(data_train)} dòng dữ liệu từ {len(unlabeled_place_ids)} PlaceIDs không có nhãn")
    
    # 1. Training SR (tìm tham số tối ưu dựa trên AUC)
    sr_params = train_sr_model(data_train)
    
    # 2. Training IsolationForest (cải thiện)
    iforest_model, scaler, threshold_stats = train_iforest_model(data_train)
    
    print("\n" + "="*50)
    print("QUÁ TRÌNH TRAINING HOÀN TẤT")
    print("="*50)
    print(f"Tham số SR tối ưu: {sr_params}")
    print(f"AUC validation: {sr_params['validation_auc']:.4f}")
    print(f"IsolationForest threshold khuyến nghị: {threshold_stats['recommended_threshold']:.4f}")
    print(f"Đã lưu tất cả model và metadata tại: {MODELS_DIR}")
    
    print("\nCác file đã được tạo:")
    print("  - sr_params.joblib: Tham số tối ưu cho SR")
    print("  - iforest_pretrained.joblib: Mô hình IsolationForest đã train")
    print("  - iforest_scaler.joblib: Scaler cho IsolationForest")
    print("  - iforest_thresholds.joblib: Thống kê threshold")
    print("  - iforest_analysis.png: Phân tích chi tiết IsolationForest")
    
    print("\nBạn có thể sử dụng các mô hình đã train như sau:")
    print("""
# Sử dụng SR với tham số tối ưu:
sr_params = joblib.load('pretrained_weights/sr_params.joblib')
spec = anom.Silency(**sr_params)
scores = spec.generate_anomaly_score(series)

# Sử dụng IsolationForest đã train:
iforest = joblib.load('pretrained_weights/iforest_pretrained.joblib')
scaler = joblib.load('pretrained_weights/iforest_scaler.joblib')
thresholds = joblib.load('pretrained_weights/iforest_thresholds.joblib')

# Scale dữ liệu và tính scores
X_scaled = scaler.transform(X)
scores = -iforest.decision_function(X_scaled)
predictions = scores > thresholds['recommended_threshold']
    """)

if __name__ == "__main__":
    main()