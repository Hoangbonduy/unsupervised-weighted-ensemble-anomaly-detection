# generate_performance_matrix.py

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
import random

# --- PHẦN 0: IMPORT CÁC MODULE CẦN THIẾT ---
# Đảm bảo các file .py này nằm cùng thư mục hoặc trong sys.path

import sys
import os
# Thêm thư mục src vào sys.path để import các module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    # Import các hàm chạy mô hình cơ sở từ thư mục src
    import anomaly_detection_base_model as ad_models
    # Import lớp để tiêm bất thường từ thư mục src  
    from anomaly_injection_2 import AnomalyInjector
    # Import hàm tính VUS-ROC từ file hiện tại
    from model_evaluation import vus_roc_calculator
except ImportError as e:
    print(f"LỖI: Không thể import các module cần thiết: {e}")
    print("Vui lòng đảm bảo các file 'anomaly_detection_base_model.py', 'anomaly_injection_2.py' nằm trong thư mục 'src'.")
    print("Và file 'model_evaluation.py' nằm trong thư mục hiện tại.")
    exit()
    exit()

warnings.filterwarnings('ignore')
tqdm.pandas()

# --- PHẦN 1: CẤU HÌNH ---

# 1.1. Danh sách các mô hình cơ sở để đánh giá
BASE_MODELS = {
    'KNN': ad_models.run_knn_scores,
    'MA': ad_models.run_moving_average_scores,
    'IForest': ad_models.run_iforest_scores, 
    'RePAD': ad_models.run_repad_scores,
    'Moment': ad_models.run_moment_scores,
    'Prophet': ad_models.run_prophet_scores,
}


# 1.2. Các kịch bản tiêm bất thường để kiểm tra mô hình
# Việc sử dụng nhiều kịch bản giúp đánh giá hiệu suất tổng thể của mô hình tốt hơn
ANOMALY_PROFILES = [
    {"type": "seasonal", "params": {"period": 7, "n_anomalies": 4, "magnitude_std": 2.5}},
    {"type": "global", "params": {"n_anomalies": 5, "magnitude_std": 3.0}},
    {"type": "trend", "params": {"min_len": 20, "max_len": 50, "magnitude": 2.0}},
    {"type": "cutoff", "params": {"min_len": 15, "max_len": 30}},
    {"type": "average", "params": {"min_len": 20, "max_len": 40, "smoothing_window": 5}},
]

# 1.3. Khởi tạo đối tượng tiêm bất thường
injector = AnomalyInjector(random_state=42)

# --- PHẦN 2: CÁC HÀM CHỨC NĂNG ---

import pandas as pd
import os

def load_data(file_path='your_data.csv'):
    """
    Tải dữ liệu chuỗi thời gian từ một file CSV.

    CSV phải có các cột: 'placeId', 'date', 'view'.
    Hàm sẽ nhóm dữ liệu theo 'placeId' và trả về một dictionary, trong đó:
    - Key: ID của địa điểm (placeId)
    - Value: một pandas Series chứa dữ liệu 'view', đã được sắp xếp theo 'date'.

    Args:
        file_path (str): Đường dẫn đến file CSV chứa dữ liệu.

    Returns:
        dict: Một dictionary chứa các chuỗi thời gian.
    """
    print(f"Đang tải dữ liệu từ file: '{file_path}'...")

    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(file_path):
        print(f"LỖI: Không tìm thấy file tại '{file_path}'.")
        print("Tạo file CSV giả lập 'dummy_data.csv' để bạn chạy thử...")
        
        # Tạo dữ liệu giả lập nếu file không tồn tại
        num_places = 10
        rows_per_place = 200
        data = []
        for i in range(1, num_places + 1):
            place_id = f'place_{i}'
            dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=rows_per_place, freq='D'))
            views = np.random.randint(50, 500, size=rows_per_place) + np.sin(np.arange(rows_per_place) / 24) * 50
            for date, view in zip(dates, views):
                data.append([place_id, date, view])
        
        dummy_df = pd.DataFrame(data, columns=['placeId', 'date', 'view'])
        dummy_df.to_csv('dummy_data.csv', index=False)
        print("Đã tạo file 'dummy_data.csv'. Vui lòng đổi tên file hoặc sửa đường dẫn trong code.")
        file_path = 'dummy_data.csv'

    # Đọc file CSV
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
    except Exception as e:
        print(f"LỖI: Không thể đọc file CSV. Vui lòng kiểm tra định dạng file. Lỗi: {e}")
        return {}

    # Kiểm tra các cột cần thiết
    required_columns = ['placeId', 'date', 'view']
    if not all(col in df.columns for col in required_columns):
        print(f"LỖI: File CSV phải chứa đủ các cột: {required_columns}")
        return {}
    
    # Sắp xếp dữ liệu để đảm bảo chuỗi thời gian đúng thứ tự
    df = df.sort_values(by=['placeId', 'date'])

    # Nhóm theo 'placeId' và tạo dictionary chứa các chuỗi thời gian
    all_time_series = {}
    for place_id, group in df.groupby('placeId'):
        # Đặt 'date' làm index và chỉ lấy cột 'view'
        time_series = group.set_index('date')['view']
        all_time_series[place_id] = time_series

    print(f"Hoàn thành: Tải và xử lý được {len(all_time_series)} chuỗi thời gian.")
    return all_time_series

def generate_performance_row(series_data, injector):
    """
    Tạo một dòng trong ma trận hiệu suất P cho một chuỗi thời gian duy nhất
    bằng cách chọn ngẫu nhiên một trong nhiều kịch bản tiêm bất thường.

    Mỗi kịch bản được thiết kế để "thử thách" các điểm mạnh của các họ mô hình khác nhau.

    Args:
        series_data (pd.Series): Dữ liệu chuỗi thời gian của một địa điểm.
        injector (AnomalyInjector): Đối tượng injector đã được khởi tạo.

    Returns:
        dict: Một dictionary chứa điểm hiệu suất của các mô hình cơ sở.
    """
    
    # Tính độ dài series để điều chỉnh tham số phù hợp
    series_length = len(series_data)
    
    # --- ĐỊNH NGHĨA CÁC KỊCH BẢN TIÊM BẤT THƯỜNG (CẢI TIẾN THAM SỐ) ---

    # Kịch bản 1: "Bão Spike" - Thử thách các mô hình phát hiện thay đổi đột ngột (KNN, IForest)
    scenario_spike_storm = {
        "num_anomalies_range": (5, 12),
        "anomaly_type_dist": {'global': 0.6, 'scale': 0.3, 'cutoff': 0.1}
    }

    # Kịch bản 2: "Lệch nhịp Mùa vụ" - Thử thách các mô hình hiểu chu kỳ (RePAD, Prophet)
    scenario_seasonal_focus = {
        "num_anomalies_range": (2, 5),
        "anomaly_type_dist": {'seasonal': 0.8, 'global': 0.2}
    }
    
    # Kịch bản 3: "Biến dạng Hình thái" - Thử thách các mô hình dựa trên hình dạng (KNN)
    scenario_shape_focus = {
        "num_anomalies_range": (3, 6),
        "anomaly_type_dist": {'shape_distortion': 0.8, 'scale': 0.2}
    }

    # Kịch bản 4: "Thay đổi Thầm lặng" - Thử thách các mô hình dựa trên trung bình trượt (MA, Moment)
    # Điều chỉnh độ dài dựa trên series length để phù hợp
    scenario_gradual_change = {
        "num_anomalies_range": (2, 4),
        "anomaly_type_dist": {'gradual_level_shift': 0.7, 'trend': 0.3}
    }

    # Chỉ sử dụng 4 kịch bản chuyên biệt để meta-model học được
    # điểm mạnh riêng biệt của từng loại base model
    all_scenarios = [
        scenario_spike_storm,      # Ưu tiên: KNN, IForest
        scenario_seasonal_focus,   # Ưu tiên: RePAD, Prophet  
        scenario_shape_focus,      # Ưu tiên: KNN
        scenario_gradual_change    # Ưu tiên: MA, Moment
    ]

    # --- CHỌN NGẪU NHIÊN VÀ ÁP DỤNG KỊCH BẢN ---
    
    # Chọn ngẫu nhiên một kịch bản cho chuỗi dữ liệu hiện tại
    chosen_scenario = random.choice(all_scenarios)
    
    # Tạo dữ liệu giả lập dựa trên kịch bản đã chọn
    injected_series, labels = injector.inject(
        series_data.values,
        num_anomalies_range=chosen_scenario["num_anomalies_range"],
        anomaly_type_dist=chosen_scenario["anomaly_type_dist"]
    )
    
    # --- ĐÁNH GIÁ HIỆU SUẤT (Không thay đổi) ---
    
    performance_row = {}
    for model_name, model_func in BASE_MODELS.items():
        # Chuyển đổi numpy array thành DataFrame với cột 'value' như mong đợi
        injected_df = pd.DataFrame({'value': injected_series})
        scores = model_func(injected_df)
        vus_roc = vus_roc_calculator(labels, scores)
        performance_row[model_name] = vus_roc
        
    return performance_row

# --- PHẦN 3: HÀM THỰC THI CHÍNH ---

def main():
    """Hàm chính điều phối toàn bộ quá trình."""
    print("Bắt đầu quy trình: Tạo Ma trận Hiệu suất (P)")

    # 1. Tải dữ liệu của 285 địa điểm
    # !!! Nhớ thay thế hàm này bằng logic tải dữ liệu thật của bạn
    all_time_series = load_data(file_path='data/cleaned_data_after_idx30.csv')

    # Giới hạn chỉ xử lý 30 địa điểm đầu tiên để test
    # all_time_series_items = list(all_time_series.items())[:30]
    all_time_series_items = list(all_time_series.items())
    print(f"Giới hạn xử lý: {len(all_time_series_items)} địa điểm đầu tiên (từ tổng {len(all_time_series)} địa điểm)")

    performance_data = []

    # 2. Lặp qua từng địa điểm để xử lý
    print(f"Đang xử lý {len(all_time_series_items)} địa điểm...")
    for place_id, series_data in tqdm(all_time_series_items, desc="Đang tạo ma trận P"):
        # Tính toán hiệu suất các mô hình cho địa điểm hiện tại
        performance_row = generate_performance_row(series_data, injector)
        performance_row['place_id'] = place_id  # Thêm ID để định danh
        performance_data.append(performance_row)

    # 3. Tạo DataFrame từ kết quả
    performance_matrix_P = pd.DataFrame(performance_data)
    
    # Sắp xếp lại các cột để 'place_id' lên đầu
    cols = ['place_id'] + [col for col in performance_matrix_P if col != 'place_id']
    performance_matrix_P = performance_matrix_P[cols]

    # 4. Hiển thị và lưu kết quả
    print("\n--- HOÀN THÀNH ---")
    print("Ma trận Hiệu suất (P) đã được tạo thành công:")
    print(performance_matrix_P.head())

    output_filename = 'results_enhanced_final/performance_matrix_P_2.csv'
    performance_matrix_P.to_csv(output_filename, index=False)
    print(f"\nMa trận đã được lưu vào file: '{output_filename}'")

if __name__ == '__main__':
    main()