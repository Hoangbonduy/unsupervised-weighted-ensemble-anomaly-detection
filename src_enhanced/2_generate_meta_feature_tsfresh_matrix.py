# generate_meta_feature_matrix.py

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
import warnings
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

# --- PHẦN 0: IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
try:
    # Thư viện chính để trích xuất 22 đặc trưng
    from pycatch22 import catch22_all
except ImportError:
    print("LỖI: Thư viện 'pycatch22' chưa được cài đặt.")
    print("Vui lòng chạy lệnh: pip install pycatch22")
    exit()

warnings.filterwarnings('ignore')
tqdm.pandas()

# --- PHẦN 1: HÀM TẢI DỮ LIỆU ---

def load_data(file_path='your_data.csv'):
    """
    Tải dữ liệu chuỗi thời gian từ một file CSV.
    CSV phải có các cột: 'placeId', 'date', 'view'.
    (Đây là hàm tương tự như ở bước trước)
    """
    print(f"Đang tải dữ liệu từ file: '{file_path}'...")

    if not os.path.exists(file_path):
        print(f"LỖI: Không tìm thấy file tại '{file_path}'.")
        return None

    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
    except Exception as e:
        print(f"LỖI: Không thể đọc file CSV. Lỗi: {e}")
        return None

    required_columns = ['placeId', 'date', 'view']
    if not all(col in df.columns for col in required_columns):
        print(f"LỖI: File CSV phải chứa đủ các cột: {required_columns}")
        return None

    df = df.sort_values(by=['placeId', 'date'])
    all_time_series = {}
    for place_id, group in df.groupby('placeId'):
        time_series = group.set_index('date')['view']
        all_time_series[place_id] = time_series

    print(f"Hoàn thành: Tải và xử lý được {len(all_time_series)} chuỗi thời gian.")
    return all_time_series

# --- PHẦN 2: HÀM TRÍCH XUẤT ĐẶC TRƯNG ---

def extract_combined_features(series_values, series_id):
    """
    Trích xuất đặc trưng từ cả Catch22 và tsfresh.
    """
    # 1. Trích xuất từ Catch22 (như cũ)
    catch22_features = catch22_all(series_values)
    feature_row = {name: val for name, val in zip(catch22_features['names'], catch22_features['values'])}
    
    # 2. Chuẩn bị dữ liệu cho tsfresh
    df_tsfresh = pd.DataFrame({
        'id': series_id,
        'time': range(len(series_values)),
        'value': series_values
    })
    
    # 3. Trích xuất từ tsfresh
    # Sử dụng một bộ cài đặt đơn giản để không quá chậm
    settings = ComprehensiveFCParameters() 
    tsfresh_features = extract_features(df_tsfresh, column_id='id', column_sort='time',
                                        default_fc_parameters=settings,
                                        disable_progressbar=True)
    
    # 4. Kết hợp hai bộ đặc trưng lại
    # Chuyển đổi tên cột của tsfresh để dễ đọc hơn
    tsfresh_dict = {f"tsf_{key}": val for key, val in tsfresh_features.iloc[0].to_dict().items()}
    feature_row.update(tsfresh_dict)
    
    return feature_row

# --- PHẦN 3: HÀM MAIN ---
def main():
    """Hàm chính điều phối toàn bộ quá trình."""
    print("Bắt đầu quy trình: Tạo Ma trận Meta-Feature (M)")

    # 1. Tải dữ liệu gốc của các địa điểm
    all_time_series = load_data(file_path='data/cleaned_data_after_idx30.csv')

    if not all_time_series:
        print("Không có dữ liệu để xử lý. Chương trình kết thúc.")
        return

    # Giới hạn chỉ xử lý 30 địa điểm đầu tiên để test
    # all_time_series_items = list(all_time_series.items())[:30]
    # print(f"Giới hạn xử lý: {len(all_time_series_items)} địa điểm đầu tiên (từ tổng {len(all_time_series)} địa điểm)")

    all_time_series_items = list(all_time_series.items())

    meta_features_data = []
    feature_names = [] # Sẽ được lấy từ lần chạy đầu tiên

    # 2. Lặp qua từng địa điểm để trích xuất đặc trưng
    print(f"Đang xử lý {len(all_time_series_items)} địa điểm để trích xuất đặc trưng kết hợp...")
    for place_id, series_data in tqdm(all_time_series_items, desc="Đang tạo ma trận M"):
        if series_data.empty:
            continue
        
        # Gọi hàm mới để trích xuất đặc trưng kết hợp
        feature_row = extract_combined_features(series_data['view'].values, place_id)
        feature_row['place_id'] = place_id
        meta_features_data.append(feature_row)

    # 3. Tạo DataFrame từ kết quả
    meta_feature_matrix_M = pd.DataFrame(meta_features_data)

    # Sắp xếp lại các cột để 'place_id' lên đầu
    cols = ['place_id'] + feature_names
    meta_feature_matrix_M = meta_feature_matrix_M[cols]

    # 4. Hiển thị và lưu kết quả
    print("\n--- HOÀN THÀNH ---")
    print("Ma trận Meta-Feature (M) đã được tạo thành công:")
    print(meta_feature_matrix_M.head())

    output_filename = 'results_enhanced_2/meta_feature_matrix_M.csv'
    meta_feature_matrix_M.to_csv(output_filename, index=False)
    print(f"\nMa trận đã được lưu vào file: '{output_filename}'")

if __name__ == '__main__':
    main()