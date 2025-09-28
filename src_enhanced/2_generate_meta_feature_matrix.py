# generate_meta_feature_matrix.py

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
import warnings

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

# --- PHẦN 2: HÀM THỰC THI CHÍNH ---

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
    print(f"Đang xử lý {len(all_time_series_items)} địa điểm để trích xuất đặc trưng Catch22...")
    for place_id, series_data in tqdm(all_time_series_items, desc="Đang tạo ma trận M"):
        
        # Chỉ trích xuất đặc trưng nếu chuỗi không rỗng
        if series_data.empty:
            continue
            
        # Trích xuất 22 đặc trưng bằng catch22_all
        features = catch22_all(series_data.values)
        
        # Tạo một dictionary cho dòng dữ liệu này
        feature_row = {name: val for name, val in zip(features['names'], features['values'])}
        feature_row['place_id'] = place_id # Thêm ID để định danh
        meta_features_data.append(feature_row)
        
        # Lấy tên của 22 đặc trưng (chỉ cần làm một lần)
        if not feature_names:
            feature_names = features['names']

    # 3. Tạo DataFrame từ kết quả
    meta_feature_matrix_M = pd.DataFrame(meta_features_data)

    # Sắp xếp lại các cột để 'place_id' lên đầu
    cols = ['place_id'] + feature_names
    meta_feature_matrix_M = meta_feature_matrix_M[cols]

    # 4. Hiển thị và lưu kết quả
    print("\n--- HOÀN THÀNH ---")
    print("Ma trận Meta-Feature (M) đã được tạo thành công:")
    print(meta_feature_matrix_M.head())

    output_filename = 'results_enhanced_final/meta_feature_matrix_M.csv'
    meta_feature_matrix_M.to_csv(output_filename, index=False)
    print(f"\nMa trận đã được lưu vào file: '{output_filename}'")

if __name__ == '__main__':
    main()