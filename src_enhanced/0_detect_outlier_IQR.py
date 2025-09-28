import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_and_plot_iqr_outliers(file_path, output_folder='iqr_outlier_plots'):
    """
    Phát hiện các điểm ngoại lai trong dữ liệu chuỗi thời gian bằng phương pháp IQR
    và vẽ biểu đồ cho từng địa điểm.

    Args:
        file_path (str): Đường dẫn đến file CSV.
        output_folder (str): Tên thư mục để lưu các biểu đồ.
    """
    # --- 1. Kiểm tra và tải dữ liệu ---
    if not os.path.exists(file_path):
        print(f"LỖI: Không tìm thấy file tại '{file_path}'.")
        print("Tạo file CSV giả lập 'dummy_data.csv' để bạn chạy thử...")
        
        # Tạo dữ liệu giả lập nếu file không tồn tại
        num_places = 5
        rows_per_place = 200
        data = []
        for i in range(1, num_places + 1):
            place_id = f'place_{i}'
            dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=rows_per_place, freq='D'))
            # Tạo dữ liệu bình thường
            views = np.random.randint(50, 150, size=rows_per_place) + np.sin(np.arange(rows_per_place) / 24) * 50
            # Thêm một vài điểm ngoại lai
            for _ in range(5):
                idx = np.random.randint(0, rows_per_place)
                views[idx] = views[idx] * np.random.choice([0.1, 2.5, 3.0]) # Cả thấp và cao
            
            for date, view in zip(dates, views):
                data.append([place_id, date, view])
        
        dummy_df = pd.DataFrame(data, columns=['placeId', 'date', 'view'])
        dummy_df.to_csv('dummy_data.csv', index=False)
        print("Đã tạo file 'dummy_data.csv'.")
        file_path = 'dummy_data.csv'

    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
    except Exception as e:
        print(f"LỖI: Không thể đọc file CSV. Vui lòng kiểm tra định dạng. Lỗi: {e}")
        return

    # Sắp xếp để đảm bảo thứ tự thời gian
    df = df.sort_values(by=['placeId', 'date'])

    # --- 2. Tạo thư mục output nếu chưa có ---
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Đã tạo thư mục: '{output_folder}'")

    # --- 3. Lặp qua từng địa điểm ---
    place_ids = df['placeId'].unique()[:10]  # Chỉ lấy 10 địa điểm đầu tiên
    print(f"\nBắt đầu xử lý {len(place_ids)} địa điểm đầu tiên...")

    for place_id in place_ids:
        # Lọc dữ liệu cho địa điểm hiện tại
        place_df = df[df['placeId'] == place_id].copy()
        
        if place_df.empty:
            continue

        # --- 4. Áp dụng thuật toán IQR ---
        Q1 = place_df['view'].quantile(0.25)
        Q3 = place_df['view'].quantile(0.75)
        IQR = Q3 - Q1

        # Xác định "hàng rào" trên và dưới
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Tìm các điểm ngoại lai
        outliers = place_df[(place_df['view'] < lower_bound) | (place_df['view'] > upper_bound)]

        # --- 5. Vẽ và lưu biểu đồ ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 6))

        # Vẽ chuỗi thời gian gốc
        ax.plot(place_df['date'], place_df['view'], label='Chuỗi thời gian gốc', color='royalblue', zorder=1)

        # Đánh dấu các điểm ngoại lai
        ax.scatter(outliers['date'], outliers['view'], color='red', s=60, label='Điểm ngoại lai (IQR)', zorder=2)

        # Định dạng biểu đồ
        ax.set_title(f"Phát hiện ngoại lai cho PlaceID: {place_id} bằng IQR", fontsize=14)
        ax.set_xlabel("Ngày", fontsize=12)
        ax.set_ylabel("Lượt xem (view)", fontsize=12)
        ax.legend()
        plt.tight_layout()

        # Lưu biểu đồ
        output_path = os.path.join(output_folder, f"outliers_{place_id}.png")
        plt.savefig(output_path)
        plt.close(fig) # Đóng biểu đồ để giải phóng bộ nhớ

    print(f"\nHoàn thành! Tất cả các biểu đồ đã được lưu vào thư mục '{output_folder}'.")

if __name__ == '__main__':
    # !!! THAY ĐỔI TÊN FILE CỦA BẠN Ở ĐÂY !!!
    csv_file_name = 'data/cleaned_data_no_zero_periods_filtered.csv'  
    detect_and_plot_iqr_outliers(csv_file_name)