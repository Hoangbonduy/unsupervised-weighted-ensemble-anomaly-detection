import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# Giả sử file anomaly_injection.py đã tồn tại và chứa các hàm inject_*
import anomaly_injection_2 as anom

# ==============================================================================
# PHẦN 0: TẢI DỮ LIỆU (Sử dụng dữ liệu mẫu nếu file của bạn không có sẵn)
# ==============================================================================
try:
    df = pd.read_csv('data/cleaned_data_no_zero_periods_filtered.csv')
    df['date'] = pd.to_datetime(df['date'])
except FileNotFoundError:
    print("Không tìm thấy file dữ liệu, đang tạo dữ liệu mẫu...")
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=365))
    views = 1000 + 500 * np.sin(np.arange(365) * 2 * np.pi / 30) + np.random.normal(0, 50, 365)
    df = pd.DataFrame({'placeId': 4611864748268400448, 'date': dates, 'view': views})

# ==============================================================================
# PHẦN 1: LẤY DỮ LIỆU VÀ TẠO CÁC PHIÊN BẢN BẤT THƯỜNG
# ==============================================================================
first_place_id = df['placeId'].unique()[0]
time_series_df = df[df['placeId'] == first_place_id].copy()
original_ts = time_series_df['view'].values

injection_functions = {
    'spike': anom.inject_spike_anomaly,
    'contextual': anom.inject_contextual_anomaly,
    'flip': anom.inject_flip_anomaly,
    'speedup': anom.inject_speedup_anomaly,
    'noise': anom.inject_noise_anomaly,
    'cutoff': anom.inject_cutoff_anomaly,
    'scale': anom.inject_scale_anomaly,
    'wander': anom.inject_wander_anomaly,
    'average': anom.inject_average_anomaly
}

# Tạo tất cả các bộ dữ liệu bất thường
anomalous_datasets = {}
base_seed = 42
for i, (name, func) in enumerate(injection_functions.items()):
    ts_anomalous, labels = func(original_ts, seed=base_seed + i)
    anomalous_datasets[name] = (ts_anomalous, labels)

# ==============================================================================
# PHẦN 2: TRỰC QUAN HÓA - CẬP NHẬT LOGIC VẼ
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
output_dir = 'anomalies_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Hàm trợ giúp đã được cập nhật
def create_and_save_plot(dates, series, title, filename, labels=None, anomaly_type='segment'):
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # 1. Luôn vẽ chuỗi gốc màu xanh
    ax.plot(dates, series, color='dodgerblue', zorder=1, label='Dữ liệu')

    if labels is not None and np.any(labels):
        is_anomaly = labels.astype(bool)
        
        # 2. Logic vẽ bất thường
        if anomaly_type == 'spike':
            # Nếu là spike, dùng điểm tròn
            ax.scatter(dates[is_anomaly], series[is_anomaly], 
                       color='red', s=50, zorder=3, label='Điểm bất thường')
        else:
            # Nếu là đoạn, tô màu đỏ
            # Tạo một chuỗi mới chỉ chứa các điểm bất thường, còn lại là NaN
            segment_series = series.copy()
            segment_series[~is_anomaly] = np.nan
            ax.plot(dates, segment_series, color='red', zorder=2, label='Vùng bất thường')

    ax.set_title(f'{title} (placeId: {first_place_id})', fontsize=16)
    ax.set_xlabel('Ngày', fontsize=12)
    ax.set_ylabel('Lượt xem', fontsize=12)
    ax.legend()
    ax.tick_params(axis='x', rotation=25)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Đã lưu biểu đồ: {filepath}")
    plt.close(fig)

# --- Tạo từng biểu đồ với logic vẽ mới ---
dates = time_series_df['date']

# 1. Biểu đồ gốc
create_and_save_plot(dates, original_ts, 'Chuỗi Thời Gian Gốc', '0_original.png')

# 2. Các biểu đồ bất thường
filename_map = {
    'spike': '1_spike_anomaly.png',
    'contextual': '2_contextual_anomaly.png',
    'flip': '3_flip_anomaly.png',
    'speedup': '4_speedup_anomaly.png',
    'noise': '5_noise_anomaly.png',
    'cutoff': '6_cutoff_anomaly.png',
    'scale': '7_scale_anomaly.png',
    'wander': '8_wander_anomaly.png',
    'average': '9_average_anomaly.png'
}

for name, (ts_anomalous, labels) in anomalous_datasets.items():
    plot_type = 'spike' if name == 'spike' else 'segment'
    create_and_save_plot(dates, ts_anomalous, f'Bất Thường: {name.capitalize()}', 
                         filename_map[name], labels=labels, anomaly_type=plot_type)

print("\nHoàn tất việc tạo các ảnh mới!")