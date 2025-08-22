# -*- coding: utf-8 -*-
"""
Manual Labeling Tool for Time Series Anomaly Detection
Công cụ gán nhãn thủ công cho phát hiện bất thường trong chuỗi thời gian
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os

class ManualLabelingTool:
    def __init__(self, data_path=None, labels_dir=None):
        # --- THAY ĐỔI: Ưu tiên load file đã lọc ---
        base_data_path = "data/cleaned_data_no_zero_periods.csv"
        filtered_data_path = "data/cleaned_data_no_zero_periods_filtered.csv"

        if data_path is None:
            # Ưu tiên file đã lọc nếu nó tồn tại
            if os.path.exists(filtered_data_path):
                data_path = filtered_data_path
                print(f"Đã tìm thấy và sẽ sử dụng file dữ liệu đã được lọc: {data_path}")
            elif os.path.exists(f"../{filtered_data_path}"):
                data_path = f"../{filtered_data_path}"
                print(f"Đã tìm thấy và sẽ sử dụng file dữ liệu đã được lọc: {data_path}")
            # Nếu không, tìm file gốc
            elif os.path.exists(base_data_path):
                data_path = base_data_path
            elif os.path.exists(f"../{base_data_path}"):
                data_path = f"../{base_data_path}"
            else:
                raise FileNotFoundError("Không tìm thấy file 'cleaned_data_no_zero_periods.csv' hoặc phiên bản đã lọc của nó.")
        # --- KẾT THÚC THAY ĐỔI ---

        if labels_dir is None:
            if os.path.exists("labels"):
                labels_dir = "labels"
            else:
                labels_dir = "../labels"
        
        self.data_path = data_path
        self.labels_dir = labels_dir
        self.current_place_id = None
        self.current_data = None
        self.labels = None
        self.fig = None
        self.ax = None
        self.anomaly_points = []
        
        os.makedirs(labels_dir, exist_ok=True)
        
        self.df = pd.read_csv(data_path)
        self.place_ids = self.df['placeId'].unique()
        print(f"Đã load dữ liệu từ: {os.path.basename(self.data_path)}")
        print(f"Tìm thấy {len(self.place_ids)} place IDs trong dữ liệu.")

    # ... các hàm khác không thay đổi ...
    def list_places(self):
        """Hiển thị danh sách các place IDs"""
        print("\nDanh sách Place IDs:")
        for i, place_id in enumerate(self.place_ids):
            print(f"{i+1}: {place_id}")
        return self.place_ids

    # --- TÍNH NĂNG MỚI ---
    def filter_and_save_long_series(self, min_length=30):
        """
        Lọc và xóa các chuỗi thời gian có độ dài < min_length và lưu vào file mới.
        """
        print(f"\nBắt đầu lọc các placeId có độ dài nhỏ hơn {min_length}...")
        
        # Đếm số lượng điểm dữ liệu cho mỗi placeId
        counts = self.df.groupby('placeId').size()
        
        # Lấy danh sách các placeId cần giữ lại
        places_to_keep = counts[counts >= min_length].index
        places_to_remove = counts[counts < min_length].index
        
        if len(places_to_remove) == 0:
            print("Không có placeId nào cần xóa. Dữ liệu đã đạt yêu cầu.")
            return

        print(f"Tổng số placeId ban đầu: {len(self.place_ids)}")
        print(f"Số placeId sẽ bị xóa ({len(places_to_remove)}): {list(places_to_remove)}")
        print(f"Số placeId được giữ lại: {len(places_to_keep)}")

        # Tạo DataFrame mới chỉ chứa các placeId cần giữ lại
        filtered_df = self.df[self.df['placeId'].isin(places_to_keep)].copy()
        
        # Tạo tên file mới
        base, ext = os.path.splitext(self.data_path)
        # Tránh thêm "_filtered" nhiều lần
        if base.endswith('_filtered'):
            new_filepath = self.data_path
        else:
            new_filepath = f"{base}_filtered{ext}"
        
        # Lưu file mới
        try:
            filtered_df.to_csv(new_filepath, index=False)
            print(f"\nĐã lưu thành công dữ liệu đã lọc vào: {new_filepath}")
            
            # Cập nhật trạng thái hiện tại của công cụ để sử dụng dữ liệu mới
            print("Đang cập nhật lại công cụ để sử dụng dữ liệu mới...")
            self.data_path = new_filepath
            self.df = filtered_df
            self.place_ids = filtered_df['placeId'].unique()
            print("Cập nhật hoàn tất. Công cụ hiện đang sử dụng dữ liệu đã lọc.")
            
        except Exception as e:
            print(f"Đã xảy ra lỗi khi lưu file: {e}")

    # ... các hàm khác không thay đổi (load_place_data, interactive_labeling, v.v.) ...
    def load_place_data(self, place_id):
        """Load dữ liệu cho một place cụ thể"""
        self.current_place_id = place_id
        place_data = self.df[self.df['placeId'] == place_id].copy()
        place_data = place_data.sort_values('date')
        place_data['date'] = pd.to_datetime(place_data['date'])
        
        self.current_data = place_data
        self.labels = np.zeros(len(place_data))  # Khởi tạo tất cả là normal (0)
        self.anomaly_points = []
        
        print(f"\nĐã load dữ liệu cho Place ID: {place_id}")
        print(f"Số điểm dữ liệu: {len(place_data)}")
        print(f"Khoảng thời gian: {place_data['date'].min()} đến {place_data['date'].max()}")
        print(f"View range: {place_data['view'].min()} đến {place_data['view'].max()}")
        
        # Tự động load nhãn hiện có nếu có
        self.load_existing_labels()

    def interactive_labeling(self):
        """Giao diện tương tác để gán nhãn"""
        if self.current_data is None:
            print("Vui lòng load dữ liệu trước!")
            return
        
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        
        # Vẽ dữ liệu
        dates = self.current_data['date']
        views = self.current_data['view']
        
        # Vẽ line với markers để dễ click. Tăng markersize để dễ click hơn.
        self.line, = self.ax.plot(dates, views, 'b-', linewidth=1, marker='.', markersize=8, picker=5, label='Views')
        self.anomaly_scatter = self.ax.scatter([], [], c='red', s=100, marker='X', label='Anomalies', zorder=5)
        
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Views')
        self.ax.grid(True, alpha=0.3)
        
        # Xoay nhãn date để dễ đọc
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha="right")
        
        # Thêm các nút
        # Tăng khoảng cách giữa các nút
        button_width = 0.08
        button_height = 0.05
        start_x = 0.1
        spacing = 0.01

        ax_save = plt.axes([start_x, 0.01, button_width, button_height])
        ax_clear = plt.axes([start_x + button_width + spacing, 0.01, button_width, button_height])
        ax_auto = plt.axes([start_x + 2 * (button_width + spacing), 0.01, button_width, button_height])
        ax_stats = plt.axes([start_x + 3 * (button_width + spacing), 0.01, button_width, button_height])
        ax_load = plt.axes([start_x + 4 * (button_width + spacing), 0.01, button_width, button_height])
        ax_zoom = plt.axes([start_x + 5 * (button_width + spacing), 0.01, button_width, button_height])
        
        self.btn_save = Button(ax_save, 'Save')
        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_auto = Button(ax_auto, 'Auto')
        self.btn_stats = Button(ax_stats, 'Stats')
        self.btn_load = Button(ax_load, 'Load')
        self.btn_zoom = Button(ax_zoom, 'Reset Zoom')
        
        # Kết nối events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.btn_save.on_clicked(self.save_labels)
        self.btn_clear.on_clicked(self.clear_labels)
        self.btn_auto.on_clicked(self.auto_detect_anomalies)
        self.btn_stats.on_clicked(self.show_stats)
        self.btn_load.on_clicked(self.load_existing_labels)
        self.btn_zoom.on_clicked(self.reset_zoom)
        
        # Điều chỉnh layout để các nút không bị che
        self.fig.subplots_adjust(bottom=0.2)
        
        # Cập nhật hiển thị ban đầu
        self.update_plot()
        
        print("\nHướng dẫn sử dụng:")
        print("- Click vào điểm trên đồ thị để đánh dấu/bỏ đánh dấu anomaly")
        print("- Có thể zoom và pan trên đồ thị")
        print("- Save: Lưu nhãn vào file")
        print("- Clear All: Xóa tất cả nhãn")
        print("- Auto: Tự động phát hiện anomaly")
        print("- Stats: Hiển thị thống kê")
        print("- Load: Load lại nhãn từ file")
        print("- Reset Zoom: Đặt lại zoom về ban đầu")
        plt.show()

    def on_click(self, event):
        """Xử lý khi click vào đồ thị"""
        if event.inaxes != self.ax:
            return

        # Tọa độ click của chuột trong không gian hiển thị (pixel)
        click_xy_display = (event.x, event.y)
        
        # Lấy tọa độ của tất cả các điểm dữ liệu trong không gian hiển thị
        line_xy_display = self.ax.transData.transform(self.line.get_xydata())
        
        # Tính khoảng cách Euclidean trong không gian hiển thị (tính bằng pixel)
        distances = np.sqrt(np.sum((line_xy_display - click_xy_display)**2, axis=1))
        
        # Tìm index của điểm có khoảng cách nhỏ nhất
        closest_idx = np.argmin(distances)
        
        # Kiểm tra xem điểm gần nhất có đủ gần không (ví dụ: trong vòng 10 pixel)
        if distances[closest_idx] < 10.0:
            # Toggle anomaly
            if closest_idx in self.anomaly_points:
                self.anomaly_points.remove(closest_idx)
                self.labels[closest_idx] = 0
                print(f"Removed anomaly at index {closest_idx}, date: {self.current_data['date'].iloc[closest_idx].strftime('%Y-%m-%d')}, view: {self.current_data['view'].iloc[closest_idx]}")
            else:
                self.anomaly_points.append(closest_idx)
                self.labels[closest_idx] = 1
                print(f"Added anomaly at index {closest_idx}, date: {self.current_data['date'].iloc[closest_idx].strftime('%Y-%m-%d')}, view: {self.current_data['view'].iloc[closest_idx]}")
            
            # Cập nhật hiển thị
            self.update_plot()
        else:
            print(f"Click quá xa các điểm dữ liệu (khoảng cách nhỏ nhất: {distances[closest_idx]:.2f} pixels)")

    def update_plot(self):
        if not self.fig:
            return
            
        dates = self.current_data['date']
        views = self.current_data['view']
        
        if self.anomaly_points:
            anomaly_dates = dates.iloc[self.anomaly_points]
            anomaly_views = views.iloc[self.anomaly_points]
            
            # --- SỬA LỖI Ở ĐÂY ---
            # Thay vì dùng np.c_, hãy tạo một danh sách các cặp (x, y)
            # Matplotlib xử lý định dạng này rất tốt và tránh được lỗi DType
            offsets = list(zip(anomaly_dates, anomaly_views))
            self.anomaly_scatter.set_offsets(offsets)
            # --- KẾT THÚC SỬA LỖI ---
        else:
            # Truyền array rỗng đúng định dạng 2D
            self.anomaly_scatter.set_offsets(np.empty((0, 2)))
        
        # Cập nhật title với số lượng anomalies hiện tại
        anomaly_count = len(self.anomaly_points)
        total_count = len(self.current_data)
        self.ax.set_title(f'Manual Labeling for Place ID: {self.current_place_id} - Anomalies: {anomaly_count}/{total_count}')
        
        # Phải có legend để hiển thị
        self.ax.legend()
        self.fig.canvas.draw_idle()
    # ... Các hàm còn lại (load, save, clear, auto, stats, etc.) giữ nguyên ...
    def load_existing_labels(self, event=None):
        """Load nhãn hiện có từ file nếu có"""
        if self.current_place_id is None:
            return
        
        # Thử tìm file nhãn với các tên khác nhau
        possible_filenames = [
            f"label_{self.current_place_id}_cleaned.csv",  # File đã được làm sạch
            f"label_{self.current_place_id}.csv"           # File gốc
        ]
        
        loaded = False
        for filename in possible_filenames:
            filepath = os.path.join(self.labels_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    label_df = pd.read_csv(filepath)
                    if 'label' in label_df.columns:
                        loaded_labels = label_df['label'].values
                        if len(loaded_labels) == len(self.current_data):
                            self.labels = loaded_labels.astype(int)
                            self.anomaly_points = list(np.where(self.labels == 1)[0])
                            print(f"Đã load {len(self.anomaly_points)} nhãn anomaly từ {filepath}")
                            self.update_plot()
                            loaded = True
                            break
                        else:
                            print(f"Độ dài nhãn ({len(loaded_labels)}) không khớp với dữ liệu ({len(self.current_data)})")
                    else:
                        print(f"File {filepath} không có cột 'label'")
                except Exception as e:
                    print(f"Lỗi khi load nhãn từ {filepath}: {e}")
        
        if not loaded:
            print(f"Không tìm thấy file nhãn phù hợp cho place {self.current_place_id}")
            print(f"Đã tìm trong thư mục: {self.labels_dir}")
            print(f"Các file đã thử: {possible_filenames}")

    def reset_zoom(self, event=None):
        """Đặt lại zoom về ban đầu"""
        if hasattr(self, 'ax') and self.ax is not None:
            self.ax.autoscale()
            self.fig.canvas.draw_idle()
            print("Đã reset zoom")

    def save_labels(self, event=None):
        """Lưu nhãn vào file"""
        if self.current_place_id is None:
            print("Không có dữ liệu để lưu!")
            return
        
        # Tạo DataFrame với nhãn
        label_df = pd.DataFrame({
            'date': self.current_data['date'],
            'view': self.current_data['view'],
            'label': self.labels.astype(int)
        })
        
        # Lưu file
        filename = f"label_{self.current_place_id}.csv"
        filepath = os.path.join(self.labels_dir, filename)
        label_df.to_csv(filepath, index=False)
        
        anomaly_count = int(np.sum(self.labels))
        total_count = len(self.labels)
        print(f"\nĐã lưu nhãn vào: {filepath}")
        print(f"Tổng số anomalies: {anomaly_count}/{total_count} ({anomaly_count/total_count*100:.1f}%)")

    def clear_labels(self, event=None):
        """Xóa tất cả nhãn"""
        self.labels = np.zeros(len(self.current_data))
        self.anomaly_points = []
        self.update_plot()
        print("Đã xóa tất cả nhãn!")

    def auto_detect_anomalies(self, event=None):
        """Tự động phát hiện anomalies"""
        if self.current_data is None:
            return
        
        views = self.current_data['view'].values
        
        # Phương pháp 1: Z-score
        # Tránh chia cho 0 nếu std là 0
        mean_val = np.mean(views)
        std_val = np.std(views)
        if std_val == 0:
            z_scores = np.zeros_like(views)
        else:
            z_scores = np.abs((views - mean_val) / std_val)
        z_anomalies = z_scores > 2.5 # Tăng ngưỡng Z-score để chính xác hơn
        
        # Phương pháp 2: IQR
        q1 = np.percentile(views, 25)
        q3 = np.percentile(views, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_anomalies = (views < lower_bound) | (views > upper_bound)
        
        # Phương pháp 3: Phát hiện sụt giảm đột ngột về 0
        # Phát hiện khi view giảm hơn 90% so với giá trị trước đó và gần bằng 0
        diff = np.diff(views, prepend=views[0])
        relative_diff = np.divide(diff, views, out=np.zeros_like(diff, dtype=float), where=views!=0)
        sudden_drop_anomalies = (relative_diff < -0.9) & (views < 5)

        # Kết hợp các phương pháp
        combined_anomalies = z_anomalies | iqr_anomalies | sudden_drop_anomalies
        
        # Cập nhật labels
        self.labels = combined_anomalies.astype(int)
        self.anomaly_points = list(np.where(combined_anomalies)[0])
        
        self.update_plot()
        
        anomaly_count = np.sum(combined_anomalies)
        print(f"Tự động phát hiện {anomaly_count} anomalies")

    def show_stats(self, event=None):
        """Hiển thị thống kê"""
        if self.current_data is None:
            return
        
        views = self.current_data['view']
        anomaly_count = int(np.sum(self.labels))
        total_count = len(self.labels)
        
        print(f"\n=== THỐNG KÊ CHO PLACE {self.current_place_id} ===")
        print(f"Tổng số điểm: {total_count}")
        print(f"Số anomalies: {anomaly_count} ({anomaly_count/total_count*100:.1f}%)")
        print(f"Số normal: {total_count - anomaly_count} ({(total_count-anomaly_count)/total_count*100:.1f}%)")
        print(f"View trung bình: {views.mean():.2f}")
        print(f"View std: {views.std():.2f}")
        print(f"View min: {views.min()}")
        print(f"View max: {views.max()}")

    def batch_create_default_labels(self):
        """Tạo nhãn mặc định (tất cả normal) cho tất cả places"""
        print("Tạo nhãn mặc định cho tất cả places...")
        
        for place_id in self.place_ids:
            place_data = self.df[self.df['placeId'] == place_id].copy()
            place_data = place_data.sort_values('date')
            
            # Tạo nhãn mặc định (tất cả normal)
            labels = np.zeros(len(place_data))
            
            label_df = pd.DataFrame({
                'date': place_data['date'],
                'view': place_data['view'],
                'label': labels.astype(int)
            })
            
            filename = f"label_{place_id}.csv"
            filepath = os.path.join(self.labels_dir, filename)
            label_df.to_csv(filepath, index=False)
            
            print(f"Tạo {filepath} với {len(place_data)} điểm (tất cả normal)")

def main():
    """Hàm main để chạy công cụ"""
    print("=== CÔNG CỤ GÁN NHÃN THỦ CÔNG ===")
    
    try:
        tool = ManualLabelingTool()
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
        print("Vui lòng đảm bảo file dữ liệu tồn tại trong thư mục 'data/' hoặc '../data/'.")
        return

    while True:
        print("\n" + "="*50)
        print("MENU:")
        print("1. Hiển thị danh sách Place IDs")
        print("2. Gán nhãn thủ công cho một place")
        print("3. Tạo nhãn mặc định cho tất cả places")
        # --- THAY ĐỔI MENU ---
        print("4. Xóa các place có độ dài < 30 và lưu file mới")
        print("5. Thoát")
        
        choice = input("Chọn (1-5): ").strip()
        # --- KẾT THÚC THAY ĐỔI ---
        
        if choice == '1':
            tool.list_places()
            
        elif choice == '2':
            tool.list_places()
            try:
                place_idx_input = input(f"Chọn Place ID (nhập số từ 1-{len(tool.place_ids)}): ")
                if not place_idx_input: continue # Bỏ qua nếu người dùng không nhập gì
                
                place_idx = int(place_idx_input) - 1
                if 0 <= place_idx < len(tool.place_ids):
                    place_id = tool.place_ids[place_idx]
                    tool.load_place_data(place_id)
                    tool.interactive_labeling()
                else:
                    print("Lựa chọn không hợp lệ!")
            except ValueError:
                print("Vui lòng nhập số!")
                
        elif choice == '3':
            confirm = input("Hành động này sẽ ghi đè các file nhãn hiện có. Tạo nhãn mặc định? (y/n): ")
            if confirm.lower() == 'y':
                tool.batch_create_default_labels()

        # --- THÊM LOGIC MỚI ---
        elif choice == '4':
            print("Hành động này sẽ tạo một file dữ liệu mới chỉ chứa các chuỗi thời gian có độ dài >= 30.")
            print("File gốc sẽ không bị thay đổi.")
            confirm = input("Bạn có chắc chắn muốn tiếp tục? (y/n): ")
            if confirm.lower() == 'y':
                tool.filter_and_save_long_series(min_length=30)
        # --- KẾT THÚC LOGIC MỚI ---
                
        elif choice == '5':
            print("Tạm biệt!")
            break
            
        else:
            print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()