#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Manual Labeling Script
Script gán nhãn thủ công đơn giản
"""

import pandas as pd
import numpy as np
import os

def create_simple_labels():
    """Tạo nhãn đơn giản bằng cách nhập thủ công"""
    
    # Load dữ liệu
    data_path = "data/Place_view_processed.csv"
    df = pd.read_csv(data_path)
    place_ids = df['placeId'].unique()
    
    print(f"Tìm thấy {len(place_ids)} place IDs")
    
    # Tạo thư mục labels
    os.makedirs("data/labels", exist_ok=True)
    
    for i, place_id in enumerate(place_ids):
        print(f"\n=== PLACE {i+1}/{len(place_ids)}: {place_id} ===")
        
        # Load dữ liệu place
        place_data = df[df['placeId'] == place_id].copy()
        place_data = place_data.sort_values('date')
        
        print(f"Số điểm dữ liệu: {len(place_data)}")
        print(f"View range: {place_data['view'].min()} - {place_data['view'].max()}")
        print(f"View trung bình: {place_data['view'].mean():.2f}")
        
        # Hiển thị một số điểm dữ liệu mẫu
        print("\nMột số điểm dữ liệu mẫu:")
        sample_data = place_data.head(10)
        for idx, row in sample_data.iterrows():
            print(f"  {row['date']}: {row['view']} views")
        
        # Lựa chọn phương pháp gán nhãn
        print("\nLựa chọn phương pháp gán nhãn:")
        print("1. Tất cả normal (0)")
        print("2. Gán nhãn dựa trên threshold")
        print("3. Gán nhãn cho các điểm cụ thể")
        print("4. Bỏ qua place này")
        
        choice = input("Chọn (1-4): ").strip()
        
        labels = np.zeros(len(place_data))  # Mặc định tất cả là normal
        
        if choice == '1':
            # Tất cả normal
            pass
            
        elif choice == '2':
            # Threshold-based labeling
            print(f"View trung bình: {place_data['view'].mean():.2f}")
            print(f"View std: {place_data['view'].std():.2f}")
            
            try:
                threshold_low = float(input("Nhập threshold thấp (views < threshold = anomaly): ") or "0")
                threshold_high = float(input("Nhập threshold cao (views > threshold = anomaly): ") or str(place_data['view'].max() + 1))
                
                anomaly_mask = (place_data['view'] < threshold_low) | (place_data['view'] > threshold_high)
                labels[anomaly_mask] = 1
                
                print(f"Đã đánh dấu {np.sum(labels)} điểm là anomaly")
                
            except ValueError:
                print("Threshold không hợp lệ, sử dụng tất cả normal")
                
        elif choice == '3':
            # Gán nhãn cho các điểm cụ thể
            print("Nhập các chỉ số điểm bất thường (cách nhau bởi dấu phẩy):")
            print(f"Phạm vi hợp lệ: 0 - {len(place_data)-1}")
            
            indices_str = input("Các chỉ số (ví dụ: 0,5,10,20): ").strip()
            if indices_str:
                try:
                    indices = [int(x.strip()) for x in indices_str.split(',')]
                    valid_indices = [i for i in indices if 0 <= i < len(place_data)]
                    labels[valid_indices] = 1
                    print(f"Đã đánh dấu {len(valid_indices)} điểm là anomaly")
                except ValueError:
                    print("Chỉ số không hợp lệ, sử dụng tất cả normal")
                    
        elif choice == '4':
            print("Bỏ qua place này")
            continue
        
        # Lưu nhãn
        label_df = pd.DataFrame({
            'date': place_data['date'],
            'view': place_data['view'],
            'label': labels.astype(int)
        })
        
        filename = f"labels/label_{place_id}.csv"
        label_df.to_csv(filename, index=False)
        
        anomaly_count = int(np.sum(labels))
        total_count = len(labels)
        print(f"Đã lưu: {filename}")
        print(f"Anomalies: {anomaly_count}/{total_count} ({anomaly_count/total_count*100:.1f}%)")

if __name__ == "__main__":
    print("=== SCRIPT GÁN NHÃN THỦ CÔNG ĐƠN GIẢN ===")
    create_simple_labels()
    print("\nHoàn thành!")
