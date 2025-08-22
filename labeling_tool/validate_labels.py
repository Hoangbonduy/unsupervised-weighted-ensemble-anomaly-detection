#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label Validation Script
Script kiểm tra và validate các file nhãn
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def validate_label_files():
    """Kiểm tra tính hợp lệ của các file nhãn"""
    
    # Load dữ liệu gốc
    data_path = "data/Place_view_processed.csv"
    if not os.path.exists(data_path):
        print(f"Không tìm thấy file dữ liệu: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    place_ids = df['placeId'].unique()
    
    labels_dir = "labels"
    if not os.path.exists(labels_dir):
        print(f"Không tìm thấy thư mục nhãn: {labels_dir}")
        return
    
    print("=== KIỂM TRA TÍNH HỢP LỆ CỦA CÁC FILE NHÃN ===\n")
    
    valid_files = 0
    invalid_files = 0
    missing_files = 0
    
    for place_id in place_ids:
        print(f"Kiểm tra Place ID: {place_id}")
        
        # Load dữ liệu gốc cho place này
        place_data = df[df['placeId'] == place_id].copy()
        place_data = place_data.sort_values('date')
        
        # Tìm file nhãn
        label_file = os.path.join(labels_dir, f"label_{place_id}.csv")
        
        if not os.path.exists(label_file):
            print(f"  ❌ Không tìm thấy file nhãn: {label_file}")
            missing_files += 1
            continue
        
        try:
            # Load file nhãn
            label_df = pd.read_csv(label_file)
            
            print(f"  📁 File nhãn: {label_file}")
            print(f"  📊 Cấu trúc: {list(label_df.columns)}")
            print(f"  📏 Số dòng: {len(label_df)}")
            
            # Kiểm tra các cột cần thiết
            has_issues = False
            
            if 'label' not in label_df.columns:
                print(f"  ⚠️  Thiếu cột 'label'")
                has_issues = True
            else:
                # Kiểm tra giá trị nhãn
                unique_labels = label_df['label'].unique()
                valid_labels = all(label in [0, 1] for label in unique_labels)
                if not valid_labels:
                    print(f"  ⚠️  Nhãn không hợp lệ: {unique_labels} (chỉ chấp nhận 0 và 1)")
                    has_issues = True
                else:
                    anomaly_count = (label_df['label'] == 1).sum()
                    anomaly_ratio = anomaly_count / len(label_df) * 100
                    print(f"  ✅ Nhãn hợp lệ: {anomaly_count}/{len(label_df)} anomalies ({anomaly_ratio:.1f}%)")
            
            # Kiểm tra độ dài
            if len(label_df) != len(place_data):
                print(f"  ⚠️  Độ dài không khớp: nhãn={len(label_df)}, dữ liệu={len(place_data)}")
                has_issues = True
            else:
                print(f"  ✅ Độ dài khớp: {len(label_df)} điểm")
            
            # Kiểm tra thời gian (nếu có)
            if 'date' in label_df.columns:
                label_df['date'] = pd.to_datetime(label_df['date'])
                
                # Kiểm tra thứ tự thời gian
                is_sorted = label_df['date'].is_monotonic_increasing
                if not is_sorted:
                    print(f"  ⚠️  Thời gian không được sắp xếp theo thứ tự")
                    has_issues = True
                else:
                    print(f"  ✅ Thời gian đã sắp xếp đúng")
                
                # Kiểm tra khoảng thời gian
                label_start = label_df['date'].min()
                label_end = label_df['date'].max()
                data_start = place_data['date'].min()
                data_end = place_data['date'].max()
                
                print(f"  📅 Khoảng thời gian nhãn: {label_start.date()} đến {label_end.date()}")
                print(f"  📅 Khoảng thời gian dữ liệu: {data_start.date()} đến {data_end.date()}")
                
                if label_start.date() != data_start.date() or label_end.date() != data_end.date():
                    print(f"  ⚠️  Khoảng thời gian không khớp")
                    has_issues = True
                else:
                    print(f"  ✅ Khoảng thời gian khớp")
            
            # Tổng kết cho place này
            if has_issues:
                print(f"  ❌ Place {place_id}: CÓ VẤN ĐỀ")
                invalid_files += 1
            else:
                print(f"  ✅ Place {place_id}: HỢP LỆ")
                valid_files += 1
                
        except Exception as e:
            print(f"  ❌ Lỗi khi đọc file: {e}")
            invalid_files += 1
        
        print("-" * 50)
    
    # Tổng kết
    total_places = len(place_ids)
    print(f"\n=== TỔNG KẾT ===")
    print(f"Tổng số places: {total_places}")
    print(f"File nhãn hợp lệ: {valid_files}")
    print(f"File nhãn có vấn đề: {invalid_files}")
    print(f"File nhãn thiếu: {missing_files}")
    
    if missing_files > 0:
        print(f"\n⚠️  Có {missing_files} places chưa có file nhãn!")
        print("Chạy script sau để tạo nhãn mặc định:")
        print("python simple_labeling.py")
    
    if invalid_files > 0:
        print(f"\n⚠️  Có {invalid_files} file nhãn có vấn đề!")
        print("Vui lòng kiểm tra và sửa các file trên.")
    
    if valid_files == total_places:
        print(f"\n🎉 Tất cả {total_places} file nhãn đều hợp lệ!")

def show_label_statistics():
    """Hiển thị thống kê tổng quan về nhãn"""
    
    labels_dir = "data/labels"
    if not os.path.exists(labels_dir):
        print(f"Không tìm thấy thư mục nhãn: {labels_dir}")
        return
    
    print("=== THỐNG KÊ NHÃN TỔNG QUAN ===\n")
    
    label_files = [f for f in os.listdir(labels_dir) if f.startswith('label_') and f.endswith('.csv')]
    
    if not label_files:
        print("Không tìm thấy file nhãn nào!")
        return
    
    total_points = 0
    total_anomalies = 0
    anomaly_ratios = []
    
    for label_file in label_files:
        try:
            file_path = os.path.join(labels_dir, label_file)
            df = pd.read_csv(file_path)
            
            if 'label' in df.columns:
                points = len(df)
                anomalies = (df['label'] == 1).sum()
                ratio = anomalies / points * 100 if points > 0 else 0
                
                total_points += points
                total_anomalies += anomalies
                anomaly_ratios.append(ratio)
                
                place_id = label_file.replace('label_', '').replace('.csv', '')
                print(f"Place {place_id}: {anomalies}/{points} anomalies ({ratio:.1f}%)")
        
        except Exception as e:
            print(f"Lỗi đọc {label_file}: {e}")
    
    if total_points > 0:
        overall_ratio = total_anomalies / total_points * 100
        avg_ratio = np.mean(anomaly_ratios)
        std_ratio = np.std(anomaly_ratios)
        
        print(f"\n=== TỔNG KẾT ===")
        print(f"Tổng số điểm: {total_points:,}")
        print(f"Tổng số anomalies: {total_anomalies:,}")
        print(f"Tỷ lệ anomaly tổng: {overall_ratio:.2f}%")
        print(f"Tỷ lệ anomaly trung bình: {avg_ratio:.2f}% ± {std_ratio:.2f}%")
        print(f"Tỷ lệ anomaly min: {min(anomaly_ratios):.2f}%")
        print(f"Tỷ lệ anomaly max: {max(anomaly_ratios):.2f}%")

def main():
    """Menu chính"""
    print("=== CÔNG CỤ KIỂM TRA NHÃN ===")
    
    while True:
        print("\nLựa chọn:")
        print("1. Kiểm tra tính hợp lệ của file nhãn")
        print("2. Hiển thị thống kê nhãn")
        print("3. Thoát")
        
        choice = input("Chọn (1-3): ").strip()
        
        if choice == '1':
            validate_label_files()
        elif choice == '2':
            show_label_statistics()
        elif choice == '3':
            print("Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()
