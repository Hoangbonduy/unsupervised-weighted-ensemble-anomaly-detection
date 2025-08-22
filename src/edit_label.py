import pandas as pd
import os
from pathlib import Path

def fix_label_files():
    """
    Sửa các file label để đồng bộ với dữ liệu trong cleaned_data_no_zero_periods_filtered.csv
    """
    # Đọc dữ liệu chính
    print("Đang đọc dữ liệu chính...")
    main_data = pd.read_csv('../data/cleaned_data_no_zero_periods_filtered.csv')
    
    # Đường dẫn đến thư mục labels
    labels_dir = Path('../labels')
    
    # Lấy danh sách tất cả các placeId có trong dữ liệu chính
    unique_place_ids = main_data['placeId'].unique()
    print(f"Tìm thấy {len(unique_place_ids)} placeId trong dữ liệu chính")
    
    # Xử lý từng file label
    processed_count = 0
    error_count = 0
    
    for place_id in unique_place_ids:
        label_file = labels_dir / f'label_{place_id}.csv'
        
        if not label_file.exists():
            print(f"⚠️  Không tìm thấy file label cho placeId {place_id}")
            continue
            
        try:
            # Đọc file label hiện tại
            label_df = pd.read_csv(label_file)
            
            # Lấy dữ liệu của placeId này từ file chính
            place_data = main_data[main_data['placeId'] == place_id][['date', 'view']].copy()
            
            if len(place_data) == 0:
                print(f"⚠️  Không có dữ liệu cho placeId {place_id} trong file chính")
                continue
            
            # Kiểm tra xem có cần sửa không
            if len(label_df) == len(place_data):
                # Kiểm tra xem dates có giống nhau không
                label_dates = set(label_df['date'].astype(str))
                main_dates = set(place_data['date'].astype(str))
                
                if label_dates == main_dates:
                    print(f"✅ PlaceId {place_id}: Đã đồng bộ (không cần sửa)")
                    continue
            
            print(f"🔧 Đang sửa PlaceId {place_id}...")
            print(f"   - Label file có {len(label_df)} records")
            print(f"   - Main data có {len(place_data)} records")
            
            # Tạo DataFrame mới với structure đúng
            # Giữ nguyên các label hiện có nếu date tồn tại, nếu không thì gán label = 0
            new_label_df = place_data.copy()
            new_label_df['label'] = 0  # Mặc định là 0
            
            # Merge với label cũ để giữ lại các label đã có
            label_df_clean = label_df.copy()
            label_df_clean['date'] = label_df_clean['date'].astype(str)
            new_label_df['date'] = new_label_df['date'].astype(str)
            
            # Merge để giữ lại các label cũ
            merged = new_label_df.merge(
                label_df_clean[['date', 'label']], 
                on='date', 
                how='left', 
                suffixes=('', '_old')
            )
            
            # Sử dụng label cũ nếu có, nếu không thì dùng 0
            merged['label'] = merged['label_old'].fillna(merged['label']).astype(int)
            
            # Chỉ giữ lại các cột cần thiết
            final_df = merged[['date', 'view', 'label']].copy()
            
            # Kiểm tra số lượng label = 1
            old_anomaly_count = label_df['label'].sum() if 'label' in label_df.columns else 0
            new_anomaly_count = final_df['label'].sum()
            
            print(f"   - Anomaly labels: {old_anomaly_count} → {new_anomaly_count}")
            
            # Lưu file mới
            final_df.to_csv(label_file, index=False)
            print(f"✅ Đã lưu file label cho placeId {place_id}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý placeId {place_id}: {str(e)}")
            error_count += 1
    
    print(f"\n📊 Kết quả:")
    print(f"   - Đã xử lý thành công: {processed_count} files")
    print(f"   - Gặp lỗi: {error_count} files")
    print(f"   - Tổng số placeId: {len(unique_place_ids)}")

def verify_label_files():
    """
    Kiểm tra xem các file label đã được đồng bộ chưa
    """
    print("\n🔍 Đang kiểm tra tính đồng bộ của các file label...")
    
    # Đọc dữ liệu chính
    main_data = pd.read_csv('../data/cleaned_data_no_zero_periods_filtered.csv')
    labels_dir = Path('../labels')
    
    sync_count = 0
    total_count = 0
    
    for label_file in labels_dir.glob('label_*.csv'):
        # Trích xuất placeId từ tên file
        place_id = int(label_file.stem.replace('label_', ''))
        
        try:
            # Đọc file label
            label_df = pd.read_csv(label_file)
            
            # Lấy dữ liệu tương ứng từ file chính
            place_data = main_data[main_data['placeId'] == place_id]
            
            # Kiểm tra đồng bộ
            if len(label_df) == len(place_data):
                label_dates = set(label_df['date'].astype(str))
                main_dates = set(place_data['date'].astype(str))
                
                if label_dates == main_dates:
                    sync_count += 1
                    print(f"✅ {label_file.name}: Đồng bộ ({len(label_df)} records)")
                else:
                    print(f"⚠️  {label_file.name}: Dates không khớp")
            else:
                print(f"⚠️  {label_file.name}: Số lượng records không khớp ({len(label_df)} vs {len(place_data)})")
            
            total_count += 1
            
        except Exception as e:
            print(f"❌ Lỗi khi kiểm tra {label_file.name}: {str(e)}")
    
    print(f"\n📊 Kết quả kiểm tra:")
    print(f"   - Files đồng bộ: {sync_count}/{total_count}")
    print(f"   - Tỷ lệ đồng bộ: {sync_count/total_count*100:.1f}%")

if __name__ == '__main__':
    print("🚀 Bắt đầu quá trình sửa file labels...")
    
    # Kiểm tra trước khi sửa
    verify_label_files()
    
    # Sửa file labels
    fix_label_files()
    
    # Kiểm tra sau khi sửa
    verify_label_files()
    
    print("\n🎉 Hoàn thành!")
