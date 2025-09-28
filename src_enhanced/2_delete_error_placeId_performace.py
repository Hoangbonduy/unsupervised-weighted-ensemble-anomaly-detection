import pandas as pd

def delete_error_place_ids():
    """
    Xóa các dòng có placeId bị lỗi khỏi file performance_matrix_P.csv
    """
    # Đường dẫn file
    file_path = 'results_enhanced_final/performance_matrix_P.csv'
    
    # Danh sách các placeId cần xóa
    error_place_ids = [
        4717964588208123629, 
        4701454798439365089, 
        4642045776824553693, 
        4634145114310470202
    ]
    
    print(f"Đang đọc file: {file_path}")
    
    # Đọc file CSV
    df = pd.read_csv(file_path)
    
    print(f"Số dòng ban đầu: {len(df)}")
    print(f"Các placeId cần xóa: {error_place_ids}")
    
    # Kiểm tra xem các placeId có tồn tại trong file không
    existing_error_ids = df[df['place_id'].isin(error_place_ids)]['place_id'].tolist()
    print(f"Các placeId tìm thấy trong file: {existing_error_ids}")
    
    # Xóa các dòng có placeId trong danh sách lỗi
    df_cleaned = df[~df['place_id'].isin(error_place_ids)]
    
    print(f"Số dòng sau khi xóa: {len(df_cleaned)}")
    print(f"Đã xóa {len(df) - len(df_cleaned)} dòng")
    
    # Lưu lại chính file đó
    df_cleaned.to_csv(file_path, index=False)
    print(f"Đã lưu file đã được làm sạch vào: {file_path}")
    
    # Hiển thị một vài dòng đầu để kiểm tra
    print("\nMột vài dòng đầu sau khi xóa:")
    print(df_cleaned.head())
    
    return df_cleaned

if __name__ == '__main__':
    delete_error_place_ids()
