import pandas as pd

def round_view_values():
    """
    Làm tròn các giá trị view trong file cleaned_data_smoothed.csv
    """
    file_path = 'data/cleaned_data_smoothed.csv'
    
    print(f"Đang đọc file: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Số dòng ban đầu: {len(df)}")
    print(f"Kiểu dữ liệu view ban đầu: {df['view'].dtype}")
    print(f"Mẫu giá trị view ban đầu:\n{df['view'].head(10)}")
    
    # Làm tròn giá trị view thành số nguyên
    df['view'] = df['view'].round().astype(int)
    
    print(f"\nSau khi làm tròn:")
    print(f"Kiểu dữ liệu view: {df['view'].dtype}")
    print(f"Mẫu giá trị view:\n{df['view'].head(10)}")
    
    # Lưu lại file
    df.to_csv(file_path, index=False)
    print(f"\nĐã lưu file đã làm tròn vào: {file_path}")
    
    return df

if __name__ == '__main__':
    rounded_df = round_view_values()
