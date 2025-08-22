    try:
                s = fn(series_df)
            except Exception:
                s = np.zeros(length)
            raw_scores_real_data[m] = s

        # --- GIAI ĐOẠN 1: IMPROVED SYNTHETIC EVALUATION + REAL VALIDATION ---
        ts_np = series_df['view'].to_numpy()
        
        # Tạo realistic synthetic anomalies
        realistic_injection_tests = cr