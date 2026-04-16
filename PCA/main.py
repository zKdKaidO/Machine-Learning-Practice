import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Xử lý đường dẫn ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from DATASET.dataset import Dataset
from pca import PCA
from BAYESIAN_LEARNING.gaussian import GaussianNB
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    print("========== BƯỚC 1: ĐỌC DỮ LIỆU ==========")
    dataset_path = r"D:\AI\Machine-Learning-Practice\BAYESIAN_LEARNING\salmon_seabass.csv"
    
    df = Dataset(dataset_path)
    X, y = df.get_data()
    X_train, X_test, y_train, y_test = df.split_data()
    
    print(f"Số lượng Train: {X_train.shape[0]}")
    print(f"Số lượng Test: {X_test.shape[0]}\n")

    print("========== BƯỚC 2: CHẠY PCA (2D -> 1D) ==========")
    model_pca = PCA(n_components=1)
    model_pca.fit(X_train)
    
    X_train_1d_pca = model_pca.transform(X_train)
    X_test_1d_pca = model_pca.transform(X_test)
    
    print(f"Kích thước X_train GỐC: {X_train.shape}")
    print(f"Kích thước X_train SAU NÉN: {X_train_1d_pca.shape}")
    
    # Khôi phục
    X_train_reconstructed = model_pca.inverse_transform(X_train_1d_pca)


    print("\n========== BƯỚC 3: KIỂM THỬ (TEST) VỚI NAIVE BAYES ==========")
    model_1d_original = GaussianNB()
    model_1d_original.fit(X_train[:, 0:1], y_train)
    acc_1d_orig = accuracy_score(y_test, model_1d_original.predict(X_test[:, 0:1]))

    model_1d_pca = GaussianNB()
    model_1d_pca.fit(X_train_1d_pca, y_train)
    acc_1d_pca = accuracy_score(y_test, model_1d_pca.predict(X_test_1d_pca))

    model_2d = GaussianNB()
    model_2d.fit(X_train, y_train)
    acc_2d = accuracy_score(y_test, model_2d.predict(X_test))

    print(f"Độ chính xác (1D - Bỏ cột độ sáng):       {acc_1d_orig * 100:.2f}%")
    print(f"Độ chính xác (1D - Nén bằng PCA):         {acc_1d_pca * 100:.2f}%  <-- Sức mạnh của PCA!")
    print(f"Độ chính xác (2D - Full 2 cột ban đầu):   {acc_2d * 100:.2f}%\n")


    print("========== BƯỚC 4: VẼ ĐỒ THỊ MINH HỌA SỰ TƯƠNG QUAN ==========")
    
    plt.figure(figsize=(10, 8))
    
    plt.scatter(X_train[:, 0], X_train[:, 1], alpha=0.3, label='Dữ liệu gốc (2D)', color='blue')
    
    # 2. Vẽ các điểm dữ liệu sau khi giải nén (Nằm xếp hàng thẳng tắp trên 1 đường thẳng)
    plt.scatter(X_train_reconstructed[:, 0], X_train_reconstructed[:, 1], alpha=0.8, label='Dữ liệu chiếu PCA (1D ép lên 2D)', color='red', marker='x')
    
    # 3. Vẽ các đường nét đứt biểu diễn "Lỗi tái tạo" (Reconstruction Error)
    # Lấy ngẫu nhiên 50 điểm để vẽ cho đỡ rối mắt
    for i in range(50):
        plt.plot([X_train[i, 0], X_train_reconstructed[i, 0]], 
                 [X_train[i, 1], X_train_reconstructed[i, 1]], 
                 'k--', alpha=0.2)
        
    # 4. Vẽ vector trục chính (Eigenvector)
    mu = model_pca.mean
    vec = model_pca.components[:, 0]
    # Kéo dài vector ra 2 phía để tạo thành đường thẳng
    scale = 3 
    plt.plot([mu[0] - vec[0]*scale, mu[0] + vec[0]*scale], 
             [mu[1] - vec[1]*scale, mu[1] + vec[1]*scale], 
             color='green', linewidth=3, label='Trục chính (PC1)')
    
    plt.title("PCA: Nén dữ liệu 2 Chiều xuống 1 Chiều")
    plt.xlabel("Đặc trưng 1 (Width)")
    plt.ylabel("Đặc trưng 2 (Lightness)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Giữ đúng tỷ lệ để thấy đường chiếu vuông góc
    plt.show()