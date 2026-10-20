# UTS Machine Learning – Linear Regression & Logistic Regression

- **Nama  :** RIZKY WISNUSAPUTRA
- **NIM   :** 241011401458
- **Kelas :** Reg B
- **random_state :** 58 (2 digit terakhir NIM)

## Isi Repositori
```
uts-machine-learning-241011401458/
├── README.md
├── tugas_UTS.ipynb      # sudah dijalankan lengkap (output & grafik)
├── Ads_Data.csv
└── Email_Data.csv
```
Model dan metrik (`train_test_split`, `LinearRegression`, `LinearRegressionGD`, `LogisticRegression`, MAE/MSE/RMSE/R², accuracy, confusion matrix, precision, recall, F1) diimplementasikan sendiri dengan NumPy.

## Ringkasan Hasil

### Bagian A – Linear Regression (`Ads_Data.csv`, split 80:20)
- Persamaan (normal equation): `y_hat = 26.9641 + 9.3156 * BiayaIklan` → tiap tambahan biaya iklan 1 juta Rp, penjualan naik ± 9.32 juta Rp.

| Data | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Training | 4.3504 | 36.2741 | 6.0228 | 0.9472 |
| Testing | 5.1478 | 36.7506 | 6.0622 | 0.8611 |

- Gradient descent (5000 iterasi): lr 0.001 terlalu lambat (b = 23.62, m = 9.81); lr 0.02 pas (sama dengan normal equation); lr 0.1 divergen (NaN, batas stabil lr ≈ 0.0268).
- Prediksi: 4.5 juta → 68.88; 7 juta → 92.17; 12 juta → 138.75 (ekstrapolasi, kurang dapat dipercaya).

### Bagian B – Logistic Regression (`Email_Data.csv`, split 75:25)
- Persamaan: `P(Spam) = sigmoid(-10.4196 + 0.8593 * JumlahLink + 0.8887 * KataPromo)`
- Evaluasi data testing (threshold 0.5): accuracy **0.8667**, precision **1.0**, recall **0.6667**, F1 **0.8** (TN = 9, FP = 0, FN = 2, TP = 4).
- Eksperimen threshold: 0.3 → precision 0.80 / recall 0.67; 0.5 dan 0.7 → precision 1.0 / recall 0.67. Threshold pilihan: **0.7** (FP dianggap lebih merugikan).
- Prediksi email baru (threshold 0.7): E1 Bukan Spam, E2 Bukan Spam, E3 Spam, E4 Spam.

### Bagian C
- Jawaban pemahaman konsep ada di notebook.

## Cara Menjalankan
Buka `tugas_UTS.ipynb` lalu *Run All* (butuh `numpy`, `pandas`, dan `matplotlib`).
