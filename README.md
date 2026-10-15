# 🔢 MNIST 10–20 Challenge: Hierarchical Multi-MLP & Domain-Matched Pipeline

Hệ thống nhận diện chữ số viết tay ghép đôi (11 lớp: từ `10` đến `20`) trên ảnh xám `28x28`, được thiết kế dưới ràng buộc nghiêm ngặt **100% Pure MLP (chỉ sử dụng `nn.Linear`, tuyệt đối không dùng `Conv1d/2d/3d`)**. 

Giải pháp kết hợp quy trình tái tạo dữ liệu nét vẽ (`Vector Stroke Rendering`) khớp với miền vật lý của tập Test, hệ thống Đa-MLP Phân cấp (`Hierarchical Multi-MLP`) có khả năng bất biến dịch chuyển cục bộ, và thuật toán cắt tách chữ số có ràng buộc khối lượng tích lũy (`Dual-Peak Cumulative-Mass Voronoi Cut`).

---

## 📈 Lịch sử Tiến triển Hiệu năng (Performance Progression)

Hệ thống được kiểm định nghiêm ngặt qua **5-Fold GroupKFold theo người viết (`writer overlap = 0` tuyệt đối)** trên `5.383` mẫu định danh (109 người viết) và đối chứng trực tiếp trên Kaggle Public Leaderboard (`220` mẫu = 40% Test):

| Phiên bản | Kiến trúc & Kỹ thuật cốt lõi | OOF `Capped_Style` (4.932 mẫu) | OOF `Raw` (4.932 mẫu) | Held-out New Writers (451 mẫu) | Public LB (220 mẫu) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline `v5`** | Flat MLP (`784→512→512→256→11`) + Shift-9 TTA | `96.07%` (194 lỗi) | `96.88%` (154 lỗi) | `96.67%` (15 lỗi) | `0.95000` (209/220) |
| **`v7` Sinkhorn** | Flat MLP Ensemble + Cân bằng Hungarian | `96.69%` (163 lỗi) | `96.95%` (150 lỗi) | `98.00%` (9 lỗi) | `0.95454` (210/220) |
| **`v12` Fourier** | Flat MLP + Bicubic Smooth-Deskew + 2D-Fourier | `96.88%` (154 lỗi) | `97.00%` (148 lỗi) | `97.78%` (10 lỗi) | `0.95000` (209/220) |
| **`v4` Cell 1–3** | **HierMultiMLP** (Huấn luyện từ đầu, Crop `18x18` + Patch-Mixer) | `97.73%` (112 lỗi) | `97.91%` (103 lỗi) | `98.67%` (6 lỗi) | `0.96818` (213/220) |
| **`v4` Cell 4** | **Ultimate20x20HierMLP** (Voronoi Crop `20x20` + 409 mẫu vô danh + Full-Data) | `97.71%` (113 lỗi) | `97.95%` (101 lỗi) | **`99.11%` (4 lỗi)** | `0.97272` (214/220) |
| **`v4` Cell 5** | **DualPeak-HardFT** (Ràng buộc khối lượng cắt `[8..15]` + Focal Loss + Surgical Gate) | **`97.83%` (107 lỗi)** | **`97.99%` (99 lỗi)** | `98.67%` (6 lỗi) | **`0.97727` (215/220)** |

---

## 🛠️ 1. Quy trình Xây dựng & Chuẩn hóa Tập Dữ liệu Huấn luyện (Training Data Pipeline)

Phân tích thống kê trên tập `test.csv` (`550` mẫu) cho thấy ảnh Test không phải là ảnh cắt thô mà có đặc trưng vật lý rất cụ thể: tỷ lệ điểm ảnh tiền cảnh `fg = 0.2044`, tỷ lệ điểm bão hòa mức trắng tuyệt đối (`255`) đạt `sat = 0.3834`, chiều cao và chiều rộng trung vị `h_med = 19.0, w_med = 19.0`, và chỉ tồn tại đúng **80 mức xám rời rạc (`V3_LEVELS`)**. 

Để khớp tuyệt đối với miền dữ liệu của `test.csv`, quy trình tạo tập Train được thiết kế qua 5 bước:

### 1.1. Khôi phục ảnh tràn khung HTML & Làm sạch nhãn (Data Cleaning & Unclipping)
* **Khôi phục nét tràn viền (`Dynamic BBox Unclipped Rendering`):** Trong quá trình thu thập qua giao diện web, nhiều người viết vẽ vượt ra ngoài khung canvas cố định `600×400` khiến ảnh xuất thẳng từ trình duyệt bị cụt nét. Hệ thống đọc trực tiếp tọa độ vector từ chuỗi JSON `strokes`, tính khung bao động `[low - 3, high + 4]` để vẽ lại toàn bộ các mẫu bị tràn viền (khôi phục thành công 59 mẫu tràn viền trong `data_v31.csv`).
* **Chuẩn hóa kiểu JS (`norm_js`):** Cắt vùng pixel `> 20`, co giãn giữ nguyên tỷ lệ (aspect-ratio preserving) bằng phép lấy mẫu diện tích chính xác (`Exact Area-Weights Resampling`) về khung tối đa `20×20`, chuẩn hóa đỉnh sáng về `255`, và đặt trọng tâm khối mực (`Center of Mass`) vào đúng tọa độ `(14, 14)` trên nền `28×28`.
* **Loại bỏ mẫu hỏng:** Loại bỏ 4 dòng lỗi nhãn trong tập cũ (`[458, 1767, 2213, 4355]`) và 18 dòng nhiễu/biến dạng nặng trong `data_v31.csv`. Khai thác thêm `409` mẫu viết tay thật bị khuyết tên người viết (`Anonymous Clean Samples` có độ đồng thuận `>= 0.80`).
* **Tổng quy mô mẫu gốc sạch:** `4.932` (94 người viết cũ) + `451` (15 người viết mới) + `409` (mẫu vô danh sạch) = **`5.792` mẫu viết tay gốc**.

### 1.2. Đa dạng hóa từ Tọa độ Vector (`Capped Strokes` & `Cross-Writer Recombination`)
Thay vì chỉ biến đổi trên ảnh `28×28` độ phân giải thấp (dễ gây vỡ hạt), mọi phép tăng cường dữ liệu đều được thực hiện trên **tọa độ vector độ phân giải cao** với cơ chế vẽ siêu lấy mẫu `2×` (`Supersampled PIL Curve Rendering`) rồi mới thu về `28×28` và ép về đúng **80 mức xám (`V3_LEVELS`)**:
1. **`Capped Strokes` (4 biến thể / mẫu gốc — ~23.000 ảnh):**
   * Biến thiên ngẫu nhiên độ nghiêng `shear ∈ [-0.25, +0.25]`, tỷ lệ co giãn ngang `squeeze_x ∈ [0.72, 1.02]`, và độ dày nét bút `pen ∈ [25, 47]`.
   * Áp dụng trần giới hạn độ phình nét **`pen_scale <= 1.25` (`cap = 1.25`)** để ngăn chặn hiện tượng nét bút quá dày làm đặc kín lỗ rỗng của các chữ số `0, 6, 8, 9`.
2. **`Cross-Writer Digit Recombination` (`Recomb` — ~43.000 ảnh):**
   * Tự động phân tích khoảng hở ngang giữa các nét vẽ (`strokes`) để tách riêng **Chữ số Hàng chục (`1` hoặc `2`)** và **Chữ số Hàng đơn vị (`0..9`)** trên các mẫu có khe tách biệt rõ (`overlap_score < 0.08` và `ties == 1`).
   * Ghép chéo chữ số Hàng chục của người viết $A$ với chữ số Hàng đơn vị của người viết $B$ (cân chỉnh chiều cao tương đối và tạo khoảng cách tự nhiên `gap ∈ [18, 52]`), nhân bản số lượng kiểu chữ lên gấp hàng chục lần.
   * **Bảo toàn tính độc lập 5-Fold CV:** Mỗi ảnh ghép chéo lưu rõ `fold_a` và `fold_b`. Khi huấn luyện Fold $k$, chỉ những ảnh thỏa mãn đồng thời `fold_a != k` **VÀ** `fold_b != k` mới được đưa vào tập Train (đảm bảo **0% rò rỉ dữ liệu** giữa các Fold).

### 1.3. Khử nghiêng Nội chữ bằng Nội suy Bicubic & Khôi phục Bão hòa Lõi (`Core Saturation Restoration`)
* **Vấn đề của nội suy Bilinear:** Khi dùng `F.grid_sample(..., mode='bilinear')` trên ảnh nhỏ `28×28`, phép trung bình 4 điểm ảnh lân cận làm nhòe phần lõi nét bút trắng `255` thành màu xám, khiến tỷ lệ bão hòa `sat` tụt từ `38.3%` xuống chỉ còn `15.6%` (lệch miền nghiêm trọng so với `test.csv`).
* **Giải pháp:** Ước lượng mô-men nghiêng nội tại của từng chữ số $\text{true\_skew} = \frac{\mu_{11}^{(\text{left})} + \mu_{11}^{(\text{right})}}{\mu_{02}^{(\text{left})} + \mu_{02}^{(\text{right})}}$ (không bị đánh lừa khi hai chữ số đặt lệch độ cao), xoay thẳng nét chữ bằng nội suy bậc ba **`mode='bicubic'`**, sau đó **khôi phục độ bão hòa lõi** qua phép chuẩn hóa `x = clamp(x / 0.91, 0.0, 1.0)` và lượng tử hóa về đúng 80 mức xám `V3_LEVELS` (đưa `sat` trở về `~39.8%`, khớp hoàn hảo với `test.csv`).

---

## 🧠 2. Kiến trúc Hệ thống Đa-MLP Phân cấp (`Hierarchical Multi-MLP`)

Một mạng MLP phẳng truyền thống (`nn.Linear(784, 512)`) nối cứng từng tọa độ điểm ảnh với trọng số cố định: chỉ cần chữ số hàng đơn vị dịch sang trái/phải 2 pixel hoặc viết hẹp lại (`squeeze_x`), ma trận trọng số đầu vào bị lệch pha hoàn toàn. 

Để tạo ra **tính bất biến dịch chuyển và tỷ lệ cục bộ (Local Translation & Scale Invariance)** tương đương mạng CNN nhưng **chỉ sử dụng 100% các lớp `nn.Linear`**, chúng ta xây dựng kiến trúc **`V4Ultimate20x20HierMLP`** gồm **6 mạng MLP chuyên biệt** phối hợp trong một đồ thị tính toán duy nhất:

```text
                       [Ảnh Gốc 28x28] & [Ảnh Bicubic Smooth-Deskew 28x28]
                                                │
        ┌───────────────────┬───────────────────┼───────────────────┬───────────────────┐
        ▼                   ▼                   ▼                   ▼                   ▼
 [1. Coarse Patch]   [2. Fine Patch]    [3. Dual Holistic]  [4. Left-Digit]    [5. Right-Digit]
 16 ô cửa sổ 10x10   36 ô cửa sổ 6x6     Gộp Gốc + Deskew   Voronoi Crop 20x20 Voronoi Crop 20x20
 (F.unfold, s=6)     (F.unfold, s=4)      (784 + 784)        + 2D-FFT (436d)    + 2D-FFT (436d)
        │                   │                   │                   │                   │
 nn.Linear(100,192)  nn.Linear(36,128)   MLP(1568→512→256)   MLP(436→256→128)   MLP(436→384→256)
 + 2x Mixer Blocks   + 2x Mixer Blocks          │                   │                   │
        │                   │                   ▼                   ▼                   ▼
        └─────────┬─────────┘             Head Holistic       Head Tens (2)      Head Units (10)
                  ▼                        (11 classes)       ('1' vs '2')        ('0' .. '9')
           Head Patch-Mixer                     │                   └─────────┬─────────┘
             (11 classes)                       │                             ▼
                  │                             │                  Factorized Joint Prob
                  │                             │               P(d1, d0) = P_tens × P_units
                  └──────────────┬──────────────┴──────────────┬──────────────┘
                                 ▼                             ▼
                    [6. Coordinator Fusion MLP]       [Valley-Adaptive Gate]
                     (1286 features → 512 → 11)      Hợp nhất theo độ sâu khe cắt
