# 🔢 MNIST 10–20 Challenge: Hệ thống Đa-MLP Phân cấp & Đặc trưng Tô-pô (Hierarchical Topological Multi-MLP)

Repository này lưu trữ toàn bộ mã nguồn, quy trình tái tạo dữ liệu nét vẽ (`Vector Stroke Rendering`), trọng số của 28 mô hình thành phần và kết quả kiểm định của hệ thống nhận diện chữ số viết tay ghép đôi (11 lớp: từ `10` đến `20`) trên ảnh xám $28 \times 28$.

Toàn bộ kiến trúc được xây dựng dưới ràng buộc nghiêm ngặt của đề bài: **100% Pure MLP** — chỉ sử dụng các tầng tuyến tính `nn.Linear`, chuẩn hóa `nn.LayerNorm`, `nn.BatchNorm1d`, hàm kích hoạt `nn.GELU`, `nn.Dropout` và các phép biến đổi hình học/cắt cửa sổ không tham số (`F.unfold`, `F.grid_sample`, `torch.fft.rfft2`, `torch.cummax`). **Tuyệt đối không sử dụng `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d` dưới bất kỳ hình thức nào.**

---

## 📈 1. Lịch sử Tiến triển Hiệu năng (Performance Progression)

Hệ thống được kiểm định qua **5-Fold GroupKFold theo người viết (`writer overlap = 0` tuyệt đối)** trên `5.383` mẫu định danh của `109` người viết (chia thành 2 miền đánh giá: miền gốc `Raw` và miền khớp phong cách tập Test `Capped_Style`), đồng thời đối chứng trực tiếp trên Kaggle Public Leaderboard (`220` mẫu = 40% của tập Test `550` mẫu):

| Thế hệ | Kiến trúc & Kỹ thuật Cốt lõi | OOF `Capped_Style` (4.932 mẫu) | OOF `Raw` (4.932 mẫu) | Held-out New Writers (451 mẫu `Raw` / `Capped`) | Toàn bộ 109 Writers (5.383 mẫu `Raw`) | Kaggle Public LB (220 mẫu) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline `v5`** | Flat MLP (`784→512→512→256→11`) + Shift-9 TTA | `96.07%` (194 lỗi) | `96.88%` (154 lỗi) | `97.34%` / `96.67%` (12 / 15 lỗi) | `96.92%` (166 lỗi) | `0.95000` (209/220) |
| **`v7` Sinkhorn** | Flat MLP Ensemble + Cân bằng Hungarian cứng | `96.69%` (163 lỗi) | `96.95%` (150 lỗi) | `98.23%` / `98.00%` (8 / 9 lỗi) | `97.06%` (158 lỗi) | `0.95454` (210/220) |
| **`v12` Fourier** | Flat MLP + Bicubic Smooth-Deskew + 2D-Fourier | `96.88%` (154 lỗi) | `97.00%` (148 lỗi) | `98.00%` / `97.78%` (9 / 10 lỗi) | `97.08%` (157 lỗi) | `0.95000` (209/220) |
| **`v4` Cell 1** | **HierMultiMLP** (Seed 7, huấn luyện từ đầu, Crop $18\times 18$) | `97.67%` (115 lỗi) | `97.83%` (107 lỗi) | `98.45%` / `98.45%` (7 / 7 lỗi) | `97.88%` (114 lỗi) | — |
| **`v4` Cell 2–3** | **MultiScaleHierMLP** (Seed 29, $10\times 10$ & $6\times 6$ Mixers + Cổng tự tin) | `97.73%` (112 lỗi) | `97.91%` (103 lỗi) | `98.67%` / `98.45%` (6 / 7 lỗi) | `97.92%` (112 lỗi) | `0.96818` (213/220) |
| **`v4` Cell 4** | **Ultimate20x20HierMLP** (Seed 2026, Voronoi $20\times 20$ + 409 mẫu vô danh + Full-Data) | `97.71%` (113 lỗi) | `97.95%` (101 lỗi) | `99.11%` / `98.45%` (4 / 7 lỗi) | `98.01%` (107 lỗi) | `0.97272` (214/220) |
| **`v4` Cell 5** | **DualPeak-HardFT** (Seed 2027, Ràng buộc khối lượng cắt `[8..15]` + Focal Loss) | **`97.83%` (107 lỗi)** | `97.99%` (99 lỗi) | `98.67%` / `98.45%` (6 / 7 lỗi) | `98.01%` (107 lỗi) | `0.97727` (215/220) |
| **`v4` Cell 6** | **TopoSubMixer-SWA** (Seed 2028, Sub-Digit $5\times 5$ Mixer + Tô-pô 1-px + 4 Cluster Heads + SWA) | **`97.83%` (107 lỗi)** | **`98.07%` (95 lỗi)** | **`99.33%` / `98.67%` (3 / 6 lỗi)** | **`98.12%` (101 lỗi)** | **`0.98181` (216/220)** |

---

## 🔬 2. Giải phẫu Bản chất Vật lý của Tập Test & Quy trình Tạo Dữ liệu Huấn luyện

### 2.1. Đặc trưng Vật lý của Tập `test.csv` (`550` mẫu)
Phân tích định lượng trên 550 ảnh của `test.csv` cho thấy dữ liệu kiểm thử có cấu trúc rất đặc thù:
1. **Phân bố lớp cân bằng tuyệt đối:** Gồm đúng `550` mẫu chia đều cho 11 lớp từ `10` đến `20` (**đúng `50` mẫu/lớp**). Tập Public Leaderboard chiếm 40% (`220` mẫu lấy ngẫu nhiên) và tập Private Leaderboard chiếm 60% (`330` mẫu).
2. **Thông số hình thái & bức xạ học:**
   * Tỷ lệ điểm ảnh tiền cảnh (`pixel > 20`): $\text{fg} = 0.2044$.
   * Tỷ lệ điểm ảnh bão hòa mức trắng tuyệt đối (`pixel == 255` trên vùng tiền cảnh): $\text{sat} = 0.3834$.
   * Kích thước khung bao trung vị của chữ số: chiều cao $h_{\text{med}} = 19.0\text{ px}$, chiều rộng $w_{\text{med}} = 19.0\text{ px}$, đặt chính giữa tâm khối $(14, 14)$ trên nền đen $28 \times 28$ (luôn có viền đen $4\text{ px}$ xung quanh).
   * **Lượng tử hóa mức xám:** Toàn bộ các điểm ảnh trong `test.csv` không nhận đủ 256 giá trị nguyên mà chỉ rơi vào đúng **80 mức xám rời rạc (`V3_LEVELS`)** sinh ra từ thuật toán hạ mẫu diện tích của canvas trình duyệt.

### 2.2. Làm sạch Nhãn & Khôi phục Nét vẽ Tràn khung HTML (`Dynamic BBox Unclipping`)
Dữ liệu gốc của dự án được thu thập từ hai nguồn: `ve_tay_10_20 - data.csv` (`5.630` dòng) và `ve_tay_10_20 - data_v31.csv` (`291` dòng mới), trong đó mỗi dòng lưu cả mảng 784 pixel xuất từ trình duyệt lẫn chuỗi JSON tọa độ nét vẽ gốc (`strokes`).
* **Phát hiện lỗi cắt cụt viền (`Canvas Clipping`):** Trong `data_v31.csv`, có tới **`59 / 291` mẫu (`20.3%`)** bị người viết vẽ vượt ra ngoài khung canvas cố định `600×400` của giao diện web (tọa độ $x < 0$, $y < 0$, $x > 600$ hoặc $y > 400$), khiến ảnh xuất tự động bị mất đầu hoặc mất đuôi chữ số.
* **Thuật toán Khôi phục (`render_unclipped_raw`):** Hệ thống đọc trực tiếp tọa độ vector $(x, y)$ và độ rộng nét $w$ từ cột `strokes`, tính khung bao động thực tế $[\min(p - w/2) - 3, \max(p + w/2) + 4]$, vẽ lại trên mặt phẳng siêu lấy mẫu $2\times$ (`PIL.ImageDraw` với nối cong `joint='curve'` và chấm tròn đầu mút), rồi thu nhỏ bằng ma trận trọng số diện tích chính xác (`Exact Area-Weights Resampling`) về khung $20 \times 20$ đặt tại tâm khối $(14, 14)$.
* **Loại bỏ mẫu hỏng/nhiễu nhãn:**
  * Loại bỏ `4` dòng lỗi nhãn trong tập cũ: `V4_V5_EXCLUSIONS = [458, 1767, 2213, 4355]`.
  * Loại bỏ `18` dòng biến dạng/nhầm nhãn trong `data_v31.csv`: `V4_V31_EXCLUSIONS = [9, 15, 16, 25, 29, 78, 95, 222, 224, 231, 233, 234, 235, 238, 244, 251, 253, 256]`.
  * Chuẩn hóa tên người viết Unicode NFC (gộp các cách ghi ngược họ tên như `'phạm minh nhật'` $\to$ `'minh nhật phạm'`).
  * Khai thác bổ sung **`409` mẫu viết tay thật bị khuyết tên người viết** (`Anonymous Clean Samples` từ các dòng `0..5451` của `data.csv`, chỉ giữ các mẫu có xác suất đồng thuận $\ge 0.80$).
* **Tổng số mẫu viết tay gốc sạch:** `4.932` (94 người viết cũ) + `451` (15 người viết mới) + `409` (mẫu vô danh sạch) = **`5.792` mẫu gốc thực tế**.

### 2.3. Sinh Biến thể Vector & Ghép chéo Chữ số Không Rò rỉ (`Zero-Leakage Vector Synthesis`)
Từ `5.792` mẫu vector gốc, hệ thống sinh ra **`221.198` mẫu huấn luyện** thuộc 4 nguồn (`src_exp ∈ {0, 1, 2, 3}`), toàn bộ đều được ép về đúng 80 mức xám `V3_LEVELS`:
1. **Nguồn `0` — `Clean Raw` (`5.792` ảnh):** Ảnh gốc đã chuẩn hóa tâm khối và khôi phục tràn viền.
2. **Nguồn `1` — `Capped Strokes` (`~23.000` ảnh — 4 biến thể / mẫu gốc):**
   * Biến đổi trực tiếp trên tọa độ nét vẽ JSON trước khi raster hóa: nghiêng chữ `shear ∈ [-0.25, +0.25]`, co giãn ngang `squeeze_x ∈ [0.72, 1.02]`, độ dày nét `pen ∈ [25, 47]`.
   * **Trần bảo vệ lỗ rỗng (`Cap = 1.25`):** Giới hạn hệ số phình nét `pen_scale <= 1.25` để ngăn mực làm đặc kín lỗ rỗng của các chữ số `0, 6, 8, 9`.
3. **Nguồn `2` — `Cross-Writer Recombination` (`Recomb` — `~42.400` ảnh):**
   * Phân tích khoảng hở giữa các nét vẽ để tách riêng **Chữ số Hàng chục (`1` hoặc `2`)** và **Chữ số Hàng đơn vị (`0..9`)** từ các mẫu sạch.
   * Ghép chữ số Hàng chục của người viết $A$ (thuộc `fold_a`) với chữ số Hàng đơn vị của người viết $B$ (thuộc `fold_b`).
   * **Quy tắc Chống Rò rỉ 5-Fold Tuyệt đối:** Khi huấn luyện Fold $k \in \{0, 1, 2, 3, 4\}$, một mẫu ghép chỉ được phép vào tập Train nếu thỏa mãn đồng thời **`(fold_a != k) & (fold_b != k)`**.
4. **Nguồn `3` — `Synthetic Composite` (`150.000` ảnh):** Chỉ sử dụng ở Giai đoạn 1 (Pre-training) để làm giàu biên dạng chữ số cơ bản, và **bị loại bỏ hoàn toàn ở Giai đoạn 2 (Real-Domain Fine-tuning)** để tránh lệch miền.

---

## 📐 3. Tiền xử lý Hình học Thế hệ 6: Khử nghiêng Bảo toàn Bão hòa & Nhát cắt `Hybrid Dual-Peak Voronoi 20×20`

Một mạng MLP phẳng nối cứng 784 pixel đầu vào sẽ thất bại khi hai chữ số viết lệch cao độ, viết nghiêng, hoặc khi khoảng cách giữa hai chữ số thay đổi. Hàm `c6_extract_hybrid_dualpeak_20x20_gpu` giải quyết bài toán này ngay trên GPU (`16.4 giây` cho toàn bộ `221.198` ảnh) qua 4 bước toán học:

### Bước 1: Định vị Khe cắt bằng Ràng buộc Khối lượng Tích lũy & Đỉnh Kép (`Dual-Peak Cumulative-Mass Cut`)
* **Điểm mù của thuật toán cắt cột cực tiểu thông thường:** Nếu chỉ tìm cột có tổng mực nhỏ nhất trên đoạn cột `[9..17]`, khi chữ số `1` bên trái viết sát vào chữ số bên phải (ở cột `10–11`) và chữ số bên phải có lòng rỗng ở giữa (`0, 4, 6, 8, 9` tại cột `14–17`), hàm `argmin` sẽ **nhảy vào cắt ngay chính giữa bụng của chữ số hàng đơn vị**!
* **Giải pháp:** Trên đoạn cột $c \in [8..15]$, hệ thống tính **Tỷ lệ Khối lượng Mực Tích lũy từ bên trái**:
  $$\text{cum\_ratio}(c) = \frac{\sum_{x=0}^{c} \sum_{y=0}^{27} I(y, x)}{\sum_{x=0}^{27} \sum_{y=0}^{27} I(y, x)}$$
  Vì chữ số Hàng chục (`1` hoặc `2`) luôn chiếm từ $16\%$ đến $56\%$ tổng lượng mực của cả bức ảnh, mọi cột $c$ có $\text{cum\_ratio}(c) < 0.16$ hoặc $> 0.56$ đều bị phạt nặng:
  $$c^* = \operatorname*{argmin}_{c \in [8..15]} \Big( \text{col\_sum}(c) + P_{\text{pos}}(c) + P_{\text{mass}}\big(\text{cum\_ratio}(c)\big) \Big)$$
  Điều này chặn đứng hoàn toàn lỗi cắt nhầm vào lòng rỗng của chữ số bên phải.

### Bước 2: Khử nghiêng Nội chữ bằng Nội suy Bậc ba & Khôi phục Độ bão hòa Lõi (`Bicubic Core-Saturation Deskew`)
* Thay vì tính mô-men quán tính trên toàn ảnh $28 \times 28$ (sẽ bị sai lệch nếu hai chữ số viết cao thấp khác nhau), hệ thống tính mô-men nghiêng riêng quanh trọng tâm $(c_{x,L}, c_{y,L})$ của chữ số Trái và $(c_{x,R}, c_{y,R})$ của chữ số Phải:
  $$\text{true\_skew} = \operatorname{clamp}\!\left(\frac{\mu_{11}^{(L)} + \mu_{11}^{(R)}}{\mu_{02}^{(L)} + \mu_{02}^{(R)}},\; -0.32,\; +0.32\right)$$
* Áp dụng phép biến đổi Affine qua `F.grid_sample(..., mode='bicubic')` rồi khôi phục độ bão hòa lõi trắng bằng phép chia chuẩn hóa `xd = clamp(xd / 0.91, 0.0, 1.0)` trước khi lượng tử hóa về 80 mức xám `V3_LEVELS`.

### Bước 3: Lọc Mặt nạ Voronoi & Trích xuất Khung `20×20` với Neo Dọc Lai (`Hybrid Vertical Anchor`)
* **Mặt nạ Voronoi (`Centroid-Distance Voronoi Mask`):** Tính khoảng cách bình phương có trọng số từ mọi điểm ảnh $(x, y)$ tới tâm chữ số Trái và tâm chữ số Phải:
  $$d_L^2 = (x - c_{x,L})^2 + 0.35(y - c_{y,L})^2, \quad d_R^2 = (x - c_{x,R})^2 + 0.35(y - c_{y,R})^2$$
  Kết hợp với hàm Sigmoid quanh biên cắt $c^*$ để xóa sạch các nét móc thừa của chữ số `1` chờm sang khung của chữ số bên phải.
* **Neo Dọc Lai (`Hybrid Vertical Anchor`):** Nếu đặt tâm khung cắt $20 \times 20$ hoàn toàn theo trọng tâm dọc $c_{y,R}$, chữ số `7` (nặng mực ở đỉnh trên, $c_{y,R} \approx 11.0$) sẽ bị kéo tụt xuống ~2.5 pixel làm nét ngang đỉnh của số `7` rơi xuống vị trí nét ngang bụng của số `4`. Do đó, tâm ngang được căn 100% theo $c_{x,R}$, nhưng tâm dọc được neo giữ $40\%$ vào đường trung tuyến tự nhiên `13.5`:
  $$y_{\text{crop}} = 0.60 \cdot c_{y,R} + 0.40 \cdot 13.5$$

---

## 🧠 4. Kiến trúc `V4TopologicalSubMixerMLP` (4,263,701 tham số — 100% `nn.Linear`)

Mô hình Thế hệ 6 (`V4TopologicalSubMixerMLP`) kết hợp **7 mạng MLP chuyên gia** và **4 đầu phân xử cụm có điều kiện (`Conditional Cluster Specialist Heads`)**:

```text
                       [Ảnh Gốc xo (784)] & [Ảnh Bicubic Deskew xd (784)]
                                                │
      ┌──────────────────┬──────────────────────┼──────────────────────┬──────────────────────┐
      ▼                  ▼                      ▼                      ▼                      ▼
[1. Coarse Mixer]  [2. Fine Mixer]     [3. Dual Holistic]   [4. Left-Digit 20x20]  [5. Right-Digit 20x20]
16 ô cửa sổ 10x10  36 ô cửa sổ 6x6      Nối xo + xd (1568)   Crop 20x20 (400d)      Crop 20x20 (400d)
(F.unfold, s=6)    (F.unfold, s=4)      MLP(1568→512→256)    + TopoDesc (214d)      + TopoDesc (214d)
      │                  │                      │            MLP(614→256→128)       + [6. Sub-Digit 5x5 Mixer]
6-Zone Anatomical  6-Zone Anatomical            │                      │              (36 ô 5x5 → 5-Band Pool)
Pooling (1152→384) Pooling (768→256)            │                      │              + Geo8 (8d) → MLP(814→256)
      │                  │                      ▼                      ▼                      ▼
      └────────┬─────────┘                Head Holistic          Head Tens (2)          Head Units (10)
               ▼                           (11 classes)          ('1' vs '2')            ('0' .. '9')
        Head Patch-Mixer                        │                      └──────────┬───────────┘
          (11 classes)                          │                                 ▼
               │                                │                      Factorized Joint Prob p_fact
               └────────────────┬───────────────┴──────────────┬──────────────────┘
                                ▼                              ▼
                  [7. Coordinator Fusion MLP]       [8. 4x Cluster Specialist Heads]
                   (1480 features → 512 → 11)        • C1: {11, 14, 17}  • C2: {15, 16, 18}
                                │                    • C3: {13, 18, 19}  • C4: {10, 18, 19, 20}
                                └───────────────┬──────────────┘
                                                ▼
                          [Exact Conditional Bayes + Valley-Adaptive Gate]