# MNIST 10-20: mô hình v4

Gói bàn giao theo notebook `v4_MNIST_10_20.ipynb` gồm 6 ô code đã cung cấp.
Notebook gốc được giữ nguyên cả code và output; `src/cell_01.py` đến
`src/cell_06.py` là bản trích code để đọc và xem thay đổi trên GitHub.

## Phạm vi

- `MNIST_data/v4_hier_multi_mlp/`: 22 checkpoint v4 (4 loại x 5 fold và 2 full-data),
  dự đoán, dữ liệu sinh/cache, submission và các log/report còn có trên Drive.
- Ba CSV đầu vào và đúng những file dưới `v5_76ddef9bc184` mà notebook đọc.
- Không gom cả MNIST_data, không gom v4_fa7d0ff87900, các phiên bản cũ khác
  hoặc toàn bộ v5. Dữ liệu sinh trong RAM nếu không được lưu không nằm trong gói.
- `manifest.json`: đường dẫn, mục đích, dung lượng và SHA256 của dữ liệu đã gom.

## Chạy lại trên Colab

1. Clone repo và lấy đầy đủ các file Git LFS, hoặc dùng một bản xuất đầy đủ dữ liệu.
2. Đặt thư mục `MNIST_data` của gói vào `MyDrive/MNIST_data` của tài khoản chạy.
   Nếu đã có thư mục cùng tên, dùng vị trí riêng rồi sửa TẤT CẢ các đường dẫn
   `/content/drive/MyDrive/MNIST_data` trong notebook cho khớp, kể cả ở ô 3 và ô 6.
3. Mở notebook gốc trên Colab, chọn GPU T4 khi huấn luyện, mount đúng Drive.
4. Chạy theo thứ tự ô 1, 2, 3, 4, 5, 6. Có checkpoint/cache nên một số bước sẽ
   tái sử dụng kết quả. Nếu cần huấn luyện lại từ đầu, làm trên một bản sao riêng.

`requirements.txt` liệt kê thư viện được import. `export_environment.json`
chỉ mô tả môi trường lúc đóng gói, không khẳng định phiên bản lúc train.

## Các ô code

| Ô | Vai trò |
|---|---|
| 1 | HierMultiMLP seed 7, 5 fold; dữ liệu và đối chứng |
| 2 | MultiScaleHierMLP seed 29, 5 fold; phối hợp dự đoán |
| 3 | Confidence-advantage gating và submission |
| 4 | Ultimate20x20 seed 2026, 5 fold và full-data |
| 5 | DualPeak HardFT seed 2027, 5 fold và full-data; các submission |
| 6 | Hậu xử lý: đặt nhãn dòng id=147 thành 11 |

## Submission của ô cuối

`submission_v4_cell5_dual20x20_surgical.csv` được tạo lại trong bản sao
từ `submission_v4_cell5_surgical_master.csv` bằng đúng thao tác ở ô 6:
đặt label của id=147 thành 11. Đây là hậu xử lý trong notebook;
file đóng gói không phải một lần dự đoán hoặc huấn luyện mới.
Ô 6 gốc ghi file ở thư mục làm việc của Colab và tải xuống trình duyệt,
không tự lưu file này vào thư mục MNIST_data trên Drive.

## Đưa lên GitHub

Repo dùng Git LFS cho npz, npy, pt, pth, zip. Cần cài và bật Git LFS
TRƯỚC lần commit dữ liệu đầu tiên. Giải nén gói rồi đưa các file bên trong
thư mục MNIST-10-20-v4 lên repo; không commit nguyên file ZIP bàn giao.
Team cần chạy `git lfs pull` sau khi clone nếu file LFS chưa tải đầy đủ.
