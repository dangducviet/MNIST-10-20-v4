# ✍️ MNIST 10-20 Data Collection Web App

Thư mục này chứa mã nguồn của công cụ thu thập dữ liệu chữ số viết tay qua trình duyệt. Đây là "chìa khóa" định hình cấu trúc dữ liệu của toàn bộ dự án, cho phép vừa chuẩn hóa ảnh xám 28x28 (giống định dạng MNIST) vừa lưu trữ chính xác tọa độ vector (`strokes`) của từng nét vẽ.

## 📊 Lịch sử Thu thập Dữ liệu

Quá trình xây dựng bộ dataset được chia làm 2 giai đoạn với tổng cộng 5.921 mẫu định danh hợp lệ:

* **Giai đoạn 1 (5.630 mẫu):** Thu thập thông qua phiên bản đầu tiên sử dụng file `Index.html`[cite: 11] và script máy chủ `Code.gs`[cite: 10]. 
* **Giai đoạn 2 (291 mẫu):** Bổ sung thêm dữ liệu để làm phong phú tập huấn luyện thông qua phiên bản nâng cấp v3.1 sử dụng file `Index_v3_1.html`[cite: 13] và script máy chủ tương ứng[cite: 12]. Phiên bản này được tối ưu hóa giao diện (hiển thị tiến độ 33 mẫu) và tăng cường kiểm tra tính hợp lệ của dữ liệu trước khi gửi[cite: 12, 13].

## 🛠️ Đặc tả Kỹ thuật Cốt lõi

Thay vì chỉ lưu ảnh tĩnh, Web App này thực hiện các bước tiền xử lý ngay trên máy khách (Client-side) trước khi gửi về Google Sheets:

1. **Ghi nhận Vector Nét vẽ (Stroke Logging):** Lưu tọa độ $(x, y)$ và độ rộng nét vẽ $w$ (thay đổi ngẫu nhiên từ 16 đến 52 pixel tùy phiên bản) trên Canvas có kích thước chuẩn $600 \times 400$[cite: 11, 13]. Dữ liệu này cho phép khôi phục lại các nét vẽ bị tràn viền (clipping) ở khâu huấn luyện mô hình.
2. **Chuẩn hóa MNIST (Bounding Box & Center of Mass):** Tự động tìm khung bao (bounding box) của chữ số, tính toán tỷ lệ thu nhỏ để vừa vặn trong khung $20 \times 20$, và đặt trọng tâm khối mực (Center of Mass) chính xác vào tọa độ $(14, 14)$ của lưới $28 \times 28$[cite: 11, 13].
3. **Lấy mẫu Trọng số Diện tích (Exact Area-Weights Resampling):** Khử răng cưa bằng thuật toán `resizeArea` tính toán chính xác tỷ lệ diện tích đè lên nhau giữa lưới gốc và lưới đích, tạo ra ảnh mượt mà với 80 mức xám tự nhiên[cite: 11, 13].

## 🚀 Hướng dẫn Triển khai (Deployment)

1. Tạo một bảng tính Google Sheets mới.
2. Truy cập **Tiện ích mở rộng (Extensions) > Apps Script**.
3. Dán nội dung file `Code.gs` vào trình soạn thảo[cite: 10, 12].
4. Tạo một file HTML mới tên là `Index.html` và dán mã nguồn giao diện vào[cite: 10, 12].
5. Chọn **Deploy > New deployment**, chọn loại **Web app**, cấp quyền truy cập cho "Anyone" và lấy URL để bắt đầu thu thập.