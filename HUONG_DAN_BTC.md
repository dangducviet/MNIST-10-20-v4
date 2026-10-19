# MNIST 10–20 — Bộ kiểm tra kết quả

## Chạy trên Windows

1. Giải nén **nội dung** gói này vào thư mục dự án, ví dụ `D:\GitHub\MNIST-10-20-v4`. Cho phép thay `run_verification.py`. File `KIEM_TRA_BTC.cmd`, `run_verification.py` và thư mục `MNIST_data` phải nằm cùng cấp. Các cell đi kèm nằm trong `src`.
2. Bấm đúp **KIEM_TRA_BTC.cmd**. Máy cần có Python 3.10 trở lên; bộ khởi chạy ưu tiên Python 3.12 nếu đã cài.
3. Nếu thiếu thư viện, chương trình tự tạo `.venv-verification` và cài đúng gói cần dùng. Lần cài đầu cần Internet. Nếu môi trường hiện tại đã đủ thư viện, chương trình sử dụng luôn.
4. Báo cáo HTML tự mở khi xong. Các CSV, nhật ký và JSON của mỗi lần chạy nằm trong thư mục riêng bên trong `verified_submissions`. Báo cáo mới nhất nằm tại `verified_submissions/verification_report.html`.

Không cần mở PowerShell hoặc tự nhập lệnh cài thư viện. Không cần quyền quản trị hay kích hoạt môi trường PowerShell. Thời gian cài đặt lần đầu không được tính là thời gian suy luận mô hình.

## Chế độ mặc định: fast

- Tính lại chỉ số OOF/holdout Raw và Capped_Style từ các cache dự đoán 5 fold.
- Tái lập 7 submission tương ứng Cell 4, Cell 5, Cell 6 và các biến thể Cell 7–9.
- Đối chiếu ID, nhãn, thứ tự ID và SHA256 với CSV tham chiếu. Ghi riêng dòng lệch nhãn vào CSV khi có.
- Chạy trên CPU với NumPy và pandas; không cần PyTorch, CUDA, Google Drive hay Colab.
- Không import hay khởi chạy huấn luyện các cell khi chạy fast.

**PASS**: mọi CSV đều khớp ID, nhãn và thứ tự với bản tham chiếu. Khác cách xuống dòng có thể làm SHA256 khác; báo cáo hiển thị việc này riêng.

**FAIL**: có kết quả hoặc thứ tự ID khác tham chiếu; trong chế độ weights, cũng có thể do xác suất lệch cache quá dung sai.

**INCOMPLETE**: đã tạo kết quả nhưng thiếu ít nhất một CSV tham chiếu. Chương trình không tự coi CSV vừa tạo là tham chiếu để cho PASS.

**ERROR**: thiếu file, còn con trỏ Git LFS, cache sai cấu trúc, không cài được thư viện hoặc có lỗi chạy.

## Dữ liệu phải đi kèm repository

Giữ nguyên bộ dữ liệu và cache lịch sử của dự án:

- `MNIST_data/test.csv` (550 mẫu, cột `id`).
- `MNIST_data/ve_tay_10_20 - data.csv` và `ve_tay_10_20 - data_v31.csv`.
- `MNIST_data/v5_76ddef9bc184/dataset_82cc33f4f691/validation.npz`.
- Trong `MNIST_data/v4_hier_multi_mlp/fold_0` đến `fold_4`: `hier_preds_s7.npz`, `mscale_preds_s29.npz`, `ult20_preds_s2026.npz`, `dp20_hardft_preds_s2027.npz`, `topo20_swa_preds_s2028.npz`.
- `test_550_cell6_all_pillars_probs.npz` tại `MNIST_data/v4_hier_multi_mlp` hoặc trong `cell6_sessions/<session>/`. Nếu nằm trong session, chương trình chọn session hoàn tất mới nhất theo tên, và ghi rõ lựa chọn vào báo cáo; session không có status vẫn được xét.
- Các submission tham chiếu trong `MNIST_data/v4_hier_multi_mlp`, hoặc ngay thư mục gốc dự án đối với file xuất ra từ Cell 7–9.

Nếu clone repository có Git LFS, tải đủ dữ liệu thật bằng `git lfs pull` trước khi chuyển cho BTC. Không chỉ gửi riêng gói mã này; gói không chứa dữ liệu hay trọng số.

Nếu thiếu file, báo cáo sẽ liệt kê đường dẫn cần bổ sung. Không xóa những cache trên trong khi còn dùng chế độ fast.

## Phạm vi kết quả

Đây là **kiểm tra tính tái lập của thí nghiệm lịch sử**. Hậu xử lý giữ đúng các lựa chọn trong nguồn gốc: Cell 6 gán nhãn tại vị trí dòng 147 và bỏ qua một số dòng khi quét; Cell 7–9 chỉnh các ID cụ thể. Không thay các lựa chọn này bằng những ngưỡng xác suất mới rồi tuyên bố tái lập.

Chương trình kiểm tra thứ tự ID giữa `test.csv` và bundle cache trước khi áp hậu xử lý. Bộ này chỉ phục vụ 550 mẫu test lịch sử; không phải giao diện suy luận chung cho dữ liệu test mới của BTC.

Độ chính xác hiển thị lấy từ nhãn validation/holdout. Mức khớp CSV tham chiếu không phải độ chính xác trên test và không xác minh điểm Kaggle. Cache cũ không có ID từng mẫu cho mọi fold nên vẫn phải dùng đúng dữ liệu và thứ tự của thí nghiệm gốc; shape, tổng xác suất và ánh xạ writer/fold được kiểm tra nhưng không chứng minh toàn bộ quá trình train không rò rỉ.

## Các chế độ bổ sung

Nếu cần chạy lại 28 mô hình v4 từ trọng số:

```powershell
python run_verification.py --mode weights --open-report
```

Chế độ này tự bổ sung PyTorch, SciPy, Pillow, tqdm nếu thiếu; có thể chạy CPU hoặc CUDA khi PyTorch hiện có hỗ trợ GPU. Nhánh v12 và chỉ số validation **vẫn dùng cache**, vì các file đính kèm chưa cung cấp bộ suy luận v12 độc lập. Chương trình so xác suất vừa tính với cache ở dung sai `atol=rtol=1e-4`. Không cam kết thời gian 15 giây trên máy bất kỳ.

Chạy lại tuần tự các cell gốc (cần CUDA và toàn bộ dữ liệu trung gian mà chúng yêu cầu; có thể tái sử dụng checkpoint/cache):

```powershell
python run_verification.py --mode train
```

Đây chưa phải pipeline huấn luyện mới từ dữ liệu thô. Chế độ này giữ hành vi tạo checkpoint/cache/session của cell gốc và ghi dữ liệu huấn luyện vào `MNIST_data`. Mặc định bấm đúp không chạy train.

Chọn thư mục dữ liệu hoặc cache cụ thể, hoặc cấm tự cài trên máy offline:

```powershell
python run_verification.py --data-root "D:\du_lieu\MNIST_data" --cache "D:\du_lieu\bundle.npz" --no-install
```

Thư mục `.venv-verification` và `verified_submissions` là dữ liệu phát sinh; có thể thêm hai thư mục này vào `.gitignore`.

## Các lỗi đã sửa và kiểm thử

Bản cũ import NumPy/PyTorch ngay lúc khởi động, import cell làm chạy train, dùng đường dẫn cache cố định mặc dù Cell 6 xuất vào session, và luôn in “100% IDENTICAL” ngay cả khi có dòng lệch. Phần hậu xử lý cũ khác Cell 6–9 gốc. Bản này xử lý các vấn đề đó và trả mã thoát khác 0 nếu chưa xác nhận được kết quả.

Các kiểm thử đi kèm quá trình sửa dùng dữ liệu tổng hợp để kiểm tra trọn luồng fast, nhãn/ID sai, thiếu file/tham chiếu, Git LFS và báo cáo. **Chưa chạy trên dữ liệu/checkpoint thật của dự án**, vì các file đính kèm chỉ gồm mã Python; chưa kiểm tra bấm đúp trực tiếp trên Windows hoặc suy luận đầy đủ 28 checkpoint. Chạy `KIEM_TRA_BTC.cmd` trên bản repository đầy đủ để xác nhận cuối cùng.

Tài liệu cơ chế môi trường và cài gói: [Python venv](https://docs.python.org/3/library/venv.html), [pip install](https://pip.pypa.io/en/stable/cli/pip_install/).
