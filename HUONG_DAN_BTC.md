# MNIST 10–20 — Bộ kiểm tra kết quả

## Chạy trên Windows

1. Giải nén **nội dung** gói này vào thư mục dự án, ví dụ `D:\GitHub\MNIST-10-20-v4`. Cho phép thay `run_verification.py`. File `KIEM_TRA_BTC.cmd`, `run_verification.py` và thư mục `MNIST_data` phải nằm cùng cấp. Các cell đi kèm nằm trong `src`.
2. Bấm đúp **KIEM_TRA_BTC.cmd** rồi chọn: **1 = fast**, **2 = weights**, **3 = kiểm tra đầu vào train**, **4 = train**, **5 = predict**, **0 = thoát**. Để tạo nhãn trực tiếp từ mô hình, chọn **5**. Máy cần có Python 3.10 trở lên; bộ khởi chạy dùng môi trường riêng đã hoạt động, hoặc lệnh `python` sẵn có, rồi mới thử `py -3`. Không yêu cầu riêng Python 3.12.
3. Nếu thiếu thư viện, chương trình tự tạo `.venv-verification` và cài đúng gói cần dùng. Lần cài đầu cần Internet. Nếu môi trường hiện tại đã đủ thư viện, chương trình sử dụng luôn.
4. Báo cáo HTML tự mở khi xong. Các CSV, nhật ký và JSON của mỗi lần chạy nằm trong thư mục riêng bên trong `verified_submissions`. Báo cáo mới nhất nằm tại `verified_submissions/verification_report.html`.

Không cần mở PowerShell hoặc tự nhập lệnh cài thư viện. Không cần quyền quản trị hay kích hoạt môi trường PowerShell. Thời gian cài đặt lần đầu không được tính là thời gian suy luận mô hình.

## Để BTC xác minh nhãn do mô hình quyết định: mục 5 — PREDICT

Chế độ `predict` tính lại xác suất từ 28 trọng số v4 và gán nhãn cho từng ảnh bằng **`10 + argmax(p10, ..., p20)`**. Cùng xác suất lớn nhất thì chọn nhãn nhỏ nhất. Đầu vào của mô hình chỉ gồm pixel ảnh; `id` chỉ được sao chép sang đầu ra. Các cột khác, kể cả `label` nếu có, bị bỏ qua và được liệt kê trong JSON. Không áp quota 50 ảnh/lớp.

Bấm đúp rồi chọn **5** để chạy `MNIST_data/test.csv`. BTC có thể tự cung cấp CSV khác, số ảnh bất kỳ lớn hơn 0, bằng lệnh:

```powershell
python run_verification.py --mode predict --test-csv "D:\BTC\test_bi_mat.csv" --open-report
```

CSV cần `id` không rỗng và không trùng, cùng 784 cột `pixel0` đến `pixel783`; mỗi ảnh là 28×28 pixel, giá trị nguyên 0..255 theo đúng quy ước ảnh của dự án. ID dạng chuỗi như `00147` hoặc `NA` được giữ nguyên. Không yêu cầu bộ test mới có đủ 11 lớp hoặc đúng 550 ảnh.

Chế độ này chỉ cần bộ chạy, các nguồn `src/cell_01.py`, `cell_02.py`, `cell_04.py`, `cell_05.py`, `cell_06.py`, 25 trọng số trong `v4_hier_multi_mlp/fold_0..4`, ba trọng số full-data và cấu hình `v5_76ddef9bc184/renderer_comparison_b409f5a57688/config.json` để lấy `renderer.gray_levels`. Không đọc các `.npz` cache dự đoán, nhánh v12, CSV submission tham chiếu hoặc dữ liệu train. Mã train và hậu xử lý trong các cell không được gọi khi suy luận. Có thể kiểm tra riêng đầu vào bằng `--mode predict --check-only`.

Công thức ensemble cố định lấy từ **`te_hier28` của Cell 6, trước cổng v12**. Mỗi nhánh `*_ens` là trung bình xác suất 5 fold; ba nhánh full-data dùng ba checkpoint tương ứng. Tích lũy tổng có trọng số bằng float64:

| Nhánh | Trọng số |
| --- | ---: |
| `te_t6_ens` | 0.224 |
| `te_t6_full` | 0.112 |
| `te_dp5_ens` | 0.184 |
| `te_dp5_full` | 0.088 |
| `te_u20_ens` | 0.128 |
| `te_u20_full` | 0.064 |
| `te_s29_ens` | 0.100 |
| `te_s7_ens` | 0.100 |

Trong thư mục lần chạy có:

- `submission_model_only.csv`: ID và nhãn từ argmax.
- `probabilities_model_only.csv`: ID và 11 xác suất ensemble `p10..p20`. BTC có thể tự tính argmax từng dòng để đối chiếu CSV nhãn.
- `prediction_manifest.json`: công thức, môi trường, SHA256 của mã chạy, năm nguồn mô hình, cấu hình, test, 28 trọng số và hai CSV đầu ra. Chương trình so hash đầu vào trước/sau suy luận và kiểm tra lại argmax từ chính các CSV đã ghi.
- Nhật ký và báo cáo HTML/JSON như các chế độ khác. **PREDICTED** nghĩa là đã hoàn tất suy luận và kiểm tra argmax; chưa phải độ chính xác test.

### Quy trình BTC tự xác minh

1. Chốt commit Git và bộ trọng số trước khi BTC giao test giữ kín. BTC giữ bản sao hoặc SHA256, rà soát đường chạy `predict`, rồi tự chạy trong bản sao repository đó.
2. BTC chạy test do họ chọn; có thể đổi toàn bộ ID, đảo thứ tự ảnh, hoặc chọn một tập con. Đối chiếu theo cùng ảnh: nhãn kỳ vọng giữ nguyên. Xác suất có thể chênh nhẹ do phép tính số trên CPU/GPU hoặc kích thước batch; nếu nhãn gần hòa, cần ghi nhận chênh lệch thay vì cam kết luôn giống tuyệt đối.
3. BTC kiểm tra `label = 10 + argmax` từ CSV xác suất và giữ hai CSV, mã, trọng số, test và báo cáo của lần chạy. Các hash giúp xác định bộ file; báo cáo do người nộp tự tạo không thay thế việc BTC tự chạy.
4. Nếu cần xác minh huấn luyện, cung cấp thêm dữ liệu có nguồn gốc, cấu hình sinh dữ liệu/chia fold, seed, môi trường và lệnh train. Luồng `predict` chưa chứng minh checkpoint không học thuộc hoặc quá trình train không dùng nhãn test.

**Giới hạn với submission cũ:** Cell 6 có gán theo vị trí dòng, Cell 7 và 9 có gán theo ID. Vì vậy, không thể dùng báo cáo fast/weights hoặc kết quả predict mới để tuyên bố các nhãn lịch sử hoàn toàn không có can thiệp theo mẫu. `submission_model_only.csv` là kết quả suy luận riêng và có thể khác năm CSV lịch sử. Nếu giữ các bản cũ khi nộp, cần mô tả rõ hậu xử lý đó cho BTC; không thay chúng bằng kết quả mới mà vẫn giữ nguyên tuyên bố điểm cũ.

## Chế độ fast (mục 1)

- Tính lại chỉ số OOF/holdout Raw và Capped_Style từ các cache dự đoán 5 fold.
- Chỉ tái lập và đối chiếu 5 submission chính từ Cell 4, Cell 5, Cell 6, Cell 7 và Cell 9.
- Đối chiếu ID, nhãn, thứ tự ID và SHA256 với CSV tham chiếu. Ghi riêng dòng lệch nhãn vào CSV khi có.
- Chạy trên CPU với NumPy và pandas; không cần PyTorch, CUDA, Google Drive hay Colab.
- Không import hay khởi chạy huấn luyện các cell khi chạy fast.

**PASS**: cả 5 CSV chính khớp ID, nhãn và thứ tự với bản tham chiếu. Khác cách xuống dòng có thể làm SHA256 khác; báo cáo hiển thị việc này riêng.

**FAIL**: có kết quả hoặc thứ tự ID khác tham chiếu; trong chế độ weights, cũng có thể do xác suất lệch cache quá dung sai.

**INCOMPLETE**: đã tạo kết quả nhưng thiếu ít nhất một CSV tham chiếu của 5 bản chính. Chương trình không tự coi CSV vừa tạo là tham chiếu để cho PASS.

**ERROR**: thiếu file, còn con trỏ Git LFS, cache sai cấu trúc, không cài được thư viện hoặc có lỗi chạy.

## Dữ liệu lịch sử cho fast/weights

Giữ nguyên bộ dữ liệu và cache lịch sử của dự án:

- `MNIST_data/test.csv` (550 mẫu, cột `id`).
- `MNIST_data/ve_tay_10_20 - data.csv` và `ve_tay_10_20 - data_v31.csv`.
- `MNIST_data/v5_76ddef9bc184/dataset_82cc33f4f691/validation.npz`.
- Trong `MNIST_data/v4_hier_multi_mlp/fold_0` đến `fold_4`: `hier_preds_s7.npz`, `mscale_preds_s29.npz`, `ult20_preds_s2026.npz`, `dp20_hardft_preds_s2027.npz`, `topo20_swa_preds_s2028.npz`.
- `test_550_cell6_all_pillars_probs.npz` tại `MNIST_data/v4_hier_multi_mlp` hoặc trong `cell6_sessions/<session>/`. Nếu nằm trong session, chương trình chọn session hoàn tất mới nhất theo tên, và ghi rõ lựa chọn vào báo cáo; session không có status vẫn được xét.
- Các submission tham chiếu trong `MNIST_data/v4_hier_multi_mlp`, hoặc ngay thư mục gốc dự án đối với file xuất ra từ Cell 7 và 9. Năm bản chính cần có: `submission_v4_cell4_ultimate_master.csv`, `submission_v4_cell5_surgical_master.csv`, `submission_v4_cell6_topological_swa_master.csv`, `submission_v4_cell6_2hop_chain_19_13_18.csv`, `submission_v4_cell6_final_single_best.csv`.

Nếu clone repository có Git LFS, tải đủ dữ liệu thật bằng `git lfs pull` trước khi chuyển cho BTC. Không chỉ gửi riêng gói mã này; gói không chứa dữ liệu hay trọng số.

Nếu thiếu file, báo cáo sẽ liệt kê đường dẫn cần bổ sung. Không xóa những cache trên trong khi còn dùng chế độ fast.

## Phạm vi kết quả

Fast/weights là **kiểm tra tính tái lập của thí nghiệm lịch sử**. Hậu xử lý giữ đúng các lựa chọn trong nguồn gốc: Cell 6 gán nhãn tại vị trí dòng 147 và bỏ qua một số dòng khi quét; Cell 7–9 chỉnh các ID cụ thể. Không thay các lựa chọn này bằng những ngưỡng xác suất mới rồi tuyên bố tái lập.

Ở fast/weights, chương trình kiểm tra thứ tự ID giữa `test.csv` và bundle cache trước khi áp hậu xử lý; hai chế độ này chỉ phục vụ 550 mẫu test lịch sử. Dùng `predict` để suy luận trên test mới của BTC.

Độ chính xác hiển thị lấy từ nhãn validation/holdout. Mức khớp CSV tham chiếu không phải độ chính xác trên test và không xác minh điểm Kaggle. Cache cũ không có ID từng mẫu cho mọi fold nên vẫn phải dùng đúng dữ liệu và thứ tự của thí nghiệm gốc; shape, tổng xác suất và ánh xạ writer/fold được kiểm tra nhưng không chứng minh toàn bộ quá trình train không rò rỉ.

## Các chế độ bổ sung

Nếu cần chạy lại 28 mô hình v4 từ trọng số:

```powershell
python run_verification.py --mode weights --open-report
```

Chế độ này tự bổ sung PyTorch, SciPy, Pillow, tqdm nếu thiếu; có thể chạy CPU hoặc CUDA khi PyTorch hiện có hỗ trợ GPU. Nhánh v12 và chỉ số validation **vẫn dùng cache**, vì các file đính kèm chưa cung cấp bộ suy luận v12 độc lập. Chương trình so xác suất vừa tính với cache ở dung sai `atol=rtol=1e-4`. Không cam kết thời gian 15 giây trên máy bất kỳ.

Chạy lại các cell gốc 1, 2, 3, 4, 5, 6, 7 và 9; bỏ qua Cell 8 (cần CUDA và toàn bộ dữ liệu trung gian mà chúng yêu cầu; có thể tái sử dụng checkpoint/cache):

```powershell
python run_verification.py --mode train
```

Đây chưa phải pipeline huấn luyện mới từ dữ liệu thô. Chế độ này giữ hành vi tạo checkpoint/cache/session của cell gốc và ghi dữ liệu huấn luyện vào `MNIST_data`. Mặc định bấm đúp không chạy train.

## Trình tự kiểm tra trước khi gửi BTC

1. Chọn **5 — PREDICT** để tạo nhãn từ mô hình bằng argmax. BTC nên chạy lại độc lập theo quy trình trên. Nếu muốn đối chiếu năm CSV lịch sử, chọn **2 — WEIGHTS**; chế độ này vẫn dùng cache v12/OOF và hậu xử lý lịch sử. Gửi `verified_submissions/verification_report.json` nếu có lỗi hoặc kết quả lệch.
2. Chọn **3 — CHECK** để kiểm tra toàn bộ file đầu vào bắt buộc của các cell train. Mục này chỉ dùng thư viện chuẩn Python, không cài thêm gói, không tải checkpoint vào mô hình và không chạy train. Các file còn thiếu, file rỗng hoặc con trỏ Git LFS được liệt kê trong HTML/JSON.
3. `INPUTS_READY` chỉ có nghĩa tìm đủ file đầu vào theo mã gốc; chưa xác nhận nội dung dataset/checkpoint, GPU CUDA hay toàn bộ pipeline chạy thành công. Khi đã đủ file và có PyTorch hỗ trợ CUDA, có thể chọn **4 — TRAIN**.

Lệnh tương đương mục 3:

```powershell
python run_verification.py --mode train --check-only --open-report
```

Đầu vào train được kiểm tra gồm: các cell 1–7 và 9; ba CSV; `train.npz`, `validation.npz`, `meta.npz` của v5; cấu hình renderer; `strokes_capped.npz`, `validation_capped.npz`; cache dữ liệu bổ sung 451 mẫu; cache OOF v5 và seed29; cấu hình baseline v5; 5 checkpoint FT v5 và checkpoint full-data v5; 5 cache dự đoán v12. Các đầu ra v4 sẽ do những cell trước tạo ra nên không bắt buộc phải có sẵn trước khi train.

Có thể kiểm tra file đầu vào của chế độ weights bằng `--mode weights --check-only`. Chế độ train thật cũng tự kiểm tra danh sách file trước khi cài thư viện hoặc bắt đầu huấn luyện. Khi chạy lại, các cell gốc có thể tái sử dụng checkpoint/cache v4 hiện có; đây chưa phải bằng chứng đã huấn luyện lại từ đầu.

Chọn thư mục dữ liệu hoặc cache cụ thể, hoặc cấm tự cài trên máy offline:

```powershell
python run_verification.py --data-root "D:\du_lieu\MNIST_data" --cache "D:\du_lieu\bundle.npz" --no-install
```

Thư mục `.venv-verification` và `verified_submissions` là dữ liệu phát sinh; có thể thêm hai thư mục này vào `.gitignore`.

## Các lỗi đã sửa và kiểm thử

Bản cũ import NumPy/PyTorch ngay lúc khởi động, import cell làm chạy train, dùng đường dẫn cache cố định mặc dù Cell 6 xuất vào session, và luôn in “100% IDENTICAL” ngay cả khi có dòng lệch. Phần hậu xử lý cũ khác Cell 6–9 gốc. Bản này xử lý các vấn đề đó và trả mã thoát khác 0 nếu chưa xác nhận được kết quả.

Các kiểm thử trong quá trình sửa dùng dữ liệu tổng hợp để kiểm tra trọn luồng fast, nhãn/ID sai, thiếu file/tham chiếu, Git LFS và báo cáo. Luồng predict được kiểm tra với forward pass giả lập để kiểm tra tải đủ 28 checkpoint, không truy cập cache/hậu xử lý, độc lập với ID/cột label, đảo thứ tự và tách tập con; kiểm tra công thức với Cell 6 gốc, argmax từ CSV và hash. Các thử nghiệm này kiểm tra cách nối luồng, chưa xác nhận tính bất biến hoặc chất lượng của mô hình thật.

Ảnh kết quả người dùng cung cấp ngày 19/10/2026 cho thấy fast đã chạy trên Windows và cả 5 submission lịch sử khớp 550/550 dòng. Báo cáo weights người dùng cung cấp cũng cho thấy 5 CSV khớp, nhưng một số xác suất v4 lệch cache quá dung sai nên trạng thái chung FAIL. Chưa chạy bộ dữ liệu/checkpoint thật hoặc luồng predict đầy đủ trong môi trường sửa mã; cần chạy trên máy chứa trọng số trước khi gửi BTC. Chưa kiểm thử menu mới bằng cmd.exe trong môi trường này.

Tài liệu cơ chế môi trường và cài gói: [Python venv](https://docs.python.org/3/library/venv.html), [pip install](https://pip.pypa.io/en/stable/cli/pip_install/).
