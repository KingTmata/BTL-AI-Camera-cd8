# Chuyển dự án và lịch sử Codex sang máy khác

Chuẩn bị ngày 17/10/2026. Các gói riêng tư nằm trong `transfer_backup/` ở thư mục gốc; thư mục này bị Git bỏ qua. Chép **toàn bộ thư mục này** sang ổ ngoài hoặc nơi lưu trữ riêng tư rồi sang máy mới. Bản sao còn nằm trên máy cũ chưa bảo vệ được khỏi hỏng ổ/mất máy.

Theo yêu cầu đóng gói cuối: bàn giao **một file** `BTL-AI-Camera-FULL-20261017.zip` nằm trực tiếp trong `C:\Users\aawm8\Downloads`, bên ngoài thư mục dự án. Chỉ cần chép file ZIP này. Giải nén lớp ngoài trước để có `READ_ME_FIRST.md`, hai ZIP thành phần, Git bundle, checksum và thông tin môi trường; sau đó làm theo các bước dưới. `transfer_backup/` là thư mục làm việc để tạo gói, không cần chép thêm nếu đã có ZIP cuối.

## Các thành phần

- GitHub: code, cấu hình dùng chung, notebook, tài liệu và báo cáo được Git theo dõi. Dataset, weights và lịch sử Codex không nằm trên GitHub.
- `project-local.zip`: code, cấu hình, tài liệu, `weights/`, `runs/`, video và báo cáo local. Theo yêu cầu cuối, **bỏ dataset huấn luyện** (`data/dataset`, `data/raw`, `data/exports`, `data/reference`), `.venv`, `.python`, `.tools`, node_modules và cache/tạm. Không xóa bản gốc trên máy cũ. Máy mới cần cài lại môi trường; đường dẫn dataset trong cấu hình là tham chiếu lịch sử, không có nghĩa dữ liệu đã được mang theo.
- `repository.bundle`: toàn bộ refs và lịch sử Git local tại thời điểm tạo; có thể clone khi không có mạng. Không chứa thay đổi chưa commit.
- `codex-local.zip`: trạng thái local từ `C:\Users\aawm8\.codex`, gồm sessions, archived_sessions nếu có, index, database, cấu hình, skills, plugins, attachments, visualizations và worktrees. Có giữ cả profile `.codex` lồng bên trong nếu tồn tại. Bỏ `auth.json`, sandbox secrets, khóa tiến trình và runtime/cache có thể cài lại. Đây vẫn là dữ liệu riêng tư: hội thoại/cấu hình có thể chứa thông tin nhạy cảm.
- `backup_report.json`, `SHA256SUMS.txt`: kết quả kiểm CRC ZIP, integrity SQLite, số file, lỗi nếu có và checksum để đối chiếu sau khi chép. Manifest từng file nằm trong mỗi ZIP.

Đây là snapshot lúc đang sử dụng Codex: các database được chụp bằng SQLite backup để hợp nhất WAL, không chép database đang ghi một cách thô. Các file hội thoại và các database không có một thời điểm giao dịch chung; tin nhắn sau thời điểm chụp chưa nằm trong gói. Chỉ coi gói hoàn tất khi `backup_report.json` có `finished_local`, `errors` rỗng và đủ ba file ZIP/bundle.

Kiểm tra nguồn ngày 17/10: profile chính có 67 bản ghi chat; profile lồng có 146 bản ghi, không cộng hai số này thành số chat duy nhất. ZIP giữ 234 file JSONL trong các thư mục sessions. Có hai bản ghi profile chính thiếu file rollout ngay trên máy cũ; không thể khẳng định đã cứu nội dung hai chat đó. Các rollout của profile lồng tìm thấy trong ZIP theo tên file nhưng đường dẫn database cũ không còn đúng. Hai profile được giữ riêng, chưa hợp nhất database hoặc sửa index. Không bảo đảm tất cả chat cũ tự xuất hiện trong sidebar sau khi giải nén.

Kiểm tra code trước push: Ruff đạt, 80 file đúng format. Pytest chưa đạt: Windows dùng encoding `gbk` gây lỗi đọc notebook; chạy Python ở UTF-8 đi tiếp được 24 test nhưng gặp lỗi encoding đầu ra subprocess ở test TUI. Lượt pytest trong sandbox không hoàn tất và đã dừng. Đây là snapshot bảo toàn công việc, không phải bản phát hành đã qua toàn bộ test.

## Khôi phục dự án

Để Codex trên máy mới nắm ngữ cảnh, mở đúng thư mục project và gửi: **“Đọc AGENTS.md, tasks/project_memory.md và mục chính thức trong tasks/plan.md trước. Tiếp đó đọc docs/guides/move_to_another_computer.md và reports/maintenance/restructure_20261015.md. Đây là bản chuyển máy không kèm dataset/môi trường; kiểm tra trạng thái rồi tiếp nối công việc, không tự train hoặc đổi mục tiêu đề tài.”** Các file này ghi mục tiêu, quyết định, cấu trúc, việc đã làm và giới hạn kiểm chứng. Không cần đọc lại toàn bộ lịch sử chat để lấy được ngữ cảnh dự án cốt lõi.

1. Trên máy Windows mới, cài Git, Python 3.11 và FFmpeg/FFprobe. Giải nén dự án vào thư mục trống; ưu tiên giữ `C:\Users\aawm8\Downloads\BTL-AI-Camera-cd8` nếu khả thi để các tham chiếu lịch sử tiếp tục đúng.
2. Clone repository rồi giải nén `project-local.zip` vào đúng thư mục clone, chấp nhận bản snapshot ghi đè file của clone. Khi không có mạng, clone từ bundle:

   ```powershell
   git clone "E:\transfer_backup\repository.bundle" "C:\Users\aawm8\Downloads\BTL-AI-Camera-cd8"
   cd "C:\Users\aawm8\Downloads\BTL-AI-Camera-cd8"
   git remote set-url origin https://github.com/KingTmata/BTL-AI-Camera-cd8
   ```

   Thay `E:` bằng ổ thực tế. Đừng giải nén ZIP thành một thư mục dự án lồng thêm bên trong.
3. Tạo lại môi trường CPU rồi cài TUI/tracking:

   ```powershell
   .\setup_demo.cmd
   .venv\Scripts\python.exe -m pip install -e ".[terminal,tracking,dev]"
   .venv\Scripts\python.exe -m classroom_ai.video --help
   ffmpeg -version
   ffprobe -version
   .\run_video.cmd
   ```

   Máy NVIDIA dùng [hướng dẫn GPU](gpu_setup.md); bootstrap trên cài bản CPU. Không sao chép `.venv` cũ. Môi trường mới cần Internet để tải dependencies; bản backup không phải bộ cài offline.
4. Kiểm `configs/dataset_current.json`, `configs/data*.yaml` và đường dẫn video. Nếu đổi username/thư mục, sửa cấu hình đang dùng khi cần; giữ nguyên JSON/báo cáo lịch sử làm bằng chứng, không thay hàng loạt toàn bộ repository.

## Khôi phục lịch sử Codex local

Tài liệu chính thức xác nhận trạng thái local mặc định ở `CODEX_HOME`, thường là `~/.codex`; thông tin đăng nhập có thể ở `auth.json` hoặc kho khóa hệ điều hành: [Config and state locations](https://learn.chatgpt.com/docs/config-file/config-advanced#config-and-state-locations). Quy trình bên dưới là sao lưu filesystem theo dữ liệu kiểm tra trên máy này; không phải tính năng import được bảo đảm tương thích mọi phiên bản.

1. Cài Codex cùng phiên bản với máy cũ nếu có thể. **Thoát hẳn Codex/CLI/extension trên máy mới trước khi khôi phục** để tránh app ghi đè database.
2. Nếu máy mới đã có chat, đổi tên `%USERPROFILE%\.codex` thành một thư mục backup riêng. Không trộn database hai máy hoặc ghi đè chat mới mà chưa sao lưu.
3. Giải nén `codex-local.zip` vào `%USERPROFILE%\.codex`. Root của ZIP phải trở thành `.codex\sessions`, `.codex\state_5.sqlite`, v.v.; không thêm một tầng `.codex` ngoài ý muốn. Profile lồng `.codex\.codex` có sẵn trong ZIP là dữ liệu thực của máy cũ, không phải lỗi giải nén.
4. Mở Codex, đăng nhập lại cùng tài khoản và mở project đã khôi phục. Cài lại plugin/runtime khi được yêu cầu. Kiểm cả chat cũ, chat archived và đính kèm trước khi bỏ bản sao máy cũ.
5. SQLite/index/config có thể giữ đường dẫn tuyệt đối `C:\Users\aawm8\...`. Giữ nguyên đường dẫn giúp giảm lỗi. Nếu username khác, giữ bản backup nguyên vẹn, mở project ở vị trí mới và xử lý đường dẫn bằng một tác vụ migration riêng; không search-replace file SQLite. Worktree Codex có `.git` trỏ ra repository gốc nên không bảo đảm hoạt động chỉ bằng giải nén; nội dung file đã được giữ để phục hồi thủ công.

Không tự bật lại các automation trên máy mới trước khi kiểm đường dẫn, tránh chạy trùng với máy cũ. Lịch sử trong gói là local; không bao gồm các hội thoại cloud chưa có trên đĩa hoặc bản ghi nằm ngoài hai profile đã tìm thấy.

## Kiểm tra bản chép

Trong thư mục backup trên máy mới:

```powershell
Get-FileHash .\project-local.zip -Algorithm SHA256
Get-FileHash .\codex-local.zip -Algorithm SHA256
Get-FileHash .\repository.bundle -Algorithm SHA256
```

Đối chiếu với `SHA256SUMS.txt`. Sau đó mở thử dataset, checkpoint và vài chat cũ. Chỉ dọn máy cũ khi đã khôi phục và kiểm tra thành công trên máy mới.
