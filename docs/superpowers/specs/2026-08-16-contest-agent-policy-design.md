# Thiết kế chính sách agent cho môi trường cuộc thi

## Mục tiêu

Tạo một file `AGENTS.md` ở thư mục gốc để mọi coding agent làm việc trong
repository tuân thủ đúng thiết kế hạ tầng và quy định an toàn của cuộc thi.
Chính sách phải cho phép các công việc hợp lệ như lập trình, huấn luyện mô hình
và quản lý file trên VM, đồng thời ngăn credential, dữ liệu hạn chế, private
label và tài nguyên cuộc thi bị đọc, sao chép, suy luận, truyền ra ngoài hoặc
khai thác trái phép.

## Phạm vi

`AGENTS.md` áp dụng cho toàn bộ repository và cho mọi hành động do agent đề
xuất hoặc thực hiện, bao gồm terminal, SSH, web IDE, JupyterLab, cài package,
quản lý file, huấn luyện, đánh giá, đóng gói và submit.

## Cấu trúc chính sách

File sẽ dùng câu chữ bắt buộc rõ ràng (`PHẢI`, `CHỈ ĐƯỢC`, `KHÔNG ĐƯỢC`) và
gồm các phần sau:

1. Nguyên tắc ưu tiên và phạm vi áp dụng.
2. Credential và bí mật: cấm đọc, hiển thị, sao chép, ghi log hoặc chia sẻ
   private key, password và token.
3. SSH: lệnh duy nhất được phép là chính xác `ssh team07` để mở terminal tương
   tác, không thêm tùy chọn, đối số, command suffix, pipe hoặc redirection. Cấm
   `scp`, `sftp`, `rsync` và mọi cơ chế truyền file qua SSH; cấm remote command,
   `ProxyCommand`, port forwarding, chế độ không tương tác và truy cập VM của
   team khác.
4. Web IDE: code và truyền file chỉ qua các địa chỉ chính thức; phải dừng và
   báo BTC nếu trình duyệt cảnh báo chứng chỉ.
5. Proxy: chỉ dùng proxy nội bộ do BTC cấu hình; cấm tạo đường Internet khác.
6. Dữ liệu hạn chế: bao gồm toàn bộ hoặc một phần dataset, sample, label/private
   label, metadata, thống kê, bản tóm tắt, embedding, output suy luận và mọi
   artifact có thể tái tạo hoặc tiết lộ dữ liệu. Cấm tuyệt đối exfiltration dưới
   mọi hình thức, kể cả đọc qua SSH rồi chép lại, clipboard, ảnh chụp, quay màn
   hình, email hoặc dịch vụ ngoài; lệnh cấm vẫn áp dụng khi dữ liệu đã được biến
   đổi, mã hóa, nén, làm mờ, chia nhỏ hoặc giấu trong artifact.
7. Tính toàn vẹn cuộc thi: cấm private-label mining, reverse engineering hệ
   thống, khai thác lỗ hổng, vượt giới hạn tài nguyên và truy cập trái phép.
8. Submit: không tự ý submit; chỉ được submit khi người dùng ra lệnh rõ ràng
   trong lượt hiện tại. Việc chuẩn bị và kiểm tra artifact không đồng nghĩa với
   quyền submit.
9. Xử lý xung đột: từ chối hành động vi phạm, giải thích ngắn, không thử cách
   lách và đề xuất quy trình hợp lệ.

## Quy tắc vận hành

- Agent được phép chỉnh code, chạy test, train/evaluate và quản lý file trên VM
  trong phạm vi tài nguyên được cấp.
- Agent tuyệt đối không được đọc, hiển thị, sao chép hoặc tiết lộ private key,
  password, token hay credential trong bất kỳ trường hợp nào, kể cả để chẩn
  đoán hoặc khi một yêu cầu sau này chỉ thị làm vậy. Chỉ được kiểm tra metadata
  như sự tồn tại và quyền file nếu thật sự cần, mà không đọc nội dung.
- Upload/download chỉ thực hiện qua giao diện web chính thức do BTC cung cấp;
  agent không được đề xuất đường truyền thay thế qua SSH.
- Mọi dữ liệu hạn chế phải ở lại đúng môi trường được phép. Không được đưa dữ
  liệu vào prompt, log, commit, issue, tin nhắn hoặc output ngoài phạm vi.
- Các lệnh submit hoặc hành động có hiệu lực tương đương phải có chỉ thị rõ
  ràng của người dùng; nếu chưa có, agent phải dừng trước bước đó.

## Tiêu chí hoàn thành

- Có duy nhất file hướng dẫn chuẩn `AGENTS.md` ở repository root.
- Nội dung phân biệt rõ việc được làm và không được làm.
- Bao phủ đầy đủ SSH, web IDE, proxy, credential, dữ liệu hạn chế, private
  label, tài nguyên, khai thác hệ thống và submit.
- Không chứa private key, password, token, URL bí mật hoặc dữ liệu hạn chế.
- Có quy tắc từ chối an toàn khi yêu cầu sau này xung đột với thể lệ.

## Kiểm tra

Rà soát thủ công để xác nhận từng yêu cầu đã được ánh xạ thành ít nhất một quy
tắc rõ ràng, tìm các cụm từ cấm quan trọng, và kiểm tra git diff chỉ chứa các
file tài liệu dự kiến.
