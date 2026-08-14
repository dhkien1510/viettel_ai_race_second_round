# RxNorm Drug Finder

Web tra cứu local bằng HTML, CSS và JavaScript. Server Python chỉ dùng standard library và đọc
trực tiếp cache của `rxnorm-linker`.

## Chạy

Khi `web` và `rxnorm-linker` nằm cạnh nhau, chạy từ thư mục repository:

```powershell
python web/server.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:8765
```

Đổi port nếu cần:

```powershell
python web/server.py --port 9000
```

## Mang folder web sang nơi khác

Chỉ cần chỉ vị trí của `rxnorm-linker`; mọi asset HTML/CSS/JS đều được resolve tương đối từ folder
`web`, không phụ thuộc ổ đĩa hay working directory:

```powershell
python D:\tools\rxnorm-web\server.py `
  --rxnorm-root D:\data\rxnorm-linker
```

Hoặc đặt biến môi trường một lần:

```powershell
$env:RXNORM_LINKER_ROOT = "D:\data\rxnorm-linker"
python D:\tools\rxnorm-web\server.py
```

Có thể trỏ thẳng cache khác bằng `--cache`, nhưng vẫn cần `--rxnorm-root` để Python đọc được cấu
trúc object trong pickle:

```powershell
python web/server.py --cache D:\rxnorm-data\rxnorm_index.pkl
```

API:

```text
GET /api/search?q=aspirin%2081%20mg&limit=20
GET /api/search?q=860975
```
