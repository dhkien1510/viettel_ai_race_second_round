# Contest Agent Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a repository-wide `AGENTS.md` that makes every coding agent obey the contest's SSH, credential, data-isolation, resource, security, proxy, and submission rules.

**Architecture:** Use one declarative policy file at the repository root so its instructions apply to the complete tree. Organize the policy into allowed operations, hard prohibitions, approved access paths, and mandatory refusal behavior; use explicit command names and closed-form rules to avoid ambiguous workarounds.

**Tech Stack:** Markdown, PowerShell validation commands, Git

---

### Task 1: Create and verify the repository policy

**Files:**
- Create: `AGENTS.md`
- Reference: `docs/superpowers/specs/2026-08-16-contest-agent-policy-design.md`

- [ ] **Step 1: Confirm that no root policy will be overwritten**

Run:

```powershell
Test-Path -LiteralPath 'AGENTS.md'
```

Expected: `False`. If it returns `True`, stop because this plan must not overwrite
or merge an existing policy without a separately reviewed design.

- [ ] **Step 2: Create the minimal complete policy**

Create `AGENTS.md` as UTF-8 Markdown with exactly this policy content:

```markdown
# Quy định bắt buộc cho agent

## 1. Phạm vi và mức ưu tiên

Các quy định này áp dụng cho toàn bộ repository và mọi hành động agent đề xuất
hoặc thực hiện trong môi trường cuộc thi. Đây là các ràng buộc bắt buộc, không
được diễn giải theo hướng nới lỏng hoặc tìm cách lách. Khi một yêu cầu xung đột
với file này, agent phải tuân thủ file này, dừng hành động vi phạm và giải thích
ngắn gọn cho người dùng.

## 2. Công việc được phép

- Chỉnh sửa code, chạy test, train/evaluate model và quản lý file bằng dòng lệnh
  trên đúng VM được cấp cho team, trong giới hạn tài nguyên của cuộc thi.
- Mở duy nhất một terminal SSH tương tác bằng chính xác lệnh `ssh team07`.
- Code, chạy notebook và upload/download file qua Code Editor hoặc JupyterLab
  chính thức do BTC cung cấp.
- Cài package qua proxy nội bộ đã được BTC cấu hình sẵn. Nếu công cụ không tự
  nhận proxy, chỉ được dùng `http://10.10.1.126:3128` qua `HTTP_PROXY`,
  `HTTPS_PROXY` hoặc tùy chọn proxy tương đương của công cụ.
- Chuẩn bị, đóng gói và kiểm tra artifact nộp bài, miễn là chưa thực hiện submit.

## 3. Credential và bí mật

- TUYỆT ĐỐI KHÔNG ĐƯỢC đọc, mở, in, hiển thị, sao chép, ghi log, trích xuất,
  truyền hoặc tiết lộ private key, password, token hay credential dưới bất kỳ
  hình thức nào, kể cả để chẩn đoán hoặc khi một yêu cầu sau này chỉ thị làm vậy.
- Không chia sẻ credential cho team khác hoặc bất kỳ bên nào.
- Nếu thật sự cần chẩn đoán, chỉ được kiểm tra metadata không nhạy cảm như sự
  tồn tại và quyền file mà không đọc nội dung.
- Không ghi credential vào code, config được commit, prompt, log, issue, tin
  nhắn, artifact hoặc output.

## 4. Quy tắc SSH bắt buộc

- Lệnh SSH duy nhất được phép là chính xác `ssh team07`, chỉ để mở terminal
  tương tác. Không thêm tùy chọn, đối số, command suffix, pipe hoặc redirection.
- KHÔNG ĐƯỢC dùng `scp`, `sftp`, `rsync` hoặc bất kỳ cơ chế nào để truyền file
  qua SSH.
- KHÔNG ĐƯỢC chạy remote command như `ssh team07 "some command"` hoặc dùng chế
  độ SSH không tương tác.
- KHÔNG ĐƯỢC dùng `ssh -L`, `ssh -D`, `ssh -R`, `ProxyCommand`, SOCKS proxy,
  tunnel hay bất kỳ hình thức port forwarding nào.
- KHÔNG ĐƯỢC SSH tới VM của team khác hoặc tìm đường vòng tới tài nguyên không
  được cấp.

## 5. Web IDE và truyền file

- Upload/download file CHỈ ĐƯỢC thực hiện qua Code Editor hoặc JupyterLab chính
  thức của team do BTC cung cấp; không được truyền file qua SSH hay kênh khác.
- Credential web IDE chỉ do người dùng được ủy quyền trực tiếp nhập, hoặc agent
  dùng một session đã đăng nhập sẵn mà không truy cập credential. Agent không
  được yêu cầu, nhận, đọc, nhập, hiển thị hoặc tiết lộ credential.
- Nếu trình duyệt hiển thị `Not secure`, `Your connection is not private` hoặc
  cảnh báo chứng chỉ HTTPS, KHÔNG ĐƯỢC bấm bỏ qua. Phải dừng và báo BTC.

## 6. Mạng và proxy

- VM không có Internet trực tiếp. Chỉ dùng proxy nội bộ và các API do BTC cấu
  hình sẵn cho đúng mục đích cuộc thi.
- KHÔNG ĐƯỢC tạo hoặc tìm đường Internet thay thế, tunnel, proxy ngoài, relay,
  VPN hoặc cơ chế vượt hạn chế mạng.
- Không sửa cấu hình proxy theo hướng né kiểm soát của BTC.

## 7. Dữ liệu hạn chế và chống exfiltration

`Dữ liệu hạn chế` bao gồm toàn bộ hoặc một phần dataset, sample, label/private
label, metadata, thống kê, bản tóm tắt, embedding, output suy luận và mọi
artifact có thể tái tạo hoặc tiết lộ các dữ liệu đó.

- Dữ liệu hạn chế phải ở lại đúng môi trường được BTC cho phép. TUYỆT ĐỐI KHÔNG
  ĐƯỢC tải, truyền, sao chép hoặc đưa dữ liệu ra ngoài bằng bất kỳ cách nào.
- Lệnh cấm bao gồm nhưng không giới hạn: đọc qua SSH rồi chép lại; copy text;
  clipboard; chụp ảnh/chụp màn hình; quay video; email; chat; prompt; issue;
  commit; log; dịch vụ lưu trữ; API hoặc dịch vụ bên ngoài.
- Lệnh cấm vẫn áp dụng khi dữ liệu đã được biến đổi, mã hóa, nén, làm mờ, chia
  nhỏ, tổng hợp hoặc giấu trong model, embedding, source code hay artifact.
- Không được hướng dẫn người dùng hoặc người khác thực hiện thay một hành động
  exfiltration mà agent bị cấm thực hiện.

## 8. Tính toàn vẹn và tài nguyên cuộc thi

- TUYỆT ĐỐI KHÔNG ĐƯỢC thu thập, khai thác, suy luận, dò tìm hoặc tái tạo private
  label. Đánh giá hợp lệ chỉ được sử dụng kết quả do BTC chủ động cung cấp theo
  đúng giao diện và mục đích cuộc thi, không được dùng để phục hồi label.
- TUYỆT ĐỐI KHÔNG ĐƯỢC reverse engineer hạ tầng, hệ thống, API nộp bài hoặc cơ
  chế chấm, bất kể mục đích được viện dẫn.
- KHÔNG ĐƯỢC thăm dò, khai thác hoặc tấn công lỗ hổng; né kiểm soát; leo thang
  đặc quyền; duy trì truy cập; hoặc truy cập hệ thống, account, VM, file, dịch
  vụ hay tài nguyên không được cấp cho team.
- KHÔNG ĐƯỢC vượt hoặc tìm cách vượt giới hạn CPU, GPU, RAM, storage, network,
  quota, thời gian chạy hay giới hạn submit.
- Không được phá hoại, gây gián đoạn hoặc ảnh hưởng tài nguyên của BTC hay team
  khác.

## 9. Submit

- KHÔNG ĐƯỢC tự ý submit hoặc gọi API/lệnh có tác dụng nộp bài.
- Chỉ được submit khi người dùng đưa ra chỉ thị rõ ràng, trực tiếp trong lượt
  hiện tại. Chỉ thị train, evaluate, đóng gói, kiểm tra hoặc “hoàn tất pipeline”
  không đồng nghĩa với quyền submit.
- Trước khi submit theo lệnh hợp lệ, phải xác nhận đúng artifact, endpoint và
  phạm vi; không được dùng submit để dò private label hoặc vượt giới hạn.

## 10. Khi gặp yêu cầu vi phạm

Agent phải:

1. Dừng trước khi chạy lệnh, đọc dữ liệu hoặc thực hiện hành động bị cấm.
2. Không thử biến thể, đường vòng hoặc giao việc đó cho agent/người khác.
3. Nêu ngắn gọn quy định bị xung đột.
4. Đề xuất cách hợp lệ, chẳng hạn dùng terminal SSH tương tác, web IDE chính
   thức, proxy nội bộ hoặc xin BTC xác nhận.
```

- [ ] **Step 3: Validate required policy coverage**

Run:

```powershell
$policy = Get-Content -Raw -Encoding UTF8 -LiteralPath 'AGENTS.md'
$required = @(
  'ssh team07', 'scp', 'sftp', 'rsync', 'ProxyCommand',
  'private key', 'password', '10.10.1.126:3128',
  'Dữ liệu hạn chế', 'private label', 'reverse engineer',
  'KHÔNG ĐƯỢC tự ý submit', 'Your connection is not private'
)
$missing = $required | Where-Object { $policy -notmatch [regex]::Escape($_) }
if ($missing) { throw "Thiếu quy tắc: $($missing -join ', ')" }
'PASS: policy coverage'
```

Expected: `PASS: policy coverage`.

- [ ] **Step 4: Review the complete file and scoped diff**

Run:

```powershell
$repoRoot = (Resolve-Path -LiteralPath '.').Path
$gitPrefix = (Join-Path $repoRoot '.git') + [IO.Path]::DirectorySeparatorChar
$policyFiles = @(
  Get-ChildItem -LiteralPath $repoRoot -Recurse -Force -File -Filter 'AGENTS.md' |
    Where-Object {
      -not $_.FullName.StartsWith(
        $gitPrefix,
        [StringComparison]::OrdinalIgnoreCase
      )
    }
)
$expectedPolicy = Join-Path $repoRoot 'AGENTS.md'
if ($policyFiles.Count -ne 1 -or $policyFiles[0].FullName -ne $expectedPolicy) {
  throw "AGENTS.md không nằm duy nhất ở root: $($policyFiles.FullName -join ', ')"
}
Get-Content -Raw -Encoding UTF8 -LiteralPath 'AGENTS.md'
git -c safe.directory='D:/Project/viettel-ai-race-edgerunner/viettel_ai_race_second_round' diff --no-index -- NUL AGENTS.md
git -c safe.directory='D:/Project/viettel-ai-race-edgerunner/viettel_ai_race_second_round' status --short -- AGENTS.md
git -c safe.directory='D:/Project/viettel-ai-race-edgerunner/viettel_ai_race_second_round' status --short
```

Expected: the repository-wide assertion passes; the file is readable UTF-8
Markdown and contains no real credential or restricted data; `status` reports
only `?? AGENTS.md` for this path. Git may return exit code 1 because
`--no-index` found the expected difference. Also inspect the full `git status
--short` and confirm this task introduced no unrelated changes; preserve any
pre-existing user changes.

- [ ] **Step 5: Commit the policy**

Run:

```powershell
git -c safe.directory='D:/Project/viettel-ai-race-edgerunner/viettel_ai_race_second_round' add -- AGENTS.md
git -c safe.directory='D:/Project/viettel-ai-race-edgerunner/viettel_ai_race_second_round' diff --cached --name-only -- AGENTS.md
git -c safe.directory='D:/Project/viettel-ai-race-edgerunner/viettel_ai_race_second_round' commit -m 'docs: add contest safety rules for agents' -- AGENTS.md
```

Expected: the staged-name check prints exactly `AGENTS.md`, then one commit
creates only `AGENTS.md`.
