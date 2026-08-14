# RxNorm Local Linker

Pipeline ánh xạ entity `THUỐC` trong JSON NER sang mã RxNorm (RXCUI). Hệ thống hỗ trợ
tiếng Anh/Việt, tên biệt dược, liều lượng, đường dùng và tần suất.

## Yêu cầu

- Python 3.10+
- PowerShell (nếu dùng menu tương tác trên Windows)
- Cài dependency:

```powershell
cd rxnorm-linker
python -m pip install -r requirements.txt
```

- Giải nén RxNorm Full Prescribable snapshot để có:

```text
data/rxnorm/rrf/RXNCONSO.RRF
```

## Build cache

Normalization hiện dùng cache version `mv-hybrid-v3-normalization`. Sau khi clone repo,
đổi normalization hoặc gặp lỗi cache version, phải build lại index:

```powershell
cd rxnorm-linker
python scripts/build_rxnorm_index.py
```

Nếu cần tier 3 (embedding):

```powershell
python scripts/build_rxnorm_index.py --build-embeddings
```

Các file RRF và cache lớn được gitignore, không được push lên repository.

## Chạy bằng menu tương tác

Từ thư mục gốc `viettel_ai_race`:

```powershell
.\rxnorm-linker\scripts\run_rxnorm.ps1
```

Script lần lượt hỏi:

1. `InputDir`.
2. Profile linking.
3. `OutputDir`.

Đường dẫn tương đối luôn được resolve từ thư mục gốc repository. Ví dụ:

```text
.\submission\0801\Cuong01
```

Các profile:

| Lựa chọn | Profile |
|---:|---|
| 1 | Tier 1 + 2 + 3, top 3 |
| 2 | Tier 1 + 2 + 3, top 1 |
| 3 | Không tier 3, top 1 (khuyến nghị để thử nghiệm bảo thủ) |
| 4 | Không tier 3, top 3 |
| 5 | Exact-only |
| 6 | Không tier 3, top 1, liều không khớp fallback về bare ingredient |
| 7 | Score-guided conservative heuristic, một candidate; ưu tiên concept cụ thể khi có dose/form rõ |

Nếu `OutputDir` chưa tồn tại, script tự tạo. Nếu output trùng input, script chạy
in-place và chỉ tiếp tục khi người dùng nhập chính xác `OVERWRITE`. Sau khi chạy,
script tự in thống kê candidate/tier.

### Tab completion cho đường dẫn

`Read-Host` không hỗ trợ Tab completion. Muốn dùng Tab giống lệnh `cd`, truyền đường
dẫn khi gọi script:

```powershell
.\rxnorm-linker\scripts\run_rxnorm.ps1 `
  -InputDir .\submission\0801\Cuong01 `
  -OutputDir .\submission\0801\Cuong01_no_tier3_top1
```

## Chạy CLI trực tiếp

### Full tier 1 + 2 + 3

```powershell
cd rxnorm-linker
python scripts/label_rxnorm_candidates.py `
  --dir ..\submission\0801\Cuong01 `
  --out ..\submission\0801\Cuong01_full `
  --top-k 3
```

### Tắt tier 3, chỉ giữ top 1

```powershell
python scripts/label_rxnorm_candidates.py `
  --dir ..\submission\0801\Cuong01 `
  --out ..\submission\0801\Cuong01_no_tier3_top1 `
  --no-tier3 `
  --top-k 1
```

### Exact-only

```powershell
python scripts/label_rxnorm_candidates.py `
  --dir ..\submission\0801\Cuong01 `
  --out ..\submission\0801\Cuong01_exact_only `
  --exact-only
```

Nếu bỏ `--out`, JSON đầu vào được cập nhật trực tiếp.

### Score-guided conservative, tối đa 1 candidate

Mode này không dùng bảng gán theo mention của submission. Nó chạy linker thật, giữ tối đa 1
candidate, tắt tier 3 rủi ro cao, và dùng vài rule tổng quát để ưu tiên concept RxNorm cụ thể
hơn khi span có dose/form/đường dùng rõ, ví dụ chọn clinical drug/product thay vì component
trừu tượng nếu điểm gần nhau.

```powershell
python scripts/label_rxnorm_candidates.py `
  --dir ..\codexmoinhat `
  --out ..\submission\0802\Cuong04_like `
  --score-guided-manual
```

## Tra cứu và debug một span

```powershell
cd rxnorm-linker
python scripts/rxnorm_lookup.py --span "vancomycin 1 gram" --debug
python scripts/rxnorm_lookup.py --span "Omez 20mg x 1 viên, uống 8h sáng" --debug
```

## Thống kê

Phân loại các entity thuốc theo exact, lexical, nearest-dose, tier 3 và rỗng:

```powershell
python scripts/stats_rxnorm_tiers.py `
  ..\submission\0801\Cuong01_no_tier3_top1 `
  --examples 10
```

So sánh các chiến thuật với một thư mục JSON ground truth:

```powershell
python scripts/eval_rxnorm_strategies.py ..\output --skip-tier3
```

## Các normalization quan trọng

- Quy đổi `G → MG` và `MCG/UG → MG` cho cả query và index.
- `Glucose/Dextrose 5% → 50 MG/ML` theo whitelist w/v có kiểm soát.
- Loại metadata tiếng Việt như `x 1 viên`, `x 2 ống`, `uống 8h sáng` trước tokenize.
- Giữ chữ số trong token như `B12` thay vì cắt thành `B`.
- Hỗ trợ phrase alias và alias nhiều token, ví dụ:
  - `vitamin C → ascorbic acid`
  - `vitamin B12/B12 → cyanocobalamin`
  - `cotrimoxazol → sulfamethoxazole / trimethoprim`
  - `Pimperan/Pimperam → metoclopramide`
  - `Omez → omeprazole`
  - `Furosemid → furosemide`

## Pipeline

1. Parse và canonicalize ingredient/form/strength.
2. Tìm lexical trên các concept có liều (SCD/SBD/PSN/SY...).
3. Nếu không có liều, hoặc không tìm được sản phẩm phù hợp, tìm ingredient/brand
   (IN/PIN/MIN/BN).
4. Nếu bật tier 3 và lexical thất bại, SapBERT retrieval tạo candidate pool, sau đó
   `bge-reranker-v2-m3` rerank và áp dụng confidence gate.

Tier 3 hữu ích cho typo/OOV nhưng có thể trả nhiều candidate. Với submission cần bảo
thủ, dùng `--no-tier3 --top-k 1` hoặc `--exact-only`.

## Test

```powershell
cd rxnorm-linker
$env:PYTHONIOENCODING = "utf-8"
python tests/test_rxnorm_linker.py
```

Bộ test bao gồm regression cho quy đổi đơn vị, token rác tiếng Việt, `B12`, vitamin C,
cotrimoxazol, Pimperan/Pimperam, Omez và các ví dụ RxNorm chuẩn.

## Cấu trúc chính

```text
rxnorm-linker/
├── scripts/
│   ├── build_rxnorm_index.py
│   ├── label_rxnorm_candidates.py
│   ├── run_rxnorm.ps1
│   ├── rxnorm_lookup.py
│   ├── stats_rxnorm_tiers.py
│   └── eval_rxnorm_strategies.py
├── src/rxnorm/
│   ├── build_index.py
│   ├── config.py
│   ├── embed_index.py
│   ├── linker.py
│   ├── normalize.py
│   ├── query_expansion.py
│   └── rerank.py
└── tests/test_rxnorm_linker.py
```
