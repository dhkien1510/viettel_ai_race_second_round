# H200 Preparation And Training Log

Tài liệu này ghi lại quá trình kiểm tra môi trường, huấn luyện và đánh giá pipeline trên một GPU NVIDIA H200 140 GB. Mục tiêu là mô phỏng sát môi trường thi của Ban Tổ chức (BTC), lưu lại cấu hình có thể tái lập và tránh lặp lại các lỗi đã gặp.

Cập nhật gần nhất: 2026-08-14.

## Trạng thái tổng quan

| Luồng | Module | Trạng thái | Kết quả |
|---|---|---|---|
| README full-fit | Qwen NER | Hoàn tất | exact span+type fit F1 98,03% |
| README full-fit | Qwen assertion SEQ_CLS | Hoàn tất | exact/Jaccard fit 99,70% |
| Holdout 80/20 | Qwen NER batch 32 | Hoàn tất, không chọn | exact span+type F1 46,34% |
| Holdout 80/20 | Qwen NER batch 8 | Hoàn tất | raw F1 56,83%; train-only repair F1 60,41% |
| Holdout 80/20 | Qwen assertion SEQ_CLS | Hoàn tất | exact/Jaccard 89,65% |
| Holdout 80/20 | Confidence calibration | Đã implement, chưa chạy | Chờ debug inference |
| Full pipeline | ICD retrieval | Hoàn tất | candidate-pool recall 88,19% |
| Full pipeline | ICD XLM-R reranker | Hoàn tất | best val entity accuracy 82,57% |
| Full pipeline | ICD end-to-end | Hoàn tất | all 90,76%; holdout 80,73% |
| Full pipeline | RxNorm lexical linker | Hoàn tất | 198/268 thuốc có RxCUI |
| Full pipeline | NER -> assertion -> ICD -> RxNorm | Hoàn tất | 100 file, validator `errors=[]` |

## 1. Môi trường đã xác nhận

| Thành phần | Phiên bản / trạng thái |
|---|---|
| GPU | NVIDIA H200, 140 GB VRAM |
| NVIDIA driver | 580.159.03 |
| CUDA runtime | 13.0 |
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu130 |
| vLLM | 0.26.0 |
| Transformers | 5.12.1 |
| PEFT | 0.20.0 |
| bitsandbytes | 0.50.1 |
| Accelerate | 1.14.0 |
| BF16 | Đã kiểm tra, hoạt động trên H200 |

Môi trường Conda thử nghiệm: `/workspace/envs/btc`.

Các model đã cache cục bộ:

- `Qwen/Qwen2.5-3B-Instruct`
- `FacebookAI/xlm-roberta-base`
- `BAAI/bge-m3`

Thứ tự cài đặt quan trọng:

1. Cài `vllm==0.26.0` trước để nhận đúng Torch 2.11/cu130 và kernel stack.
2. Cài các dependency còn lại.
3. Pin `transformers==5.12.1` theo môi trường BTC.
4. Không cài đè `torch`, `torchvision`, `torchaudio` hoặc các kernel package do vLLM quản lý.

Kiểm tra `pip check` đã trả về `No broken requirements found`.

## 2. Kiểm tra dữ liệu và source

- Silver dataset có 100 JSON, 2.709 entity và đã qua schema validation.
- Toàn bộ span trong silver dataset khớp văn bản đầu vào theo validator hiện tại.
- Các entrypoint NER, assertion và ICD đều import/chạy được trên Transformers 5.12.1.
- RxNorm index có sẵn tại `rxnorm-linker/data/rxnorm/cache/rxnorm_index.pkl`; module RxNorm không cần train neural model.
- ICD reranker vẫn cần tạo `models/reranker/best.pt` bằng script train ICD.

## 3. Lỗi split nghiêm trọng đã phát hiện

File `output/superbest_dataset_split.json` cũ chứa cùng 100 ID trong cả `train` và `validation`. Vì vậy metric validation cũ đo trên file này là in-sample/fit score, không phải khả năng tổng quát hóa.

Quy tắc đánh giá từ thời điểm này:

- Dùng split stratified 80/20 với seed 42 cho đánh giá holdout.
- Không dùng nhãn validation trong inference hoặc boundary repair.
- Không gọi kết quả full-fit trên chính 100 tài liệu là validation F1.
- Chỉ full-fit 100 tài liệu sau khi chốt hyperparameter bằng holdout.

Split hợp lệ hiện được lưu kèm checkpoint tại:

`models/btc_h200/ner_fold0_b8/document_split.json`

## 4. NER smoke test

Cấu hình smoke test ban đầu:

```yaml
model: Qwen/Qwen2.5-3B-Instruct
task: TOKEN_CLS
quantization: NF4 4-bit
precision: BF16
max_length: 128
stride: 64
batch_size: 2
gradient_accumulation: 1
epochs: 1
lora_rank: 8
```

Kết quả:

- Hoàn tất trong khoảng 5 phút.
- Xác nhận Qwen token classification, QLoRA, bitsandbytes và BF16 chạy được.
- Metric của smoke test không hợp lệ để báo cáo vì lúc đó split cũ vẫn làm train và validation trùng nhau.

## 5. NER thử nghiệm batch 32

Cấu hình:

```yaml
train_documents: 80
validation_documents: 20
train_windows: 96
validation_windows: 23
max_length: 768
overlap_stride: 128
batch_size: 32
gradient_accumulation: 1
epochs: 16
optimizer_updates: 48
learning_rate: 2.0e-4
lora_rank: 32
precision: BF16
```

Kết quả:

- Thời gian train: 2 phút 33 giây.
- VRAM sử dụng khoảng 19 GB; GPU utilization đạt 100%.
- Holdout exact span+type: precision 40,44%, recall 54,25%, F1 46,34%.
- Dự đoán 774 entity trong khi reference có 577 entity: mô hình bị dư entity rõ rệt.

Kết luận:

- Batch 32 chạy rất nhanh nhưng chỉ có 3 optimizer update mỗi epoch.
- Tổng 48 update không đủ cho tập NER nhỏ và mất cân bằng nhãn.
- Checkpoint này chỉ giữ để ablation, không dùng làm bản cuối.

## 6. NER thử nghiệm batch 8

Cấu hình:

```yaml
train_documents: 80
validation_documents: 20
train_windows: 96
validation_windows: 23
max_length: 768
overlap_stride: 128
batch_size: 8
gradient_accumulation: 1
epochs: 24
optimizer_updates: 288
learning_rate: 2.0e-4
lora_rank: 32
precision: BF16
```

Kết quả:

- Thời gian train: 4 phút 18 giây.
- VRAM sử dụng khoảng 8 GB; GPU utilization đạt 100%.
- Holdout exact span+type: precision 49,30%, recall 67,07%, F1 56,83%.
- Dự đoán 785 entity trong khi reference có 577 entity.
- F1 tăng 10,49 điểm tuyệt đối so với batch 32.

Kết luận:

- Batch 8 là lựa chọn tốt hơn batch 32 cho dữ liệu hiện tại vì tạo đủ optimizer update.
- Nút thắt tiếp theo là precision/entity over-generation, không phải tốc độ GPU.
- Cần hiệu chỉnh confidence guard hoặc postprocessor trên validation, sau đó khóa cấu hình trước khi full-fit.

### 6.1 Phân tích lỗi và boundary repair hợp lệ

Phân tích prediction batch 8 trên 20 file holdout:

- 387 entity khớp exact span+type.
- 392 entity khớp exact span nếu bỏ qua type: lỗi type chỉ chiếm phần nhỏ.
- 493 entity có cùng type và overlap IoU ít nhất 0,5 với reference.
- IoU trung bình của nhóm overlap là khoảng 0,937.

Như vậy một phần lớn lỗi còn lại là sai boundary nhỏ, không phải model hoàn toàn không tìm thấy entity.

Boundary repair được chạy bằng lexicon chỉ xây từ 80 file train. Không đọc 20 file validation khi tạo lexicon. Kết quả:

```yaml
boundary_repaired: 97
adjacent_merged: 8
precision: 0.5264
recall: 0.7088
exact_span_type_f1: 0.6041
```

F1 tăng từ 56,83% lên 60,41%. Precision vẫn là nút thắt do prediction còn 777 entity so với 577 reference. Bước kế tiếp là confidence/margin calibration theo type trên validation.

## 7. Vì sao vẫn dùng max_length 768 và overlap 128

Checkpoint Qwen2.5-3B hỗ trợ tối đa 32.768 token. `max_length=768` là giới hạn của pipeline NER, không phải giới hạn cứng của model.

- Token classification với context rất dài làm chi phí attention tăng mạnh.
- Với split hiện tại, 80 tài liệu chỉ tạo 96 window; overhead sliding window nhỏ.
- Overlap 128 bảo vệ entity nằm tại ranh giới cắt và giữ ngữ cảnh hai phía.
- Script hiện padding mỗi mẫu tới `max_length`; tăng thẳng lên 1.536 hoặc 2.048 có thể làm tốn compute hơn dù số window giảm.

Hướng tối ưu sau này là dynamic padding hoặc sentence/segment-aware chunking, thay vì bỏ overlap hoàn toàn.

## 8. Assertion classifier

Kiến trúc đúng cần dùng:

```text
Qwen NER entity
  -> Assertion Context Builder
  -> Qwen sequence classifier (3 logits)
  -> sigmoid
  -> threshold theo validation
  -> isHistorical / isNegated / isFamily
  -> rule guard/postprocess
```

Checkpoint assertion đóng gói cũ có `task_type=CAUSAL_LM`, trong khi script inference hiện tại yêu cầu `AutoModelForSequenceClassification` với ba logits. Hai định dạng này không tương thích; vì vậy phải train checkpoint assertion SEQ_CLS mới, không được dùng adapter cũ với script classifier.

Dataset assertion hợp lệ đã tạo trên cùng split 80/20:

```yaml
train_rows: 1648
validation_rows: 367
base_model: Qwen/Qwen2.5-3B-Instruct
task: SEQ_CLS
num_labels: 3
max_length: 384
batch_size: 32
gradient_accumulation: 1
epochs: 8
learning_rate: 8.0e-5
precision: BF16
```

Kết quả holdout:

```yaml
validation_exact: 0.89646
validation_jaccard: 0.89646
thresholds:
  isHistorical: 0.65
  isNegated: 0.50
  isFamily: 0.75
positive_weights:
  isHistorical: 5.4627
  isNegated: 9.7712
  isFamily: 15.0
```

Checkpoint assertion holdout được lưu riêng tại `models/btc_h200/assertion_fold0`.

## 8.1 Hai hướng train được tách riêng

### Hướng A: README baseline/full-fit

- Chạy nguyên cấu hình README trên toàn bộ 100 tài liệu.
- NER dùng batch 4, 16 epoch, max length 768, stride 192 và LoRA rank 32.
- Mục tiêu là tái lập fit score khoảng 98% đã ghi nhận trước đây.
- Metric này chỉ là full-fit/in-sample score, không được gọi là holdout F1.

Kết quả NER full-fit đã tái lập trên H200:

```yaml
runtime: 5m37s
optimizer_updates: 480
true_positive: 2661
predicted: 2720
reference: 2709
precision: 0.97831
recall: 0.98228
exact_span_type_f1: 0.98029
```

Checkpoint: `models/btc_h200/readme_fullfit_ner`.

Assertion full-fit dùng 2.015 dòng cho cả train và validation theo đúng manifest README. Metric của lượt này cũng là fit score vì hai tập trùng nhau.

Kết quả assertion full-fit:

```yaml
runtime: 6m05s
optimizer_updates: 504
train_rows: 2015
validation_rows: 2015
fit_exact: 0.99702
fit_jaccard: 0.99702
thresholds:
  isHistorical: 0.45
  isNegated: 0.10
  isFamily: 0.25
positive_weights:
  isHistorical: 5.5422
  isNegated: 10.2570
  isFamily: 15.0
```

Checkpoint: `models/btc_h200/readme_fullfit_assertion`.

Lưu ý: threshold full-fit thấp hơn threshold holdout và không nên dùng để kết luận khả năng tổng quát hóa, vì 2.015 dòng validation cũng xuất hiện trong train.

### Hướng B: Holdout/generalization

- Split stratified 80 train / 20 validation, seed 42.
- Dùng để so sánh batch, confidence guard, boundary repair và khả năng tổng quát hóa.
- NER tốt nhất hiện tại trước confidence calibration là F1 60,41% sau train-only lexicon repair.
- Assertion SEQ_CLS đạt exact/Jaccard 89,65% trên holdout.

Hai hướng dùng checkpoint, output và report riêng. Không ghép prediction hoặc nhãn giữa hai hướng.

## 8.2 ICD linker: retrieval và XLM-R reranker

ICD linker có hai tầng và ba metric cần báo cáo riêng:

1. Candidate-pool recall của retrieval.
2. Entity accuracy của XLM-R trên pool đã dựng khi train.
3. End-to-end exact accuracy khi retrieval không được ép chứa gold code.

### Tạo training pairs

Cấu hình:

```yaml
gold_dir: ../output/superbest_dataset
input_dir: ../data/input-part2-real/input
context_window_chars: 220
cascade_top_k: 8
dense_top_k: 30
dense_model: BAAI/bge-m3
device: cuda
```

Kết quả:

```yaml
diagnosis_entities: 779
entities_with_icd_label: 398
entities_with_none_label: 381
unique_diagnosis_queries: 369
training_pair_rows: 24797
training_pairs_size: 19 MB
gold_found_by_candidate_pool: 351
gold_missing_from_candidate_pool: 47
candidate_pool_recall: 0.8819
```

`88,19%` là recall của candidate pool trên 398 entity có ICD label, không phải accuracy của XLM-R hoặc toàn pipeline.

Training pairs đã được lưu tại:

`icd-linker-release/data/btc_h200/training_pairs.jsonl`

### Train XLM-R

Cấu hình:

```yaml
base_model: FacebookAI/xlm-roberta-base
epochs: 12
learning_rate: 2.0e-5
candidate_cap: 40
max_length: 192
validation_ratio_by_document: 0.15
meta_batch_entities: 8
seed: 42
train_entities: 670
validation_entities: 109
validation_documents: 15
```

Kết quả tốt nhất ở epoch 11:

```yaml
train_loss: 0.1953
train_entity_accuracy: 0.9448
validation_loss: 1.4139
validation_entity_accuracy: 0.8257
```

Epoch 12 đạt train accuracy 96,42% nhưng validation giảm còn 80,73%, vì vậy checkpoint epoch 11 được giữ theo `val_acc`.

### Đánh giá end-to-end

`evaluate.py` dựng lại candidate pool mà không ép mã gold vào pool, sau đó chạy reranker:

```yaml
all:
  correct: 707
  total: 779
  exact_accuracy: 0.9076
validation_only:
  correct: 88
  total: 109
  exact_accuracy: 0.8073
train_only:
  correct: 619
  total: 670
  exact_accuracy: 0.9239
mistakes: 72
```

Diễn giải metric:

- `92,39% train_only` là ICD fit score trên 670 entity thuộc các document dùng để train XLM-R.
- `80,73% validation_only` là accuracy trên 109 entity thuộc 15 document holdout.
- `90,76% all` là end-to-end ICD accuracy trên toàn bộ 779 gold diagnosis entities của 100 tài liệu.
- Evaluation này dùng entity chuẩn từ silver dataset (`text`, `position`, `type` đúng). Nó chưa đo sai số lan truyền từ Qwen NER.
- Kết quả của `Qwen NER -> assertion -> ICD` chưa được đo; entity thiếu/dư/sai boundary từ Qwen có thể làm candidate score cuối thay đổi.

Do đó, nếu hỏi "ICD fit" thì con số đúng là 92,39% train-only. Nếu hỏi ICD inference trên toàn bộ 100 file nhưng vẫn dùng gold entities thì con số đúng là 90,76%.

Checkpoint hợp lệ trên H200:

`icd-linker-release/models/btc_h200/reranker/best.pt`

Checkpoint và metadata đã được tải về local tại cùng đường dẫn tương đối trong repository:

```text
icd-linker-release/models/btc_h200/reranker/best.pt
icd-linker-release/models/btc_h200/reranker/config.json
icd-linker-release/models/btc_h200/reranker/tokenizer.json
icd-linker-release/models/btc_h200/reranker/tokenizer_config.json
icd-linker-release/models/btc_h200/reranker/val_docs.json
```

Xác minh checkpoint:

```yaml
size_bytes: 1112247488
sha256: be792e09eba60070572abfeb19ac69a80a0e4226f9459614f62674d6d162f7a5
local_remote_hash_match: true
```

Report:

`output/btc_h200/icd/evaluation_report.json`

Report và log đã được tải về local:

```text
output/btc_h200/icd/evaluation_report.json
output/btc_h200/icd/build_pairs.log
output/btc_h200/icd/train_xlmr_v2.log
output/btc_h200/icd/evaluate.log
```

## 9. Lỗi và cảnh báo đã gặp

### 9.1 Thiếu `data/section_cues.yaml`

`build_assertion_dataset.py` lỗi vì `ContextRouter` cần `data/section_cues.yaml`. File cấu hình section cue đã được bổ sung từ source pipeline nội bộ; đây chỉ là lexicon điều hướng context, không chứa nhãn/output dự đoán.

### 9.2 Assertion adapter sai task type

Adapter cũ là CausalLM nhưng inference script là sequence classification. Cách sửa là train adapter mới với `TaskType.SEQ_CLS` và ba output logits.

### 9.3 Batch lớn làm giảm số optimizer update

Tăng batch không đảm bảo chất lượng tăng. Với 96 training window:

```text
batch 32 x 16 epoch = 48 updates
batch 8  x 24 epoch = 288 updates
```

Trên tập nhỏ, số update và cách chọn learning rate quan trọng hơn việc dùng hết VRAM H200.

### 9.4 Tokenizer exact-boundary ceiling

Chỉ 2.682/2.709 entity, tương đương 99,0033%, có boundary biểu diễn chính xác theo tokenizer hiện tại. Khoảng 1% span còn lại cần boundary-aware postprocessing; không được repair bằng cách đọc ground truth của tập cần dự đoán.

### 9.5 Hugging Face warning

Remote chưa có `HF_TOKEN`, nên Hub cảnh báo request không xác thực. Model đã cache đầy đủ nên không ảnh hưởng train hiện tại; khi dựng máy mới nên cấu hình token để tránh rate limit/timeout.

### 9.6 Transformers deprecation warnings

- `torch_dtype` được cảnh báo chuyển sang `dtype`.
- `use_return_dict` được cảnh báo chuyển sang `return_dict`.
- Torch checkpointing cảnh báo cần khai báo rõ `use_reentrant`.

Các cảnh báo này chưa làm job thất bại nhưng nên được sửa trước bản release để tương thích lâu dài với Torch/Transformers mới.

### 9.7 ICD release có file rỗng và Git LFS pointer

Hai artifact trong clone round 2 không sử dụng được:

- `icd-linker-release/src/icd/qualifier.py`: 0 byte.
- `icd-linker-release/data/processed/icd_tt06_vectors.npy`: 133 byte, chỉ là Git LFS pointer thay vì vector index thật.

Đã phục hồi đúng hai file từ bản `icd-linker-release` nguồn tương ứng:

- `qualifier.py`: khoảng 1,4 KB.
- `icd_tt06_vectors.npy`: khoảng 54 MB.

Sau phục hồi, `build_training_data.py --help` và pair builder chạy được. Khi clone trên máy thi phải chạy Git LFS pull và kiểm tra kích thước artifact, không chỉ kiểm tra file có tồn tại.

### 9.8 ICD train loss NaN do forward padding rows

`train.py` ban đầu forward toàn bộ tensor `[meta_batch, cap, max_length]`, gồm các candidate padding có `attention_mask` toàn 0. XLM-R tạo NaN trên hàng bị mask hoàn toàn; NaN lan vào gradient dù logits padding được mask khỏi cross-entropy sau đó.

Biểu hiện của checkpoint lỗi:

```text
train_loss=nan
val_loss=nan
val_acc=0.3028 không đổi
```

Cách sửa:

- Chỉ forward các candidate rows có `cand_mask=True`.
- Scatter valid logits trở lại tensor `[batch, cap]` với `-inf` cho padding.
- Fail-fast khi loss không hữu hạn.
- Dùng `clip_grad_norm_(..., error_if_nonfinite=True)`.

Smoke test sau sửa có loss hữu hạn (`train_loss=2.8756`, `val_loss=2.5409`) và val accuracy epoch 1 là 49,54%. File checkpoint NaN được đổi tên `best.invalid_nan.pt` và tuyệt đối không dùng cho inference.

## 10. Artifact và log hiện có trên H200

```text
models/btc_h200/ner_fold0/                 # NER batch 32, ablation
models/btc_h200/ner_fold0_b8/              # NER batch 8, checkpoint holdout tốt hơn
models/btc_h200/assertion_fold0/            # Assertion holdout SEQ_CLS
models/btc_h200/readme_fullfit_ner/         # README NER full-fit, fit F1 98,03%
models/btc_h200/readme_fullfit_assertion/   # README assertion full-fit, fit 99,70%
icd-linker-release/models/btc_h200/reranker/best.pt
icd-linker-release/data/btc_h200/training_pairs.jsonl
output/btc_h200/ner_fold0_metrics.json
output/btc_h200/ner_fold0_b8_metrics.json
output/btc_h200/ner_fold0_b8_repaired_metrics.json
output/btc_h200/ner_fold0_b8_repair_report.json
output/btc_h200/ner_train.log
output/btc_h200/ner_b8_train.log
output/btc_h200/assertion/train.jsonl
output/btc_h200/assertion/validation.jsonl
output/btc_h200/assertion/train.log
output/btc_h200/assertion/validation_report.json
output/btc_h200/readme_baseline/ner_fit_metrics.json
output/btc_h200/readme_baseline/ner_train.log
output/btc_h200/readme_baseline/ner_infer.log
output/btc_h200/readme_baseline/assertion/train.log
output/btc_h200/readme_baseline/assertion/validation_report.json
output/btc_h200/icd/build_pairs.log
output/btc_h200/icd/train_xlmr_v2.log
output/btc_h200/icd/evaluation_report.json
output/btc_h200/icd/evaluate.log
```

Các artifact quan trọng phải được tải về máy local trước khi instance Vast bị destroy.

Tại lần kiểm tra gần nhất, filesystem H200 dùng khoảng 30/80 GB và còn khoảng 51 GB. `/workspace` không phải Vast persistent volume; stop/start giữ dữ liệu nhưng destroy instance sẽ xóa dữ liệu.

## 11. Các bước tiếp theo

1. Chạy debug inference và confidence calibration của NER holdout để giảm entity dư.
2. Không thay checkpoint README full-fit bằng checkpoint thử nghiệm nếu chưa chứng minh metric tốt hơn.
3. Đưa checkpoint ICD hợp lệ vào canonical path dùng bởi full pipeline.
4. Chạy NER full-fit inference, assertion full-fit inference, ICD và RxNorm thành full pipeline.
5. Chạy schema/span validator và kiểm tra mọi candidates theo type.
6. Tải toàn bộ checkpoint, index, log, report và submission về local.

## 12. Nguyên tắc chống leakage

- Không sao chép entity/assertion/candidate từ output tham chiếu sang prediction.
- Không dùng label directory của tài liệu đang đánh giá trong postprocessing.
- Reference chỉ được mở sau khi inference kết thúc để tính metric.
- Các rule/lexicon được phép dùng phải độc lập với output của từng tài liệu.
- Báo cáo tách rõ holdout F1, full-fit score và leaderboard score.

## 13. Lệnh tái lập trên H200

### 13.1 Kích hoạt môi trường

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /workspace/envs/btc
export HF_HOME=/workspace/.hf_home
export HF_HUB_DISABLE_XET=1
cd /workspace/viettel-ai-race-second-round
```

### 13.2 README NER full-fit

```bash
python scripts/train_qwen_token_ner.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --input data/input-part2-real/input \
  --train output/superbest_dataset \
  --out models/btc_h200/readme_fullfit_ner \
  --split-manifest output/superbest_dataset_split.json \
  --fold 0 \
  --epochs 16 \
  --lr 2e-4 \
  --batch 4 \
  --grad-accum 1 \
  --max-len 768 \
  --stride 192 \
  --lora-r 32
```

Inference và tính fit score:

```bash
python scripts/infer_qwen_token_ner.py \
  --checkpoint models/btc_h200/readme_fullfit_ner \
  --input data/input-part2-real/input \
  --output output/btc_h200/readme_baseline/ner_pred \
  --ids-file output/superbest_submission_ids.json \
  --split validation \
  --evaluate-after-inference output/superbest_dataset \
  --metrics-out output/btc_h200/readme_baseline/ner_fit_metrics.json
```

### 13.3 Assertion full-fit

```bash
python scripts/build_assertion_dataset.py \
  --input data/input-part2-real/input \
  --labels output/superbest_dataset \
  --split-manifest output/superbest_dataset_split.json \
  --train-output output/btc_h200/readme_baseline/assertion/train.jsonl \
  --validation-output output/btc_h200/readme_baseline/assertion/validation.jsonl

python scripts/train_qwen_assertion_classifier.py \
  --base Qwen/Qwen2.5-3B-Instruct \
  --train output/btc_h200/readme_baseline/assertion/train.jsonl \
  --validation output/btc_h200/readme_baseline/assertion/validation.jsonl \
  --output models/btc_h200/readme_fullfit_assertion \
  --report output/btc_h200/readme_baseline/assertion/validation_report.json \
  --epochs 8 \
  --batch-size 32 \
  --grad-accum 1 \
  --max-length 384 \
  --learning-rate 8e-5
```

### 13.4 Holdout NER batch 8

Không truyền manifest full-fit. Script tự tạo split stratified 80/20 bằng `--val-ratio 0.2 --seed 42`.

```bash
python scripts/train_qwen_token_ner.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --input data/input-part2-real/input \
  --train output/superbest_dataset \
  --out models/btc_h200/ner_fold0_b8 \
  --val-ratio 0.2 \
  --seed 42 \
  --epochs 24 \
  --lr 2e-4 \
  --batch 8 \
  --grad-accum 1 \
  --max-len 768 \
  --stride 128 \
  --lora-r 32
```

### 13.5 Confidence calibration NER

Ba script hỗ trợ được giữ độc lập với hành vi mặc định của README:

- `scripts/infer_qwen_token_ner.py --debug-output ...` ghi confidence vào report riêng.
- `scripts/calibrate_qwen_ner_confidence.py` dò metric/ngưỡng riêng theo type trên holdout.
- `scripts/apply_qwen_ner_confidence.py` lọc entity và xóa toàn bộ trường debug trước khi xuất JSON.

Không được dò threshold trên tập submission không có nhãn hoặc dùng nhãn của tài liệu cần dự đoán trong repair.

## 14. Kết luận hiện tại

- H200 chạy đúng môi trường BTC và rút thời gian train mỗi module Qwen xuống khoảng 5-6 phút.
- Pipeline README đã tái lập được NER fit F1 98,03% và assertion fit exact 99,70%.
- Các con số full-fit không đại diện cho generalization; holdout NER hiện chỉ đạt 60,41% sau repair.
- ICD retrieval đạt candidate-pool recall 88,19%; XLM-R best val accuracy 82,57%; end-to-end holdout accuracy 80,73%.
- ICD train-only fit là 92,39% và end-to-end trên toàn bộ 779 gold entities là 90,76%; chưa phải metric sau Qwen NER.
- Batch quá lớn không phù hợp với NER dataset nhỏ nếu không bù đủ optimizer update.
- Full inference NER -> assertion -> ICD -> RxNorm đã hoàn tất 100 file và ZIP đã được tải về local.
- Confidence calibration holdout vẫn là nhánh thử nghiệm tùy chọn, không thay đổi baseline full-flow vừa đóng gói.

## 15. Full-flow inference trên H200

Full flow hợp lệ đã chạy theo thứ tự:

```text
Qwen NER full-fit
  -> Qwen assertion SEQ_CLS full-fit
  -> ICD retrieval + XLM-R best.pt
  -> RxNorm lexical linker
  -> schema/span validator
  -> deterministic ZIP packager
```

Không chạy boundary repair sử dụng silver labels trong full flow này. ICD và RxNorm chỉ sửa trường `candidates`; `text`, `position`, `type` và `assertions` được bảo toàn từ output Qwen.

### Qwen NER và assertion inference

```yaml
files: 100
entities: 2720
assertable_entities_classified: 2017
assertion_thresholds:
  isHistorical: 0.45
  isNegated: 0.10
  isFamily: 0.25
assertion_counts:
  empty: 1512
  isHistorical: 302
  isNegated: 170
  isFamily: 24
  isHistorical_and_isNegated: 9
```

Fit check trước linker:

```yaml
span_type_text_f1: 0.98029
matched_assertable_rows: 1978
assertion_exact: 0.99646
assertion_jaccard: 0.99646
```

### ICD inference trên Qwen entities

```yaml
files: 100
entities: 2720
diagnoses: 775
diagnoses_with_icd: 346
diagnoses_predicted_none: 429
```

Đây là bước áp checkpoint đã train lên 100 JSON Qwen, không phải train lại ICD. Tiến độ dạng `20/100`, `49/100` là số file inference đã xử lý.

### RxNorm inference

```yaml
medication_entities: 268
medications_with_rxnorm: 198
medications_without_rxnorm: 70
top_k: 1
strategy: most_specific
context_loaded: true
```

RxNorm không train neural model; module đọc `rxnorm_index.pkl` và xếp hạng theo lexical/ingredient/strength/form/route heuristics.

Đánh giá candidate RxNorm của output full pipeline so với `output/superbest_dataset`, chỉ trên các thực thể `THUỐC` trùng chính xác `position` và `type`:

```yaml
prediction_medications: 268
reference_medications: 269
exact_matched_medications: 262
candidate_true_positive: 167
candidate_predicted: 198
candidate_reference: 215
micro_precision: 84.34%
micro_recall: 77.67%
micro_f1: 80.87%
macro_jaccard: 80.15%
exact_candidate_set_accuracy: 80.15%
```

`micro_f1` đo trực tiếp tập RxCUI. `macro_jaccard` gần với cách nhìn của chỉ số candidate hơn, nhưng đây vẫn là phép so sánh local với silver/superbest, không phải điểm leaderboard chính thức.

### Final output và validator

```yaml
json_files: 100
entities: 2720
entities_with_candidates: 544
candidate_codes: 544
diagnoses_with_candidates: 346
medications_with_candidates: 198
schema_or_span_errors: 0
```

ZIP có đúng `1.json` đến `100.json` tại root, không có thư mục bọc ngoài:

```text
submission/btc_h200_full_pipeline.zip
```

Xác minh ZIP local:

```yaml
size_bytes: 86388
json_files: 100
sha256: a380b1e4516432a1e697c97a99fd447e30cc3aceec829ca8e89bdf248d42d0d6
```

## 16. Lỗi full-flow inference đã sửa

### 16.1 RxNorm index phụ thuộc working directory

Lệnh README chạy từ repo root nhưng `build_index.py` ban đầu định nghĩa:

```python
CACHE_PATH = Path("data/rxnorm/cache/rxnorm_index.pkl")
```

Vì vậy linker không thấy index thật trong `rxnorm-linker/data/...` và cố rebuild từ `RXNCONSO.RRF`, artifact không được đóng gói.

Cách sửa:

- Resolve `PROJECT_ROOT` từ `Path(__file__).resolve()`.
- Đặt `CACHE_PATH` và `RRF_PATH` tuyệt đối theo root của `rxnorm-linker`.
- Smoke-test lại lệnh từ repository root; context và index đều load thành công.

### 16.2 Inline ZIP script bị lỗi quoting qua SSH

Linker và validator đã thành công, nhưng inline Python truyền qua PowerShell/SSH làm mất quote trong f-string. Đã thay bằng script repository:

`scripts/package_submission.py`

Script kiểm tra đủ ID, từ chối JSON dư, thống kê candidates, ghi ZIP theo thứ tự và xác nhận member nằm ở ZIP root.
