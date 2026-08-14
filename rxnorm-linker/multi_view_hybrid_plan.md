# RxNorm Multi-View Corpus + Hybrid Scoring Plan

## Mục tiêu

Nâng cấp pipeline RxNorm linking để:

1. Mỗi RXCUI được biểu diễn bởi nhiều view text, không còn chỉ một STR đại diện.
2. Embedding của mỗi RXCUI được tạo bằng average pooling trên nhiều view: STR chính, brand names, synonyms, và các biến thể normalized.
3. Cache có token -> index mapping để hỗ trợ lexical scoring và reranking nhanh.
4. Query được mở rộng trước khi embed bằng aliases và normalized variants.
5. Reranking cuối dùng hybrid scoring giữa embedding cosine similarity và token overlap.
6. Toàn bộ tham số phải là config-driven, có thể tune lại sau khi evaluate, không hard-code cứng trong logic.

## Nguyên tắc thiết kế

- Giữ pipeline deterministic làm nền, embedding chỉ là một thành phần trong rerank/fallback.
- Không dùng một representative string duy nhất cho RXCUI nữa.
- Không để expansion sinh quá nhiều noise; mọi expansion đều phải có cap và có thể điều chỉnh.
- Không cố định trọng số hoặc threshold trong code logic; đưa vào config hoặc hằng số tập trung.
- Mọi thay đổi phải tương thích với cache cũ theo hướng rebuild khi schema thay đổi.

## Hiện trạng code cần bám vào

Các điểm chạm chính trong repo:

- [src/rxnorm/normalize.py](src/rxnorm/normalize.py): parse span, token normalize, alias map, form hints.
- [src/rxnorm/query_expansion.py](src/rxnorm/query_expansion.py): đang là stub docstring, có thể biến thành module thật cho query views.
- [src/rxnorm/build_index.py](src/rxnorm/build_index.py): build cache từ RXNCONSO.RRF.
- [src/rxnorm/embed_index.py](src/rxnorm/embed_index.py): build và load embedding cache.
- [src/rxnorm/linker.py](src/rxnorm/linker.py): scoring, candidate generation, fallback.
- [scripts/build_rxnorm_index.py](scripts/build_rxnorm_index.py): entrypoint build cache.
- [tests/test_rxnorm_linker.py](tests/test_rxnorm_linker.py): regression test hiện có.

## Phạm vi thay đổi

### 1. Chuẩn hóa cấu hình

Tạo một lớp cấu hình trung tâm cho toàn pipeline, ví dụ:

- max number of views per RXCUI
- max number of query expansions
- max candidate pool before rerank
- lexical weight
- embedding weight
- exact match bonus
- token overlap bonus
- penalties cho mismatch về form/strength/salt variant
- minimum score để giữ candidate
- minimum cosine similarity để chấp nhận embedding hit
- collapse threshold cho top-1 confidence

Các tham số này không được cố định trong hàm scoring. Mặc định chỉ là seed ban đầu để chạy thử, sau đó tune bằng eval.

### 2. Multi-view corpus cho RXCUI

Thay vì encode một string đại diện cho mỗi RXCUI, build nhiều view text từ nhiều nguồn:

- STR canonical / preferred string.
- Brand names.
- Synonyms.
- Các STR đã normalize.
- Các variant sinh từ alias map và token normalization.

Yêu cầu:

- Dedupe các view sau normalize.
- Loại bỏ view quá ngắn hoặc quá nhiễu theo rule config.
- Giới hạn số view tối đa mỗi RXCUI để tránh cache phình.
- Giữ source tag cho từng view để sau này có thể weight khác nhau nếu cần.

Cách encode:

- Encode từng view riêng.
- Average pooling các vector đã normalize.
- Normalize lại vector cuối cùng.
- Nếu cần, cho phép weighted average theo source tag, nhưng weight phải là config, không hard-code.

### 3. Token index trong cache

Cache mới cần thêm token -> indices mapping, có thể là:

- token -> entry indices
- hoặc token -> concept ids nếu muốn dedupe sớm hơn

Khuyến nghị:

- Lưu token index cùng metadata đủ để rerank lexical nhanh.
- Có thể thêm document frequency / frequency statistics để lexical score có IDF-like signal về sau.
- Token normalization của corpus và query phải dùng cùng chuẩn.

Mục tiêu của token index:

- Candidate retrieval nhanh.
- Tính token overlap cho rerank.
- Hỗ trợ exact / near-exact lexical signal mà không cần scan toàn bộ cache.

### 4. Query expansion trước khi embed

`query_expansion.py` nên trở thành module thật, trả về một tập view text có cấu trúc rõ ràng.

Views query nên gồm:

- raw query
- lowercase variant
- accent-stripped variant
- alias-expanded variant
- normalized token variant
- synonym-expanded variant nếu có match từ corpus

Yêu cầu:

- Expansion phải có cap.
- Expansion phải dedupe.
- Expansion phải có source tag.
- Nếu expansion tạo ra quá nhiều noise, phải có cách giảm weight hoặc bỏ qua bằng config.

Điểm quan trọng:

- Query expansion phải dùng cùng alias/normalization logic với corpus.
- Không expand vô hạn theo tất cả synonym có thể có.
- Nếu query đã đủ rõ, expansion chỉ là tăng robustness chứ không làm loãng tín hiệu gốc.

### 5. Hybrid scoring cho rerank

Thay scoring hiện tại bằng score tổng hợp giữa:

- embedding cosine similarity
- token overlap / containment
- exact alias or normalized match bonus
- penalty cho mismatch mạnh về form / salt / strength

Khuyến nghị workflow:

1. Lấy candidate pool từ lexical.
2. Lấy candidate pool từ embedding.
3. Union hai pool.
4. Rerank bằng hybrid score.

Yêu cầu:

- Không hard-code weight.
- Trọng số có thể tune lại bằng config hoặc file tham số.
- Giữ các rule an toàn hiện có như strength consistency và form penalty, nhưng cho vào tổng score thay vì ràng buộc rời rạc nếu có thể.

### 6. Fallback strategy

Embedding không được thay thế toàn bộ lexical pipeline.

Chiến lược đề xuất:

- Lexical vẫn là nguồn chính.
- Embedding chỉ bổ trợ cho typo, synonym, spelling variant, OCR-ish glued tokens.
- Nếu lexical đã đủ confident, không cần đẩy embedding vào quá sớm.
- Nếu lexical thiếu ứng viên, mới mở rộng pool bằng embedding.

### 7. Cache versioning và backward compatibility

Vì schema cache sẽ đổi, cần:

- Bump cache version.
- Lưu metadata version trong cache.
- Khi load cache cũ không tương thích, tự rebuild hoặc báo rõ.
- Không silently đọc sai schema cũ.

### 8. Test và evaluation

Cần bổ sung test cho từng phần sau:

- Query expansion: alias map, normalized variants, dedupe, cap số view.
- Multi-view pooling: cùng RXCUI ra vector ổn định, không phụ thuộc thứ tự view.
- Token index: candidate retrieval đúng và không mất entry quan trọng.
- Hybrid scoring: lexical thắng khi exact/near-exact, embedding cứu typo và synonym.
- Regression trên bộ span hiện có trong `tests/test_rxnorm_linker.py`.

Chỉ số nên theo dõi:

- recall@1
- recall@3
- exact RXCUI hit rate
- số false positive trên span một từ
- coverage trên các span typo / synonym / brand

## Thứ tự triển khai khuyến nghị

### Phase 1: Scaffold config và schema cache

- Tạo config object hoặc module config trung tâm.
- Định nghĩa schema cache version mới.
- Chốt danh sách metadata cần lưu trong cache.

### Phase 2: Build multi-view corpus

- Viết hàm tạo views cho RXCUI từ các STR/brand/synonym/normalized variants.
- Deduplicate views.
- Giới hạn số view tối đa.

### Phase 3: Multi-view embedding

- Encode từng view.
- Average pooling.
- Save vector theo RXCUI.
- Cập nhật embed index loader.

### Phase 4: Token index

- Lưu token -> index mapping trong cache.
- Thêm thống kê hỗ trợ lexical scoring nếu cần.
- Đảm bảo query tokenization dùng cùng chuẩn.

### Phase 5: Query expansion

- Biến `query_expansion.py` thành module thực.
- Thêm raw / normalized / alias-expanded / synonym-expanded views.
- Dedupe và cap expansion.

### Phase 6: Hybrid rerank

- Tính lexical score và embedding score riêng.
- Rerank bằng tổng hợp có trọng số.
- Thêm penalty/bonus có thể tune.

### Phase 7: Validation

- Chạy regression tests.
- So sánh baseline và bản mới trên sample thực tế.
- Tune tham số từ kết quả eval, không chốt cứng trước khi đo.

## Acceptance criteria

- Cache build xong có multi-view embeddings và token index.
- Query expansion chạy được và có test.
- Rerank hybrid hoạt động ổn định, không làm hỏng baseline rõ rệt.
- Các tham số chính đều có thể chỉnh lại mà không sửa logic lõi.
- Bộ test regression chính vẫn pass hoặc có giải thích rõ nếu thay đổi làm lệch một số case.

## Ghi chú triển khai cho coding agent

- Ưu tiên sửa `query_expansion.py` và `build_index.py` trước, vì đây là nền dữ liệu cho cả embedding lẫn rerank.
- Không làm ngay các tối ưu phức tạp nếu chưa có benchmark; trước hết phải đúng và có test.
- Nếu phát hiện cache cũ không tương thích, tăng version và rebuild thay vì cố đọc ngầm.
- Khi tune trọng số, giữ code cấu hình hóa để có thể thay đổi lại sau khi đánh giá.
