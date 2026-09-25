# Individual contribution report

Copy file này thành `<student-id>-<short-name>.md` và chỉ ghi đóng góp có thể đối chiếu.

## Thông tin

- Họ và tên: Trần Tuấn Cường
- Mã học viên:2A202602717
- Nhóm: T-149
- Repository/branch: https://github.com/CuongTT-04/K4-L3B-RAG-Pipeline / `main`

## Phần việc đã thực hiện

| Module/deliverable | Việc trực tiếp làm | File/commit/PR | Trạng thái |
| --- | --- | --- | --- |
| Thu thập dữ liệu | Xây dựng manifest và tải 3 PDF chính thức; crawl 5 trang European Commission, kiểm tra metadata và tính hợp lệ của dữ liệu nguồn | `src/task1_collect_legal_docs.py`, `src/task2_crawl_news.py`, `data/landing/` | Done |
| Chuẩn hóa dữ liệu | Dùng MarkItDown để chuyển PDF sang Markdown, thêm front matter lưu title/source/doc_type/URL; chuẩn hóa JSON bài viết | `src/task3_convert_markdown.py`, `data/standardized/` | Done |
| Chunking, embedding, vector database | Thiết kế chunk ID ổn định, recursive chunking 800/100, hashing embedding 384 chiều và ChromaDB cosine upsert idempotent | `src/task4_chunking_indexing.py` | Done |
| Hybrid retrieval | Triển khai semantic search, BM25 và Reciprocal Rank Fusion; bảo đảm sort, deduplicate và schema SearchResult | `src/task5_semantic_search.py`, `src/task6_lexical_search.py`, `src/task7_reranking.py` | Done |
| Fallback và retrieval pipeline | Triển khai vectorless section fallback, dùng dense cosine score gốc cho threshold và chỉ fuse RRF một lần | `src/task8_pageindex_vectorless.py`, `src/task9_retrieval_pipeline.py` | Done |
| Generation và citation | Tích hợp Mistral/OpenAI/Gemini/Anthropic, local fallback, safe refusal, reorder context và citation `[S1]` ánh xạ đúng sources | `src/task10_generation.py`, `.env.example` | Done |
| Chatbot UI | Nối pipeline vào Streamlit, lưu lịch sử chat, hiển thị nguồn, URL, retrieval method và score | `app.py` | Done |
| Evaluation | Xây dựng 15 golden cases, evaluator 4 metrics, A/B dense-only với hybrid + RRF và phân tích lỗi | `src/evaluate.py`, `group_project/evaluation/golden_dataset.json`, `group_project/evaluation/RESULT.md` | Done |
| Kiểm thử và tài liệu | Sửa các edge case theo module contract, chạy toàn bộ 20 tests, bổ sung hướng dẫn tái lập và cấu hình Mistral | `tests/`, `README.md`; working tree trên base commit `a23df34` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Kết hợp dense retrieval và BM25 bằng RRF, đồng thời dùng dense cosine score gốc để quyết định fallback.  
   **Lý do/evidence:** Dense-only đạt average 0.602; hybrid + RRF đạt 0.696, context recall tăng 0.150 và context precision tăng 0.120 trên cùng 15 golden cases. RRF tránh cộng trực tiếp hai thang điểm cosine và BM25 không tương thích.  
   **Trade-off:** Hybrid cần xây thêm BM25 index và tốn thêm CPU, nhưng không phát sinh API cost; RRF score chỉ dùng xếp hạng, không thể dùng làm confidence score.

2. **Quyết định:** Duy trì đường chạy offline bằng hashing embeddings và extractive generation, nhưng hỗ trợ Mistral làm generator chính khi có key.  
   **Lý do/evidence:** Pipeline, test và demo vẫn chạy khi provider lỗi hoặc chưa có quota; Mistral được gọi qua endpoint chính thức và `.env` không bị Git theo dõi.  
   **Trade-off:** Hashing/extractive baseline dễ tái lập nhưng answer relevance thấp hơn LLM và semantic quality kém hơn sentence-transformer; API Mistral phụ thuộc rate limit/quota bên ngoài.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `pytest -q`; chạy index hai lần để kiểm tra idempotency; query in-domain “What are the three chapters of the GPAI Code of Practice?”; query ngoài domain “Who won the 2018 football World Cup?”; API probe `MISTRAL_API_OK`.
- Kết quả trước/sau nếu có: Từ skeleton có 10 module `NotImplemented`, dữ liệu/evaluation rỗng và UI chưa tích hợp thành pipeline 151 chunks; toàn bộ **20/20 tests passed**. Hybrid + RRF tăng average từ 0.602 lên 0.696. Query in-domain lấy đúng ba chương và citation; query ngoài domain trả safe refusal với `retrieval_source="none"`.
- Lỗi đã phát hiện và cách xử lý: Sửa Chroma mock không có `count()`, BM25 corpus nhỏ có score 0, citation bị lệch sau context reorder, source list mất thứ tự score và false-positive “World” ở query ngoài domain. API Mistral đã đọc đúng key/model/endpoint nhưng lần kiểm tra ngày 25/09/2026 trả HTTP 429 code 1300 `Rate limit exceeded`; pipeline tự fallback sang local generator thay vì làm UI crash.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: Evaluation hiện dùng evaluator lexical offline để bảo đảm tái lập; vectorless fallback chạy cục bộ theo heading/section; Mistral free-tier có thể bị giới hạn request nên chưa đo được chất lượng generation ổn định trên toàn bộ golden set.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: Chờ/reset quota Mistral rồi chạy lại đủ 15 câu với LLM generation và RAGAS judge, đồng thời hiệu chỉnh threshold trên ít nhất 10 query ngoài domain.

## Xác nhận đóng góp

- Ngày: 25/09/2026
- Tên thành viên: Trần Tuấn Cường
