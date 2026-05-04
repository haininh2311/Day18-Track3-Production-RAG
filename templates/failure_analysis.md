# Failure Analysis — Lab 18

**Nhóm:** Mình tôi  
**Thành viên:** Nguyễn Trần Hải Ninh

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.9492 | 0.9389 | -0.0103 |
| Answer Relevancy | 0.7886 | 0.7727 | -0.0159 |
| Context Precision | 0.8139 | 0.9137 | +0.0998 |
| Context Recall | 0.8611 | 0.8590 | -0.0021 |

## Bottom-5 Failures

*(Do phiên bản RAGAS mới thay đổi tên trường dữ liệu nên câu hỏi cụ thể không được lưu lại trong report, dưới đây là phân tích dựa trên các chỉ số bị lỗi nặng nhất từ hệ thống)*

### #1
- **Question:** [Bị ẩn trong report]
- **Expected:** [Câu trả lời đúng]
- **Got:** [Câu trả lời do hệ thống sinh ra]
- **Worst metric:** Faithfulness (Điểm: 0.0)
- **Error Tree:** Output sai → Context đúng? → Query OK? → Root cause: LLM hallucinating — câu trả lời không dựa trên context.
- **Suggested fix:** Tighten prompt, lower temperature, thêm instruction 'Chỉ trả lời dựa trên context'.

### #2
- **Question:** [Bị ẩn trong report]
- **Expected:** [Câu trả lời đúng]
- **Got:** [Câu trả lời do hệ thống sinh ra]
- **Worst metric:** Context Recall (Điểm: 0.0)
- **Error Tree:** Output sai → Context đúng? (SAI) → Root cause: Missing relevant chunks — retrieval bỏ sót thông tin quan trọng.
- **Suggested fix:** Improve chunking strategy, tăng top_k, thêm BM25 hoặc query expansion.

### #3
- **Question:** [Bị ẩn trong report]
- **Expected:** [Câu trả lời đúng]
- **Got:** [Câu trả lời do hệ thống sinh ra]
- **Worst metric:** Context Recall (Điểm: 0.1111)
- **Error Tree:** Output sai → Context đúng? (SAI) → Root cause: Missing relevant chunks — retrieval bỏ sót thông tin quan trọng.
- **Suggested fix:** Improve chunking strategy, tăng top_k, thêm BM25 hoặc query expansion.

### #4
- **Question:** [Bị ẩn trong report]
- **Expected:** [Câu trả lời đúng]
- **Got:** [Câu trả lời do hệ thống sinh ra]
- **Worst metric:** Answer Relevancy (Điểm: 0.0)
- **Error Tree:** Output lạc đề → Root cause: Answer doesn't match question — câu trả lời không giải quyết trực tiếp câu hỏi.
- **Suggested fix:** Improve prompt template, thêm explicit instruction về format câu trả lời.

### #5
- **Question:** [Bị ẩn trong report]
- **Expected:** [Câu trả lời đúng]
- **Got:** [Câu trả lời do hệ thống sinh ra]
- **Worst metric:** Answer Relevancy (Điểm: 0.0)
- **Error Tree:** Output lạc đề → Root cause: Answer doesn't match question — câu trả lời không giải quyết trực tiếp câu hỏi.
- **Suggested fix:** Improve prompt template, thêm explicit instruction về format câu trả lời.

## Case Study (presentation)

**Question:** [Chọn một câu hỏi cụ thể bạn nhớ bị sai để làm case study]

**Error Tree walkthrough:**
1. Output đúng? → Sai
2. Context đúng? → Thiếu thông tin
3. Query rewrite OK? → Truy vấn tốt nhưng do dữ liệu chưa được làm sạch
4. Fix ở bước: Bước Retrieval (Chunking lại hoặc tăng Top K)

**Nếu có thêm 1 giờ:**
- Sửa lỗi tương thích của thư viện RAGAS để hiển thị chi tiết câu hỏi bị sai.
- Chạy thử nghiệm với Flashrank hoặc mô hình reranker mạnh hơn.
- Áp dụng kỹ thuật Self-Reflect LLM để câu trả lời bớt đi phần "ảo giác" (hallucination).
