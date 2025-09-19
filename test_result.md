tail -50 logs/app.log
2025-09-19 11:37:16 | INFO | __main__:__call__:233 - Request [9febe320] started: POST /query
2025-09-19 11:37:16 | INFO | __main__:handle_query:651 - Processing query: 测试系统响应能力
2025-09-19 11:37:16 | INFO | core.query_engine:process_query:337 - Processing query: 测试系统响应能力
2025-09-19 11:37:16 | WARNING | core.api_client:make_request:199 - Request failed: 401 - "Invalid token"
2025-09-19 11:37:16 | WARNING | core.api_client:make_request:199 - Request failed: 401 - "Invalid token"
2025-09-19 11:37:16 | WARNING | core.api_client:make_request:199 - Request failed: 401 - "Invalid token"
2025-09-19 11:37:16 | INFO | core.api_client:make_request_with_fallback:248 - 尝试使用备用提供商: zhipu
2025-09-19 11:37:16 | WARNING | core.api_client:_create_fallback_client:447 - 备用提供商 zhipu 未配置API密钥
2025-09-19 11:37:16 | WARNING | core.api_client:make_request_with_fallback:265 - 所有策略失败，返回默认响应
2025-09-19 11:37:16 | ERROR | core.query_engine:analyze_query:95 - Query analysis failed: Expecting value: line 1 column 1 (char 0)
2025-09-19 11:37:16 | INFO | core.query_engine:decompose_query:301 - Query decomposed into 1 sub-queries
2025-09-19 11:37:16 | INFO | core.query_engine:determine_routing:236 - Routing decision: entity_search
2025-09-19 11:37:17 | INFO | core.retrieval_engine:search:295 - Vector search returned 0 results
2025-09-19 11:37:17 | INFO | __main__:__call__:254 - Request [9febe320] completed: 200 in 1.133s
2025-09-19 11:37:43 | INFO | __main__:__call__:233 - Request [be242a5e] started: GET /health
2025-09-19 11:37:43 | INFO | __main__:__call__:254 - Request [be242a5e] completed: 200 in 0.001s
2025-09-19 11:53:34 | INFO | __main__:__call__:233 - Request [560e2aeb] started: POST /upload
2025-09-19 11:53:34 | INFO | __main__:__call__:254 - Request [560e2aeb] completed: 400 in 0.001s
2025-09-19 11:54:26 | INFO | __main__:__call__:233 - Request [f9117d4f] started: POST /query
2025-09-19 11:54:26 | INFO | __main__:handle_query:651 - Processing query: 根据联邦制药（03933.HK）的公司发展历程，请简述其在2023年的重大产品临床进展
2025-09-19 11:54:26 | INFO | core.query_engine:process_query:337 - Processing query: 根据联邦制药（03933.HK）的公司发展历程，请简述其在2023年的重大产品临床进展
2025-09-19 11:54:26 | WARNING | core.api_client:make_request:199 - Request failed: 401 - "Invalid token"
2025-09-19 11:54:26 | WARNING | core.api_client:make_request:199 - Request failed: 401 - "Invalid token"
2025-09-19 11:54:26 | WARNING | core.api_client:make_request:199 - Request failed: 401 - "Invalid token"
2025-09-19 11:54:26 | INFO | core.api_client:make_request_with_fallback:248 - 尝试使用备用提供商: zhipu
2025-09-19 11:54:26 | WARNING | core.api_client:_create_fallback_client:447 - 备用提供商 zhipu 未配置API密钥
2025-09-19 11:54:26 | WARNING | core.api_client:make_request_with_fallback:265 - 所有策略失败，返回默认响应
2025-09-19 11:54:26 | ERROR | core.query_engine:analyze_query:95 - Query analysis failed: Expecting value: line 1 column 1 (char 0)
2025-09-19 11:54:26 | INFO | core.query_engine:decompose_query:301 - Query decomposed into 1 sub-queries
2025-09-19 11:54:26 | INFO | core.query_engine:determine_routing:236 - Routing decision: entity_search
2025-09-19 11:54:27 | INFO | core.retrieval_engine:search:295 - Vector search returned 0 results
2025-09-19 11:54:27 | INFO | __main__:__call__:254 - Request [f9117d4f] completed: 200 in 0.916s
2025-09-19 11:57:51 | INFO | __main__:__call__:233 - Request [7829d16c] started: POST /upload
2025-09-19 11:57:52 | INFO | __main__:__call__:254 - Request [7829d16c] completed: 200 in 0.003s
2025-09-19 11:57:52 | INFO | __main__:process_document_background:810 - Processing document: test_doc.pdf
2025-09-19 11:57:52 | INFO | core.document_processor:preprocess_document:632 - Starting multi-modal preprocessing: /tmp/tmprzkffbc8.pdf
2025-09-19 11:57:52 | INFO | core.document_processor:process_pdf:72 - Processing PDF: /tmp/tmprzkffbc8.pdf
2025-09-19 11:57:56 | WARNING | core.api_client:analyze_image:554 - 视觉模型调用失败，返回默认响应
2025-09-19 11:57:56 | INFO | core.api_client:save_checkpoint:106 - Checkpoint saved: pdf_5a916b9b
2025-09-19 11:57:56 | INFO | core.document_processor:process_pdf:114 - PDF processing completed: 1 chunks, 0 tables, 1 charts
2025-09-19 11:57:56 | INFO | core.api_client:remove_checkpoint:123 - Checkpoint removed: pdf_5a916b9b
2025-09-19 11:57:56 | ERROR | __main__:process_document_background:838 - Background processing failed: FOREIGN KEY constraint failed
2025-09-19 12:00:13 | INFO | __main__:__call__:233 - Request [b9932ec3] started: GET /
2025-09-19 12:00:13 | INFO | __main__:__call__:254 - Request [b9932ec3] completed: 200 in 0.001s
2025-09-19 12:25:44 | INFO | __main__:__call__:233 - Request [6a90c32b] started: POST /query
2025-09-19 12:25:44 | INFO | __main__:process_query:565 - Returning cached result
2025-09-19 12:25:44 | INFO | __main__:__call__:254 - Request [6a90c32b] completed: 200 in 0.003s
2025-09-19 12:28:53 | INFO | __main__:__call__:233 - Request [fd9a0d39] started: POST /query
2025-09-19 12:28:53 | INFO | __main__:process_query:565 - Returning cached result
2025-09-19 12:28:53 | INFO | __main__:__call__:254 - Request [fd9a0d39] completed: 200 in 0.001s


tail -20 logs/error.log
2025-09-19 11:32:13 | ERROR | core.query_engine:analyze_query:95 - Query analysis failed: Expecting value: line 1 column 1 (char 0)

2025-09-19 11:37:16 | ERROR | core.query_engine:analyze_query:95 - Query analysis failed: Expecting value: line 1 column 1 (char 0)

2025-09-19 11:54:26 | ERROR | core.query_engine:analyze_query:95 - Query analysis failed: Expecting value: line 1 column 1 (char 0)

2025-09-19 11:57:56 | ERROR | __main__:process_document_background:838 - Background processing failed: FOREIGN KEY constraint failed


ls -la data/
ls -la data/chromadb/
ls -la data/pdfs/
total 88
drwxr-xr-x  6 root root  4096 Sep 19 11:57 .
drwxr-xr-x 12 root root  4096 Sep 19 10:36 ..
drwxr-xr-x  2 root root  4096 Sep 19 09:03 cache
drwxr-xr-x  3 root root  4096 Sep 19 11:32 chromadb
-rw-r--r--  1 root root 36864 Sep 19 10:37 financerag.db
-rw-r--r--  1 root root 16928 Sep 19 11:57 financerag.db-journal
drwxr-xr-x  2 root root  4096 Sep 19 11:57 pdfs
-rw-r--r--  1 root root   915 Sep 19 08:52 README.md
drwxr-xr-x  2 root root  4096 Sep 19 09:03 sqlite
total 160
drwxr-xr-x 3 root root   4096 Sep 19 11:32 .
drwxr-xr-x 6 root root   4096 Sep 19 11:57 ..
drwxr-xr-x 2 root root   4096 Sep 19 11:32 6221958e-a1ad-4d7c-bd27-585008e33fd0
-rw-r--r-- 1 root root 147456 Sep 19 10:41 chroma.sqlite3
total 268
drwxr-xr-x 2 root root   4096 Sep 19 11:57 .
drwxr-xr-x 6 root root   4096 Sep 19 11:57 ..
-rw-r--r-- 1 root root 263961 Sep 19 11:57 test_doc.pdf


python -c "
  import os
  from dotenv import load_dotenv
  load_dotenv()
  print('SILICONFLOW_API_KEY:', 'Yes' if os.getenv('SILICONFLOW_API_KEY') else 'No')
  print('OPENAI_API_KEY:', 'Yes' if os.getenv('OPENAI_API_KEY') else 'No')
  "
  File "<string>", line 2
    import os
IndentationError: unexpected indent


curl -X GET "https://域名/documents/12be648dca1a5c4cc9f24343b00c2af0"
{"detail":"Not Found"}# 


