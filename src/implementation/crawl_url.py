import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from langchain_core.documents import Document
from src.ingest_url import create_chunks, create_embeddings


async def crawl_single_url(crawler, url, config):
    """Crawl một URL và trả về nội dung markdown."""
    result = await crawler.arun(url=url, config=config)
    if result.success:
        return {
            "url": url,
            "markdown": result.markdown_v2.raw_markdown if result.markdown_v2 else result.markdown,
            "title": result.metadata.get("title", "") if result.metadata else "",
        }
    else:
        print(f"[LỖI] Không crawl được: {url} - {result.error_message}")
        return None


async def crawl_urls(urls: list[str]) -> list[Document]:
    """
    Crawl danh sách URLs bằng crawl4ai và chuyển đổi sang LangChain Document.
    """
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,     # Không dùng cache, luôn crawl mới
        word_count_threshold=50,          # Bỏ qua block có ít hơn 50 từ (quảng cáo, menu...)
        exclude_external_links=True,      # Loại bỏ link ngoài
        remove_overlay_elements=True,     # Xóa popup, overlay
    )

    documents = []
    async with AsyncWebCrawler() as crawler:
        for url in urls:
            print(f"[CRAWL] Đang crawl: {url}")
            data = await crawl_single_url(crawler, url, config)
            if data and data["markdown"].strip():
                doc = Document(
                    page_content=data["markdown"],
                    metadata={
                        "source": data["url"],
                        "title": data["title"],
                    }
                )
                documents.append(doc)
                print(f"[OK] Đã crawl thành công: {url} ({len(data['markdown'])} ký tự)")

    print(f"\n[TỔNG KẾT] Crawl thành công {len(documents)}/{len(urls)} URLs")
    return documents


def crawl_and_ingest(urls: list[str]):
    """
    Pipeline hoàn chỉnh: Crawl URLs → Chunk → Embedding → Ensemble Retriever.
    Đồng bộ với ingest_url.py (dùng chung create_chunks & create_embeddings).
    """
    # Bước 1: Crawl URLs
    docs = asyncio.run(crawl_urls(urls))
    if not docs:
        print("[CẢNH BÁO] Không có dữ liệu nào được crawl thành công!")
        return None

    # Bước 2: Chia nhỏ văn bản thành chunks (dùng SemanticChunker từ ingest_url)
    print(f"\n[CHUNK] Đang chia {len(docs)} tài liệu thành các đoạn nhỏ...")
    chunks = create_chunks(docs)
    print(f"[CHUNK] Tạo được {len(chunks)} chunks")

    # Bước 3: Tạo embeddings và ensemble retriever (dùng Milvus + BM25 từ ingest_url)
    print(f"\n[EMBED] Đang tạo embeddings và retriever...")
    ensemble_retriever = create_embeddings(chunks)
    print(f"[EMBED] Ensemble retriever đã sẵn sàng!")

    return ensemble_retriever


# Chạy trực tiếp để test
if __name__ == "__main__":
    test_urls = [
        # Thêm URLs cần crawl vào đây
        "https://example.com",
    ]
    retriever = crawl_and_ingest(test_urls)
    if retriever:
        # Test thử truy vấn
        results = retriever.invoke("test query")
        for doc in results:
            print(f"\n--- Source: {doc.metadata.get('source', 'N/A')} ---")
            print(doc.page_content[:200])
