from llmwiki_core.chunking import MAX_CHUNK_CHARS, chunk_pages, chunk_text


def test_cjk_chunks_preserve_page_breadcrumb_and_overlap():
    paragraphs = ["数据本地化要求。" * 20 for _ in range(20)]
    content = "# 印尼合规\n\n" + "\n\n".join(paragraphs)
    chunks = chunk_text(content, chunk_size=128, overlap=64, page=7)
    assert len(chunks) > 1
    assert all(chunk.page == 7 for chunk in chunks)
    assert all(len(chunk.content) <= MAX_CHUNK_CHARS for chunk in chunks)
    assert chunks[0].header_breadcrumb == "印尼合规"
    assert chunks[1].start_char < chunks[0].start_char + len(chunks[0].content)


def test_chunk_pages_assigns_global_indexes():
    chunks = chunk_pages([(1, "A sentence. " * 200), (2, "B sentence. " * 200)])
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.page for chunk in chunks} == {1, 2}


def test_compatibility_module_exports_core_function():
    from services.chunker import chunk_text as api_chunk_text

    assert api_chunk_text is chunk_text
