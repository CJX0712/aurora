from aurora.rag import KnowledgeBase, chunk_text


def test_chunk_text_splits_and_overlaps():
    text = "abcdefghij" * 20  # 200 chars
    chunks = chunk_text(text, size=50, overlap=10)
    assert len(chunks) > 1
    # overlaps preserved
    assert chunks[0][-10:] == chunks[1][:10]


def test_ingest_and_retrieve_relevant(cfg, embedder):
    kb = KnowledgeBase(cfg, embedder)
    kb.clear()
    kb.ingest_text("The quick brown fox jumps over the lazy dog near the river.", source="animals.txt")
    kb.ingest_text("Python is a programming language used for automation and AI.", source="tech.txt")

    hits = kb.retrieve("fox river animal", top_k=2)
    assert hits
    assert hits[0]["source"] == "animals.txt"
    assert 0.0 <= hits[0]["score"] <= 1.0


def test_retrieve_empty_when_no_data(cfg, embedder):
    kb = KnowledgeBase(cfg, embedder)
    kb.clear()
    assert kb.retrieve("anything") == []


def test_embedding_dim_consistency(cfg, embedder):
    kb = KnowledgeBase(cfg, embedder)
    kb.clear()
    kb.ingest_text("alpha beta gamma delta epsilon", source="x.txt")
    hits = kb.retrieve("alpha beta", top_k=1)
    # fake embedder uses 8-dim vectors; retrieval ran without dimension mismatch
    assert isinstance(hits[0]["score"], float)
