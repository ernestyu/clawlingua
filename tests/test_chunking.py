from clawlingua.chunking.splitter import split_into_chunks


def test_chunking_stable_ids() -> None:
    text = (
        "Sentence one. Sentence two. Sentence three.\n\n"
        "Sentence four. Sentence five. Sentence six."
    )
    chunks_a = split_into_chunks(
        run_id="run1",
        text=text,
        max_chars=60,
        max_sentences=2,
        min_chars=10,
        overlap_sentences=1,
    )
    chunks_b = split_into_chunks(
        run_id="run1",
        text=text,
        max_chars=60,
        max_sentences=2,
        min_chars=10,
        overlap_sentences=1,
    )
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert all(c.source_text for c in chunks_a)

