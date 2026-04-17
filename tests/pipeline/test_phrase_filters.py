from __future__ import annotations

from clawlearn.phrase_filters import filter_phrases


def test_filter_phrases_en_applies_hard_rules_and_advanced_threshold() -> None:
    kept, stats = filter_phrases(
        source_lang="en",
        phrases=[
            "one of the",
            "look at the",
            "to train on",
            "made solid progress",
        ],
        context="We made solid progress and discussed what data to train on.",
        difficulty="advanced",
    )

    assert kept == ["made solid progress"]
    assert stats["language"] == "en"
    assert stats["dropped_count"] == 3
    assert stats["dropped_by_rule"]["drop:blacklist_ngram"] == 2
    assert stats["dropped_by_rule"]["drop:stopword_ratio"] == 1


def test_filter_phrases_en_whitelist_skips_stopword_drop() -> None:
    kept, stats = filter_phrases(
        source_lang="en",
        phrases=["in terms of", "as a result", "to be fair", "on the one hand", "on the other hand"],
        context="In terms of strategy, this is useful.",
        difficulty="advanced",
    )
    assert kept == ["in terms of", "as a result", "to be fair", "on the one hand", "on the other hand"]
    assert stats["dropped_count"] == 0
    assert stats["dropped_by_rule"] == {}


def test_filter_phrases_non_english_routes_to_generic_only() -> None:
    kept, stats = filter_phrases(
        source_lang="fr",
        phrases=["one of the", "  one of the  ", "clean phrase"],
        context="ignored",
        difficulty="advanced",
    )
    assert kept == ["one of the", "clean phrase"]
    assert stats["language"] == "fr"
    assert stats["dropped_count"] == 1
    assert stats["dropped_by_rule"]["drop:duplicate_phrase"] == 1


def test_filter_phrases_examples_are_capped_per_rule() -> None:
    kept, stats = filter_phrases(
        source_lang="en",
        phrases=["look at the", "one of the", "in the same", "look at the"],
        context="ignored",
        difficulty="advanced",
    )
    assert kept == []
    examples = stats["examples_by_rule"]["drop:blacklist_ngram"]
    assert len(examples) == 3


def test_filter_phrases_en_advanced_min_score_drops_low_value_phrase() -> None:
    kept, stats = filter_phrases(
        source_lang="en",
        phrases=["you move", "solid move"],
        context="you move when you focus and make solid move changes over time.",
        difficulty="advanced",
    )
    assert kept == ["solid move"]
    assert stats["dropped_by_rule"]["drop:advanced_min_phrase_score"] == 1


def test_filter_phrases_en_drops_open_ending_incomplete_fragments() -> None:
    kept, stats = filter_phrases(
        source_lang="en",
        phrases=["We're moving from", "economic impact would", "diffused through the economy"],
        context="We're moving from one paradigm to another and economic impact would lag behind.",
        difficulty="advanced",
    )
    assert kept == ["diffused through the economy"]
    assert stats["dropped_by_rule"]["drop:ends_with_open_word"] == 2


def test_filter_phrases_en_advanced_drops_placeholder_tokens() -> None:
    kept, stats = filter_phrases(
        source_lang="en",
        phrases=["AI stuff", "Another thing", "go to some place", "make things happen"],
        context="AI stuff sounds vague and another thing often adds little value.",
        difficulty="advanced",
    )
    assert kept == ["make things happen"]
    assert stats["dropped_by_rule"]["drop:placeholder_token"] == 2
    assert stats["dropped_by_rule"]["drop:discourse_placeholder_blacklist"] == 1


def test_filter_phrases_en_non_advanced_does_not_hard_drop_placeholder_tokens() -> None:
    kept, stats = filter_phrases(
        source_lang="en",
        phrases=["AI stuff", "practical guidance"],
        context="AI stuff can be vague but here we only hard-drop in advanced mode.",
        difficulty="intermediate",
    )
    assert kept == ["AI stuff", "practical guidance"]
    assert "drop:placeholder_token" not in stats["dropped_by_rule"]


def test_filter_phrases_en_advanced_drops_discourse_placeholder_blacklist() -> None:
    kept, stats = filter_phrases(
        source_lang="en",
        phrases=["there is another explanation", "solid trade-off"],
        context="There is another explanation, but the solid trade-off matters.",
        difficulty="advanced",
    )
    assert kept == ["solid trade-off"]
    assert stats["dropped_by_rule"]["drop:discourse_placeholder_blacklist"] == 1
