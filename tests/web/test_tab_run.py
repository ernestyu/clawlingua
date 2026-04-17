from __future__ import annotations

from clawlearn_web.ui.tab_run import learning_mode_to_domain, learning_mode_visibility_flags


def test_learning_mode_domain_defaults_to_lingua_for_unknown() -> None:
    assert learning_mode_to_domain("other") == "lingua"
    assert learning_mode_to_domain("") == "lingua"
    assert learning_mode_to_domain(None) == "lingua"


def test_learning_mode_domain_switches_to_textbook_prefix() -> None:
    assert learning_mode_to_domain("textbook_focus") == "textbook"
    assert learning_mode_to_domain(" TEXTBOOK_review ") == "textbook"


def test_learning_mode_visibility_flags_follow_domain() -> None:
    assert learning_mode_visibility_flags("lingua_expression") == (True, False)
    assert learning_mode_visibility_flags("textbook_focus") == (False, True)
