"""cleanup.py の純粋関数の単体テスト（torch 等の重い依存なしで実行できる）。"""

from cleanup import (
    collapse_dupes,
    collapse_repeats,
    heading_key,
    html_table_to_md,
    is_heading_line,
    text_to_paragraphs,
)


def test_collapse_repeats():
    assert collapse_repeats("あいうえおかあいうえおか") == "あいうえおか"
    assert collapse_repeats("ふつうの文。") == "ふつうの文。"


def test_collapse_dupes():
    assert collapse_dupes("これは文です。これは文です。次の文。") == "これは文です。次の文。"


def test_heading_key_strips_edge_numbers():
    assert heading_key("037 第一章") == "第一章"
    assert heading_key("第一章 12") == "第一章"
    assert heading_key("第一章") == "第一章"


def test_is_heading_line():
    assert is_heading_line("第一章")
    assert not is_heading_line("これは本文です。")  # 文末記号で終わる
    assert not is_heading_line("```")  # 記号だけ
    assert not is_heading_line("12")  # 数字だけ


def test_html_table_to_md():
    md = html_table_to_md(
        "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>"
    )
    assert "| a | b |" in md
    assert "| --- | --- |" in md
    assert "| 1 | 2 |" in md


def test_text_to_paragraphs_footnote_and_split():
    refs: list[str] = []
    paras = text_to_paragraphs("ある文である。[2]次の段落である。", refs)
    assert refs == ["2"]
    assert paras[0] == "ある文である。"
    assert paras[1].startswith("[^2]")


def test_text_to_paragraphs_inline_number_stripped():
    refs: list[str] = []
    (para,) = text_to_paragraphs("ページ末に123という数字。", refs)
    assert "123" not in para  # CJK に挟まれた行内ノンブルは除去される
