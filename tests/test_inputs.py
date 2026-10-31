import json

import pytest

from cot_oversight_case_studies.io import load_rows


@pytest.mark.parametrize("row", [{}, [], {"id": []}, {"id": 1}, {"id": " "}])
def test_malformed_case_ids_fail_clearly(tmp_path, row):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1.*string case ID"):
        load_rows(path)


def test_jsonl_validation_and_blank_lines(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('\n{"id": "one"}\n\n{"id": "two"}\n', encoding="utf-8")
    assert [row["id"] for row in load_rows(path)] == ["one", "two"]
    path.write_text('{"id": "one"}\n{"id": "one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate.*line 2"):
        load_rows(path)
    path.write_text("not JSON\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON.*line 1"):
        load_rows(path)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        load_rows(path)
