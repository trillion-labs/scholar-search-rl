import math

from s2cs.agent.tools import specs
from s2cs.eval import simpleqa


def test_parse_grade_maps_letters_and_defaults_to_not_attempted():
    assert simpleqa._parse_grade("A") == "correct"
    assert simpleqa._parse_grade("B") == "incorrect"
    assert simpleqa._parse_grade("C") == "not_attempted"
    assert simpleqa._parse_grade("The grade is A.") == "correct"
    assert simpleqa._parse_grade("nonsense") == "not_attempted"
    assert simpleqa._parse_grade("") == "not_attempted"
    assert simpleqa._parse_grade(None) == "not_attempted"


def test_score_matches_simpleqa_definitions():
    labels = ["correct", "correct", "incorrect", "not_attempted"]
    m = simpleqa.score(labels)
    assert m["correct"] == 0.5
    assert m["incorrect"] == 0.25
    assert m["not_attempted"] == 0.25
    assert m["attempted"] == 0.75
    assert math.isclose(m["correct_given_attempted"], 0.5 / 0.75)
    cga = 0.5 / 0.75
    assert math.isclose(m["f_score"], 2 * 0.5 * cga / (0.5 + cga))


def test_score_empty_is_empty():
    assert simpleqa.score([]) == {}


def test_score_no_attempts_avoids_div_by_zero():
    m = simpleqa.score(["not_attempted", "not_attempted"])
    assert m["correct"] == 0.0
    assert m["attempted"] == 0.0
    assert m["correct_given_attempted"] == 0.0
    assert m["f_score"] == 0.0


def _write_csv(path, n):
    lines = ["metadata,problem,answer"]
    for i in range(n):
        lines.append(f"m{i},question {i}?,answer {i}")
    lines.append("mbad,,missing answer row")  # filtered out (no answer)
    path.write_text("\n".join(lines), encoding="utf-8")


def test_load_questions_filters_sample_and_caps(tmp_path):
    csv_path = tmp_path / "sqa.csv"
    _write_csv(csv_path, 10)

    full = simpleqa.load_questions(str(csv_path))
    assert len(full) == 10  # blank-answer row dropped
    assert full[0][0] == "simpleqa_0"

    s1 = simpleqa.load_questions(str(csv_path), sample=3, seed=0)
    s2 = simpleqa.load_questions(str(csv_path), sample=3, seed=0)
    assert len(s1) == 3 and s1 == s2  # seeded -> deterministic

    capped = simpleqa.load_questions(str(csv_path), limit=2)
    assert len(capped) == 2


def test_build_tools_emit_native_web_specs():
    tools = simpleqa.build_tools(serper_key="x")
    assert set(tools) == {"web_search", "fetch_url", "submit_answer"}
    by_name = {s["function"]["name"]: s["function"] for s in specs(tools)}

    ws = by_name["web_search"]["parameters"]
    assert "query" in ws["properties"] and ws["properties"]["query"]["type"] == "string"
    assert "query" in ws["required"]
    assert ws["properties"]["num_results"]["type"] == "integer"
    assert "num_results" not in ws["required"]  # has a default

    fu = by_name["fetch_url"]["parameters"]
    assert fu["properties"]["url"]["type"] == "string" and "url" in fu["required"]


def test_web_search_parses_answer_box_and_organic(monkeypatch):
    import requests

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "answerBox": {"answer": "42", "link": "http://a"},
                "organic": [
                    {"position": 1, "title": "T1", "link": "http://1", "snippet": "s1"},
                    {"position": 2, "title": "T2", "link": "http://2", "snippet": "s2"},
                ],
            }

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(requests, "post", fake_post)

    web_search = simpleqa.make_web_search("KEY", default_num=10)
    out = web_search("life, the universe and everything", num_results=5)

    assert captured["url"] == "https://google.serper.dev/search"
    assert captured["headers"]["X-API-KEY"] == "KEY"
    assert captured["json"]["q"] == "life, the universe and everything"
    assert out[0]["rank"] == 0 and out[0]["snippet"] == "42"  # answer box first
    assert [r["title"] for r in out[1:]] == ["T1", "T2"]


def test_fetch_url_strips_html(monkeypatch):
    import requests

    class _Resp:
        text = "<html><head><style>x{}</style></head><body><p>Hello&amp; world</p><script>bad()</script></body></html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())

    fetch_url = simpleqa.make_fetch_url(max_chars=1000)
    out = fetch_url("http://x")
    assert "Hello& world" in out["text"]
    assert "bad()" not in out["text"] and "<" not in out["text"]
