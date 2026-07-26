from s2cs.eval import gaia


def test_question_scorer_numeric():
    assert gaia.question_scorer("34689", "34689")
    assert gaia.question_scorer("34,689", "34689")   # commas/units stripped
    assert gaia.question_scorer("$34689", "34689")
    assert not gaia.question_scorer("34690", "34689")


def test_question_scorer_string():
    assert gaia.question_scorer("Blue Whale", "blue whale")   # case/space/punct-insensitive
    assert not gaia.question_scorer("whale", "blue whale")


def test_question_scorer_nonstring_answer():
    # the GLM tool-arg parser JSON-coerces "34689" -> int 34689; scorer must coerce, not crash
    assert gaia.question_scorer(34689, "34689")
    assert gaia.question_scorer(None, "34689") is False


def test_question_scorer_list():
    assert gaia.question_scorer("a, b, c", "a, b, c")
    assert not gaia.question_scorer("a, b", "a, b, c")   # length mismatch


def test_canon_gold():
    assert gaia._canon_gold({"correct_answer": "['34689']"}) == "34689"
    assert gaia._canon_gold({"correct_answer": "['a', 'b']"}) == "a, b"
    assert gaia._canon_gold({"correct_answer": "plain string"}) == "plain string"


def test_score():
    assert gaia.score([True, False, True, True]) == {"accuracy": 0.75}
    assert gaia.score([]) == {}
