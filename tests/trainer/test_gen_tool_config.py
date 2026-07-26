from s2cs.trainer.tools.gen_tool_config import build_tool_config


def test_each_tool_becomes_a_verl_entry():
    names = ["search_papers", "submit_answer"]
    cfg = build_tool_config(names)
    entries = cfg["tools"]
    assert [e["config"]["tool_name"] for e in entries] == names
    for e in entries:
        assert e["class_name"].endswith("S2CSTool")
        fn = e["tool_schema"]["function"]
        assert fn["name"] in names
        assert fn["parameters"]["type"] == "object"


def test_mask_paper_ids_stripped_from_schema():
    cfg = build_tool_config(["search_papers"])
    props = cfg["tools"][0]["tool_schema"]["function"]["parameters"]["properties"]
    assert "mask_paper_ids" not in props


def test_unknown_tool_raises():
    import pytest

    with pytest.raises(KeyError):
        build_tool_config(["not_a_tool"])
