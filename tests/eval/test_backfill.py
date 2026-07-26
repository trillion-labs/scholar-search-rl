from scripts.backfill_eval_manifest import _build_checkpoint_index, _discover_policies, infer_trial


def test_build_index_extracts_trial_from_real_layouts(tmp_path):
    (tmp_path / "checkpoints/org/trainer/value4k_band07/default/epoch3epochstep116globalstep467").mkdir(parents=True)
    (tmp_path / "checkpoints_verl/value4k_band17_lr_seal/eval_layout/epoch0epochstep60globalstep60").mkdir(parents=True)
    idx = _build_checkpoint_index(tmp_path)
    assert idx["467"] == ["value4k_band07"]
    assert idx["60"] == ["value4k_band17_lr_seal"]


def test_unique_trial_match():
    # globalstep -> trials that have a checkpoint at that step
    idx = {"19": ["value4k"], "39": ["value4k"], "249": ["value4k"]}
    r = infer_trial(["base", "step019", "step039", "step249"], idx)
    assert r["trial"] == "value4k" and r["confidence"] == "3/3" and r["ambiguous"] is False


def test_ambiguous_when_steps_in_multiple_trials():
    idx = {"49": ["A", "B"], "59": ["A", "B"]}
    r = infer_trial(["step049", "step059"], idx)
    assert r["ambiguous"] is True


def test_all_base_returns_unknown():
    r = infer_trial(["base"], {})
    assert r["trial"] is None
    assert r["confidence"] == "0/0"
    assert r["ambiguous"] is False


def test_partial_coverage_wins_by_count():
    # trial_a covers 2 steps, trial_b covers 1
    idx = {"10": ["trial_a"], "20": ["trial_a"], "30": ["trial_b"]}
    r = infer_trial(["step010", "step020", "step030"], idx)
    assert r["trial"] == "trial_a"
    assert r["confidence"] == "2/3"
    assert r["ambiguous"] is False


def test_unknown_step_not_in_index():
    idx = {"19": ["value4k"]}
    r = infer_trial(["step019", "step999"], idx)
    assert r["trial"] == "value4k"
    assert r["confidence"] == "1/2"
    assert r["ambiguous"] is False


def test_discover_policies_from_pf_jsons(tmp_path):
    run = tmp_path / "litqa2_verl_strict"
    (run / "litqa2_all").mkdir(parents=True)
    for lbl in ("base", "step020", "step040", "step060"):
        (run / "litqa2_all" / f"pf_{lbl}.json").write_text("{}")
    (run / "traj").mkdir()      # noise dir must be ignored
    (run / "scores").mkdir()
    assert _discover_policies(run) == ["base", "step020", "step040", "step060"]


def test_discover_policies_flat_layout(tmp_path):
    run = tmp_path / "pf_sweep"
    run.mkdir(parents=True)
    (run / "pf_base.json").write_text("{}")
    (run / "pf_step019.json").write_text("{}")
    assert _discover_policies(run) == ["base", "step019"]
