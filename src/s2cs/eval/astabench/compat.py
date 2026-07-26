"""Compatibility shims for running astabench under our pinned dependency set.

astabench pins ``datasets~=3.2.0``. The LitQA2 dataset card
(``futurehouse/lab-bench``) declares feature types with ``_type: "List"``, a
Feature class that only exists in datasets >= 3.3. Under 3.2.0 the loader's
``generate_from_dict`` falls back to ``globals().get("List")``, which resolves
to ``typing.List`` (not a dataclass) and raises

    TypeError: must be called with a dataclass type or instance

``List`` is the >=3.3 replacement for ``Sequence(feature=...)``, so aliasing it
to ``Sequence`` in the feature registry lets 3.2.0 parse the card. This affects
every litqa2-backed task: ``litqa2_*`` and ``paper_finder_litqa2_*``.
"""

import logging

log = logging.getLogger(__name__)


def patch_datasets_list_feature() -> None:
    """Register a ``List`` -> ``Sequence`` alias in datasets' feature registry.

    Idempotent and a no-op on datasets versions that already ship ``List``.
    """
    import datasets.features.features as feats

    if "List" in feats._FEATURE_TYPES:
        return
    feats._FEATURE_TYPES["List"] = feats.Sequence
    log.info("patched datasets feature registry: List -> Sequence (litqa2 compat)")
