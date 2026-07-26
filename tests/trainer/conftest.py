"""Make the vendored verl fork importable as `verl` for trainer tests.

The fork is vendored (gitignored) by `vendor_setup.sh`, not pip-installed into
the dev venv. For unit tests we only need `verl.tools.*`, which imports cleanly
with `tensordict` + `omegaconf` (in the `trainer` group) against the venv
torch. So we add the vendored path here.
"""

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[2] / "src" / "s2cs" / "trainer" / "vendor" / "verl_patched"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
