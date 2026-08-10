"""IWP0 acceptance: UT0.1 shim exposes both phase packages.
(The former UT0.2 module-integrity guard self-test was retired when
the modules were unfrozen for release.)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def test_ut0_1_shim():
    from core._modules import generator, reference_net
    assert hasattr(generator, "__version__")
    from generator.factory import SoftModelFactory        # noqa: F401
    from reference_net.net import Network                    # noqa: F401


if __name__ == "__main__":
    test_ut0_1_shim()
    print("iwp0 tests passed")
