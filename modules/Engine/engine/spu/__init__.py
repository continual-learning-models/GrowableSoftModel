"""SPU core — the generic Self-Processing machinery (objective /
loop / policy / report / attention-body binding). Model-specific
SPU drivers live with their model family."""
from .spu_policy import DEFAULT_SPU_POLICY, validate_spu_policy  # noqa: F401
from . import spu_policy                                          # noqa: F401
