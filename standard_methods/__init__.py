"""standard_methods — the OPTIONAL industry-standard mode.

Fixed-architecture standard transformer/MLP trained from scratch
on the user's data (NOT pretrained-LLM fine-tuning). The tool's
primary method is softmodel (see core/); this family is the
secondary, user-selectable alternative. Separation is by
non-exposure: no evolution verbs exist here.
"""
from .facade import (create, evaluate, infer,      # noqa: F401
                     list_models, load, save, train)
