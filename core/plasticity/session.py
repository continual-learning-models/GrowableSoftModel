"""PlasticSystem (Stage 2) — retained as a thin alias: the store
hooks and quarantine-first behavior were INTEGRATED into the System
facade itself in Stage 7 (owner: new capabilities must be usable, not
shelf parts). Import kept for backward compatibility."""
from core.facade import System


class PlasticSystem(System):
    pass
