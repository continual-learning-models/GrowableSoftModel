# trained_models/ — your trained models live here

- softmodel/  : models trained with the tool's primary softmodel
  method (override location: SOFTMODEL_MODELS_ROOT).
- standard/   : models trained with the optional industry-standard
  fixed-architecture mode (override: STANDARD_MODELS_ROOT).

Each model is a directory named after your model id; responses
from create/train/save always include the exact path.
