# S0 baseline record (param-interface batch, internal design docs 21-23)
2026-07-19 · tags: pre-param-interface (Lib a7a5013+scanner, SMS 0cb6342)
- Lib full suite: 598 passed, 0 failed, 326.59s (venv repointed to
  -20260719 editables; five packages verified).
- SMS full suite: 208 passed, 1 failed = test_t30c version-pin
  tripwire (asserts Lib at tag attnet-v1.1; fires on the sanctioned
  S0 scanner commit; to be updated at S8 with param-interface-v1).
- Census scanner: 45 findings = 42 censused + 3 allowlisted; gate
  clean; manifest matches doc 22 exactly.
G0: PASSED.
