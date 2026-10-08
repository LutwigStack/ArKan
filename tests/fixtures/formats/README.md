These valid fixtures were captured from ArKan commit
271d09a7ea0eb446fa983bd380765e526350546c before its runtime layout changed.
The fixture model is constructed explicitly in `tests/format_compatibility.rs`;
its weights and normalization do not depend on random initialization.

`network-v1.bin` has the ARKAN/version-1 envelope. `network-legacy.bin` is its
raw bincode body. `baked-v2.bin` has the existing baked/version-2 envelope.
The JSON files preserve direct-serde field names, order, and nested records.
Do not regenerate these files to accommodate a runtime refactor.
