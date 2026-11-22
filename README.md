# NXN1X current source snapshot

Use the NXN1X artwork in `source/dist/nxn1x.ico`, `nxn1x.png`, and `nxn1x.svg` for all Windows application and installer icons.

Snapshot of the current Windows ARM64 NXN1X workspace, including the active
NVN host and the current local Turnip/Mesa source with the F2R overlay applied.

- `source/`: emulator, NCE, shader and existing LSFG integration sources.
- `host/`: active NVN host selected by `source/src/nvn_host/CMakeLists.txt`.
- `turnip/mesa/`: complete locally tracked Mesa tree plus local added source
  files, including Windows WDDM changes; F2R `tu_cs.h` and `tu_cs.cc` applied.
- `drivers/`: preserved F1R/F2R overlays and their original notes.
- `tools/`, `assets/`, `docs/`: current workspace support files.
- `SOURCE_SHA256.json`: hashes of copied files, including applied Turnip files.
- `SOURCE_SNAPSHOT.json`: provenance and explicit exclusions.

This is a source snapshot, not a new release or a newly verified build.
The BananaNative PR #138 port remains incomplete; see
`docs/nvn-memory/PR138_PORT_RECONCILIATION.txt`.
Existing NCE, shader, LSFG and memory protection code is preserved verbatim.

## Build context

The original build helpers are retained without changing their behavior.
`tools/build_nx2.py` uses CMake/Ninja and the local MSYS2 ARM64 toolchain;
Qt/CPM dependencies are not bundled in this source upload. The pinned shader
translator binaries are also excluded; consult the host shader-chain lock and
build documentation for those dependencies.

`tools/build_turnip_f2r.py` records the original incremental build procedure,
which depends on a sibling donor build and its existing object files. It is
not a portable clean-build command. `turnip/native.local.ini` and
`turnip/build-options.snapshot.json` preserve the local Mesa configuration;
toolchain paths must be adjusted for another machine before a clean build.

No game data, keys, saves, build outputs, runtime binaries, logs or backup
archives are included. Alternate inactive host and donor copies are excluded.
Original component licenses and copyright notices are retained in each tree.
