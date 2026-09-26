# Downstream llama.cpp patches

`build.rs` applies every `*.patch` here, in file-name order, to a copy of the `llama.cpp` submodule
in `OUT_DIR` and compiles only that copy with CMake. The submodule itself stays pristine. See the
repository's `AGENTS.md` for the rules a patch must follow.

This file keeps the directory present when no patch is needed: the build script watches it, and
Cargo reruns a build script on every build while a watched path is missing.
