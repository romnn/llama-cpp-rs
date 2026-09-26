# Agent guidelines

## llama.cpp submodule

`llama-cpp-sys-2/llama.cpp` is an upstream Git submodule. Treat its checkout as immutable:

- Never edit files inside it, apply patches to it, commit inside it, or leave its worktree dirty.
- Keep upstream updates as ordinary submodule pointer bumps so they remain conflict-free.
- Prefer fixes in the Rust wrappers or build integration when the problem can be solved without
  changing upstream source.
- Put unavoidable downstream llama.cpp source fixes in `llama-cpp-sys-2/patches/*.patch`, with
  upstream provenance, rationale, and validation notes in a header, following the existing files.
  Diff paths are rooted at the llama.cpp repository root.
- `build.rs` applies those patches, in file-name order, to a copy of the tree in `OUT_DIR` and
  compiles only that copy with CMake. The submodule itself is never touched.
- A patch may change sources but not headers: bindgen and the wrapper shims compile against the
  pristine headers, so a header change would reach only half the build. `build.rs` refuses one.
- Patches for ggml do not belong here. llama.cpp builds against the shared libggml of the
  consuming workspace's `ggml-sys` crate, which owns every ggml patch.
- Once upstream merges a fix, bump the submodule and remove the obsolete downstream patch. A patch
  that no longer applies fails the build with its name.

Before finishing work, verify `git -C llama-cpp-sys-2/llama.cpp status --short` is empty.
