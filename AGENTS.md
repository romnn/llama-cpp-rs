# Agent guidelines

## llama.cpp submodule

`llama-cpp-sys-2/llama.cpp` is an upstream Git submodule. Treat its checkout as immutable:

- Never edit files inside it, apply patches to it, commit inside it, or leave its worktree dirty.
- Keep upstream updates as ordinary submodule pointer bumps so they remain conflict-free.
- Put unavoidable downstream source fixes in `patches/*.patch`, with upstream provenance, rationale,
  and validation notes.
- Do not apply those patches automatically during normal builds. Validate them in a disposable
  checkout outside the submodule.
- Prefer fixes in the Rust wrappers or build integration when the problem can be solved without
  changing upstream source.
- Once upstream merges a fix, bump the submodule and remove the obsolete downstream patch.

Before finishing work, verify `git -C llama-cpp-sys-2/llama.cpp status --short` is empty.
