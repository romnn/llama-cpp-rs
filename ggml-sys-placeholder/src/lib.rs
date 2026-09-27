//! Empty stand-in for `ggml-sys`, so this workspace builds and tests on its own.
//!
//! The `system-ggml` feature links against a shared ggml that the consuming workspace provides
//! through `[patch.crates-io]`; crates.io has only an unrelated `ggml-sys`, whose missing backend
//! features would otherwise stop dependency resolution even with the feature off.
//! A consumer's own patch replaces this one, because Cargo applies only the root workspace's
//! patches.
//! Enabling `system-ggml` in a standalone build fails in `llama-cpp-sys-2`'s build script, which
//! finds none of the link metadata a real `ggml-sys` exports.
