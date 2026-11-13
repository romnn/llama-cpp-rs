//! Cargo output-path resolution used by the build script and its regression tests.

use std::path::Path;

/// Finds the profile directory above Cargo's build-script outputs.
///
/// Cargo 1.100 adds a package directory before the fingerprint; the `build` ancestor is stable.
pub fn cargo_profile_dir(out_dir: &Path) -> Option<&Path> {
    out_dir
        .ancestors()
        .find(|path| path.file_name().is_some_and(|name| name == "build"))?
        .parent()
}

#[cfg(test)]
mod tests {
    use super::cargo_profile_dir;
    use std::path::Path;

    #[test]
    fn legacy_output_layout_resolves_the_profile_directory() {
        assert_eq!(
            cargo_profile_dir(Path::new("target/debug/build/llama-cpp-sys-2-hash/out")),
            Some(Path::new("target/debug"))
        );
    }

    #[test]
    fn package_scoped_output_layout_resolves_the_profile_directory() {
        assert_eq!(
            cargo_profile_dir(Path::new(
                "target/x86_64-unknown-linux-gnu/custom/build/llama-cpp-sys-2/hash/out"
            )),
            Some(Path::new("target/x86_64-unknown-linux-gnu/custom"))
        );
    }

    #[test]
    fn an_output_path_without_a_build_directory_is_rejected() {
        assert_eq!(cargo_profile_dir(Path::new("target/debug/out")), None);
    }
}
