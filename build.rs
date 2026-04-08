//! ScreenCaptureKit links `libswift_Concurrency.dylib` with `@rpath` while other Swift libs use
//! absolute `/usr/lib/swift/...`. Embed that path as the sole rpath so the binary runs from the
//! shell (e.g. `just install`) without `DYLD_LIBRARY_PATH` and without loading a second Swift runtime.

use std::path::Path;

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.contains("apple-darwin") {
        return;
    }

    let swift_lib = Path::new("/usr/lib/swift");
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        swift_lib.to_string_lossy()
    );
}
