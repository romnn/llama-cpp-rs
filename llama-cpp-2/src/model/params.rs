//! A safe wrapper around `llama_model_params`.

use crate::context::params::LlamaContextParams;
use crate::model::params::kv_overrides::KvOverrides;
use crate::LlamaCppError;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::ptr::null;

pub mod kv_overrides;

/// Result of [`LlamaModelParams::fit_params`], containing the fitted context size.
#[cfg(feature = "common")]
#[derive(Debug, Clone)]
pub struct FitResult {
    /// The context size after fitting (may have been reduced from the requested value).
    pub n_ctx: u32,
}

/// Error returned by [`LlamaModelParams::fit_params`].
#[cfg(feature = "common")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FitError {
    /// Could not find allocations that are projected to fit available memory.
    #[error("could not find allocations that fit available memory")]
    Failure,
    /// A hard error occurred during fitting (e.g. model not found at the specified path).
    #[error("hard error during parameter fitting")]
    Error,
}

#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_possible_truncation)]
const LLAMA_SPLIT_MODE_NONE: i8 = llama_cpp_sys_2::LLAMA_SPLIT_MODE_NONE as i8;
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_possible_truncation)]
const LLAMA_SPLIT_MODE_LAYER: i8 = llama_cpp_sys_2::LLAMA_SPLIT_MODE_LAYER as i8;
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_possible_truncation)]
const LLAMA_SPLIT_MODE_ROW: i8 = llama_cpp_sys_2::LLAMA_SPLIT_MODE_ROW as i8;
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_possible_truncation)]
const LLAMA_SPLIT_MODE_TENSOR: i8 = llama_cpp_sys_2::LLAMA_SPLIT_MODE_TENSOR as i8;

/// A rusty wrapper around `llama_split_mode`.
#[repr(i8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaSplitMode {
    /// Single GPU
    None = LLAMA_SPLIT_MODE_NONE,
    /// Split layers and KV across GPUs
    Layer = LLAMA_SPLIT_MODE_LAYER,
    /// Split layers and KV across GPUs, use tensor parallelism if supported
    Row = LLAMA_SPLIT_MODE_ROW,
    /// Experimental tensor parallelism across GPUs
    Tensor = LLAMA_SPLIT_MODE_TENSOR,
}

/// An error that occurs when unknown split mode is encountered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaSplitModeParseError(pub i32);

/// Create a `LlamaSplitMode` from a `i32`.
///
/// # Errors
/// Returns `LlamaSplitModeParseError` if the value does not correspond to a valid `LlamaSplitMode`.
impl TryFrom<i32> for LlamaSplitMode {
    type Error = LlamaSplitModeParseError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let i8_value = value
            .try_into()
            .map_err(|_| LlamaSplitModeParseError(value))?;
        match i8_value {
            LLAMA_SPLIT_MODE_NONE => Ok(Self::None),
            LLAMA_SPLIT_MODE_LAYER => Ok(Self::Layer),
            LLAMA_SPLIT_MODE_ROW => Ok(Self::Row),
            LLAMA_SPLIT_MODE_TENSOR => Ok(Self::Tensor),
            _ => Err(LlamaSplitModeParseError(value)),
        }
    }
}

/// Create a `LlamaSplitMode` from a `u32`.
///
/// # Errors
/// Returns `LlamaSplitModeParseError` if the value does not correspond to a valid `LlamaSplitMode`.
impl TryFrom<u32> for LlamaSplitMode {
    type Error = LlamaSplitModeParseError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let i8_value = value
            .try_into()
            .map_err(|_| LlamaSplitModeParseError(value.try_into().unwrap_or(i32::MAX)))?;
        match i8_value {
            LLAMA_SPLIT_MODE_NONE => Ok(Self::None),
            LLAMA_SPLIT_MODE_LAYER => Ok(Self::Layer),
            LLAMA_SPLIT_MODE_ROW => Ok(Self::Row),
            LLAMA_SPLIT_MODE_TENSOR => Ok(Self::Tensor),
            _ => Err(LlamaSplitModeParseError(
                value.try_into().unwrap_or(i32::MAX),
            )),
        }
    }
}

/// Create a `i32` from a `LlamaSplitMode`.
impl From<LlamaSplitMode> for i32 {
    fn from(value: LlamaSplitMode) -> Self {
        match value {
            LlamaSplitMode::None => LLAMA_SPLIT_MODE_NONE.into(),
            LlamaSplitMode::Layer => LLAMA_SPLIT_MODE_LAYER.into(),
            LlamaSplitMode::Row => LLAMA_SPLIT_MODE_ROW.into(),
            LlamaSplitMode::Tensor => LLAMA_SPLIT_MODE_TENSOR.into(),
        }
    }
}

/// Create a `u32` from a `LlamaSplitMode`.
impl From<LlamaSplitMode> for u32 {
    fn from(value: LlamaSplitMode) -> Self {
        match value {
            LlamaSplitMode::None => LLAMA_SPLIT_MODE_NONE as u32,
            LlamaSplitMode::Layer => LLAMA_SPLIT_MODE_LAYER as u32,
            LlamaSplitMode::Row => LLAMA_SPLIT_MODE_ROW as u32,
            LlamaSplitMode::Tensor => LLAMA_SPLIT_MODE_TENSOR as u32,
        }
    }
}

/// The default split mode is `Layer` in llama.cpp.
impl Default for LlamaSplitMode {
    fn default() -> Self {
        LlamaSplitMode::Layer
    }
}

/// The maximum number of devices supported.
///
/// The real maximum number of devices is the lesser one of this value and the value returned by
/// `llama_cpp_2::max_devices()`.
pub const LLAMA_CPP_MAX_DEVICES: usize = 16;

/// Combines the two independent `use_mmap`/`use_mlock` flags this crate's public
/// API exposes into the single `load_mode` enum llama.cpp now stores them as.
fn load_mode_from_flags(use_mmap: bool, use_mlock: bool) -> llama_cpp_sys_2::llama_load_mode {
    match (use_mmap, use_mlock) {
        (false, false) => llama_cpp_sys_2::LLAMA_LOAD_MODE_NONE,
        (true, false) => llama_cpp_sys_2::LLAMA_LOAD_MODE_MMAP,
        (false, true) => llama_cpp_sys_2::LLAMA_LOAD_MODE_MLOCK,
        (true, true) => llama_cpp_sys_2::LLAMA_LOAD_MODE_MMAP_MLOCK,
    }
}

/// A safe wrapper around `llama_model_params`.
#[allow(clippy::module_name_repetitions)]
pub struct LlamaModelParams {
    pub(crate) params: llama_cpp_sys_2::llama_model_params,
    kv_overrides: Vec<llama_cpp_sys_2::llama_model_kv_override>,
    buft_overrides: Vec<llama_cpp_sys_2::llama_model_tensor_buft_override>,
    /// The patterns `buft_overrides` points into for the overrides added through these
    /// parameters, owned here so no caller's string has to outlive them.
    buft_patterns: Vec<CString>,
    devices: Pin<Box<[llama_cpp_sys_2::ggml_backend_dev_t; LLAMA_CPP_MAX_DEVICES]>>,
    tensor_split: Vec<f32>,
    progress_callback: Option<Box<dyn FnMut(f32) -> bool>>,
}

impl Debug for LlamaModelParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModelParams")
            .field("n_gpu_layers", &self.params.n_gpu_layers)
            .field("main_gpu", &self.params.main_gpu)
            .field("vocab_only", &self.params.vocab_only)
            .field("use_mmap", &self.use_mmap())
            .field("use_mlock", &self.use_mlock())
            .field("split_mode", &self.split_mode())
            .field("devices", &self.devices)
            .field("kv_overrides", &"vec of kv_overrides")
            .finish()
    }
}

impl LlamaModelParams {
    /// See [`KvOverrides`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use llama_cpp_2::model::params::LlamaModelParams;
    /// let params = Box::pin(LlamaModelParams::default());
    /// let kv_overrides = params.kv_overrides();
    /// let count = kv_overrides.into_iter().count();
    /// assert_eq!(count, 0);
    /// ```
    #[must_use]
    pub fn kv_overrides<'a>(&'a self) -> KvOverrides<'a> {
        KvOverrides::new(self)
    }

    /// Appends a key-value override to the model parameters. It must be pinned as this creates a self-referential struct.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::ffi::{CStr, CString};
    /// use std::pin::pin;
    /// # use llama_cpp_2::model::params::LlamaModelParams;
    /// # use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
    /// let mut params = pin!(LlamaModelParams::default());
    /// let key = CString::new("key").expect("CString::new failed");
    /// params.as_mut().append_kv_override(&key, ParamOverrideValue::Int(50));
    ///
    /// let kv_overrides = params.kv_overrides().into_iter().collect::<Vec<_>>();
    /// assert_eq!(kv_overrides.len(), 1);
    ///
    /// let (k, v) = &kv_overrides[0];
    /// assert_eq!(v, &ParamOverrideValue::Int(50));
    ///
    /// assert_eq!(k.to_bytes(), b"key", "expected key to be 'key', was {:?}", k);
    /// ```
    #[allow(clippy::missing_panics_doc)] // panics are just to enforce internal invariants, not user errors
    pub fn append_kv_override(
        mut self: Pin<&mut Self>,
        key: &CStr,
        value: kv_overrides::ParamOverrideValue,
    ) {
        let kv_override = self
            .kv_overrides
            .last_mut()
            .expect("kv_overrides did not have a next allocated");

        assert_eq!(kv_override.key[0], 0, "last kv_override was not empty");

        // There should be some way to do this without iterating over everything.
        for (i, &c) in key.to_bytes_with_nul().iter().enumerate() {
            kv_override.key[i] = c_char::try_from(c).expect("invalid character in key");
        }

        kv_override.tag = value.tag();
        kv_override.__bindgen_anon_1 = value.value();

        // set to null pointer for panic safety (as push may move the vector, invalidating the pointer)
        self.params.kv_overrides = null();

        // push the next one to ensure we maintain the iterator invariant of ending with a 0
        self.kv_overrides
            .push(llama_cpp_sys_2::llama_model_kv_override {
                key: [0; 128],
                tag: 0,
                __bindgen_anon_1: llama_cpp_sys_2::llama_model_kv_override__bindgen_ty_1 {
                    val_i64: 0,
                },
            });

        // set the pointer to the (potentially) new vector
        self.params.kv_overrides = self.kv_overrides.as_ptr();

        eprintln!("saved ptr: {:?}", self.params.kv_overrides);
    }
}

/// The tensor-name pattern llama.cpp matches a mixture-of-experts model's routed expert weights
/// with, its `LLM_FFN_EXPS_REGEX`.
///
/// It covers separate up, down and gate tensors and fused gate-up tensors, chunked or not, and
/// never a shared expert's (`_shexp`) tensors.
/// It is kept here rather than read from llama.cpp's `common` library, so builds without that
/// library can place experts too; a test of the `common` build checks the two agree.
pub const LLM_FFN_EXPS_REGEX: &str = r"\.ffn_(up|down|gate|gate_up)_(ch|)exps";

/// The pattern that matches the routed expert tensors of blocks `0..blocks`, as one alternation
/// so llama.cpp's loader, which compiles every override's pattern for every tensor, compiles one.
fn expert_prefix_pattern(blocks: u32) -> String {
    let indices = (0..blocks)
        .map(|block| block.to_string())
        .collect::<Vec<_>>()
        .join("|");
    format!(r"blk\.({indices}){LLM_FFN_EXPS_REGEX}")
}

/// The routed expert weights of a mixture-of-experts GGUF file, block by block.
#[cfg(feature = "common")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertTensorBytes {
    /// Bytes of each block's routed expert tensors, by block index, zero for a block without
    /// routed experts.
    pub block_bytes: Vec<u64>,
    /// How many of the last blocks are multi-token-prediction (`NextN`) blocks, which llama.cpp
    /// loads only for speculative decoding.
    pub nextn_blocks: u32,
}

/// Reads how many bytes the routed experts of each block of the GGUF file at `path` take, from
/// its tensor table, without loading any weights.
///
/// Expert tensors are the ones llama.cpp's loader matches with its own `LLM_FFN_EXPS_REGEX`, so a
/// dense model reports zero bytes for every block.
/// Returns `None` for a path or file that cannot be read.
#[cfg(feature = "common")]
#[must_use]
pub fn model_file_expert_bytes(path: &std::path::Path) -> Option<ExpertTensorBytes> {
    let path = CString::new(path.to_str()?).ok()?;
    // Enough for every model llama.cpp knows of; a file with more blocks is read a second time.
    let mut block_bytes = vec![0_u64; 256];
    loop {
        let mut block_count = 0_usize;
        let mut nextn_blocks = 0_u32;
        // SAFETY: `path` is NUL-terminated, `block_bytes` holds `block_bytes.len()` writable
        // entries, and both counts are valid for writes, all for the duration of the call.
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_model_file_expert_bytes(
                path.as_ptr(),
                block_bytes.as_mut_ptr(),
                block_bytes.len(),
                &raw mut block_count,
                &raw mut nextn_blocks,
            )
        };
        if !crate::status_is_ok(status) {
            return None;
        }
        if block_count <= block_bytes.len() {
            block_bytes.truncate(block_count);
            return Some(ExpertTensorBytes {
                block_bytes,
                nextn_blocks,
            });
        }
        block_bytes = vec![0; block_count];
    }
}

impl LlamaModelParams {
    /// Adds a buffer type override that keeps every routed expert of a mixture-of-experts model
    /// in system memory, as llama.cpp's `--cpu-moe` does.
    pub fn add_cpu_moe_override(&mut self) {
        self.add_cpu_buft_override_pattern(LLM_FFN_EXPS_REGEX);
    }

    /// Appends a buffer type override that keeps the tensors whose names match `pattern` in
    /// system memory, whatever layers are offloaded.
    ///
    /// The parameters keep their own copy of `pattern`, so the caller's string may be dropped
    /// right after the call.
    /// Overrides apply in the order they were added, the first match deciding a tensor's buffer.
    pub fn add_cpu_buft_override(&mut self, pattern: &CStr) {
        self.push_cpu_buft_override(pattern.to_owned());
    }

    /// Keeps the routed experts of the model's first `blocks` blocks in system memory, as
    /// llama.cpp's `--n-cpu-moe` does, with every other tensor placed as the layer split says.
    ///
    /// Zero blocks adds no override.
    #[must_use]
    pub fn with_cpu_expert_prefix(mut self, blocks: u32) -> Self {
        if blocks > 0 {
            self.add_cpu_buft_override_pattern(&expert_prefix_pattern(blocks));
        }
        self
    }

    /// Keeps every routed expert in system memory, the multi-token-prediction blocks' included,
    /// as llama.cpp's `--cpu-moe` does.
    #[must_use]
    pub fn with_cpu_all_experts(mut self) -> Self {
        self.add_cpu_moe_override();
        self
    }

    /// [`Self::add_cpu_buft_override`] for a pattern built in Rust.
    fn add_cpu_buft_override_pattern(&mut self, pattern: &str) {
        #[allow(clippy::expect_used)]
        let pattern = CString::new(pattern).expect("override patterns never contain a NUL byte");
        self.push_cpu_buft_override(pattern);
    }

    fn push_cpu_buft_override(&mut self, pattern: CString) {
        let entry = llama_cpp_sys_2::llama_model_tensor_buft_override {
            // A `CString` keeps its bytes on the heap, so the pointer stays valid as the vector of
            // owned patterns grows and as these parameters move.
            pattern: pattern.as_ptr(),
            // SAFETY: the CPU buffer type is a static of ggml's CPU backend, valid for the process.
            buft: unsafe { llama_cpp_sys_2::ggml_backend_cpu_buffer_type() },
        };
        self.buft_patterns.push(pattern);

        // Clear the pointer while the array changes, as resizing it may move it.
        self.params.tensor_buft_overrides = null();

        // The new override goes before the first null entry, which ends the list llama.cpp reads,
        // and a fresh terminator follows it.
        // A list `fit_params` filled holds null entries past its terminator, which are dropped.
        let terminator = self
            .buft_overrides
            .iter()
            .position(|buft_override| buft_override.pattern.is_null())
            .unwrap_or(self.buft_overrides.len());
        self.buft_overrides.truncate(terminator);
        self.buft_overrides.push(entry);
        self.buft_overrides
            .push(llama_cpp_sys_2::llama_model_tensor_buft_override {
                pattern: std::ptr::null(),
                buft: std::ptr::null_mut(),
            });

        self.params.tensor_buft_overrides = self.buft_overrides.as_ptr();
    }

    /// Returns the tensor-name patterns of the buffer-type overrides currently set on these
    /// parameters, in order.
    ///
    /// This is the read-only counterpart to [`add_cpu_buft_override`](Self::add_cpu_buft_override)
    /// and [`add_cpu_moe_override`](Self::add_cpu_moe_override). After
    /// [`fit_params`](Self::fit_params) it reflects the overrides the auto-fit chose — for example
    /// the routed-expert tensors (`blk.<N>.ffn_(up|down|gate_up|gate)_(ch|)exps`) a
    /// mixture-of-experts fit assigns to the CPU buffer type to make the model fit. Returns an empty
    /// vector when no overrides are set. The trailing null-terminator entry the override list
    /// carries is skipped; only entries before it are returned.
    #[must_use]
    pub fn tensor_buft_override_patterns(&self) -> Vec<String> {
        self.buft_overrides
            .iter()
            .take_while(|o| !o.pattern.is_null())
            .map(|o| {
                // SAFETY: a non-null `pattern` is a NUL-terminated C string.
                // For fit-produced overrides it points into process-lifetime function-local
                // `static` storage in llama.cpp's `common/fit.cpp`.
                // For overrides added through these parameters it points into a `CString` they
                // own and keep until they drop.
                // In both cases the string outlives this `&self` borrow.
                unsafe { CStr::from_ptr(o.pattern) }
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    }
}

#[cfg(feature = "common")]
impl LlamaModelParams {
    /// Automatically fit model parameters to available device memory.
    ///
    /// Wraps llama.cpp's `common_fit_params` (libcommon), which determines optimal `n_gpu_layers`,
    /// `tensor_split`, and `tensor_buft_overrides` based on available VRAM. On success
    /// the model and context params are updated in place.
    ///
    /// # Requirements
    ///
    /// Per the C API docstring, only parameters that still hold their default value
    /// are modified. In practice this means:
    /// - `n_gpu_layers` must be at its default (`-1`). Do not call
    ///   [`with_n_gpu_layers`](Self::with_n_gpu_layers) before this.
    /// - No `tensor_buft_overrides` may be set. Do not call
    ///   [`add_cpu_buft_override`](Self::add_cpu_buft_override) or
    ///   [`add_cpu_moe_override`](Self::add_cpu_moe_override) before this.
    /// - No tensor split may be set. A split from
    ///   [`with_tensor_split`](Self::with_tensor_split) is discarded and replaced by the fitted one.
    /// - `cparams.n_ctx` is only auto-selected if it is `0`; otherwise it is left alone.
    ///
    /// # Arguments
    ///
    /// - `model_path` — path to the GGUF model file.
    /// - `cparams` — context parameters; `n_ctx` may be modified (see above).
    /// - `margins` — memory margin per device in bytes. Must have at least
    ///   `llama_max_devices()` elements.
    /// - `n_ctx_min` — minimum context size to preserve when reducing memory usage.
    /// - `log_level` — minimum log level for fitting output; lower levels are routed
    ///   to the debug log.
    ///
    /// # Thread safety
    ///
    /// This function is **not** thread safe: the underlying C call mutates the global
    /// llama logger state.
    ///
    /// # Errors
    ///
    /// Returns [`FitError::Failure`] if no fitting allocation could be found, or
    /// [`FitError::Error`] on a hard error (e.g. the model file could not be read).
    pub fn fit_params(
        mut self: Pin<&mut Self>,
        model_path: &CStr,
        cparams: &mut LlamaContextParams,
        margins: &mut [usize],
        n_ctx_min: u32,
        log_level: llama_cpp_sys_2::ggml_log_level,
    ) -> Result<FitResult, FitError> {
        let max_devices = unsafe { llama_cpp_sys_2::llama_max_devices() };
        let max_buft = unsafe { llama_cpp_sys_2::llama_max_tensor_buft_overrides() };

        // Allocate tensor_split output buffer.
        self.tensor_split.clear();
        self.tensor_split.resize(max_devices, 0.0);

        // Reset and resize buft_overrides for fit output (null-terminated).
        self.buft_overrides.clear();
        self.buft_patterns.clear();
        self.buft_overrides.resize(
            max_buft + 1,
            llama_cpp_sys_2::llama_model_tensor_buft_override {
                pattern: std::ptr::null(),
                buft: std::ptr::null_mut(),
            },
        );

        // Clear pointers before the call — fit writes directly into the buffers above.
        self.params.tensor_split = null::<f32>();
        self.params.tensor_buft_overrides = null();

        let status = unsafe {
            llama_cpp_sys_2::llama_rs_fit_params(
                model_path.as_ptr(),
                &raw mut self.params,
                &raw mut cparams.context_params,
                self.tensor_split.as_mut_ptr(),
                self.buft_overrides.as_mut_ptr(),
                margins.as_mut_ptr(),
                n_ctx_min,
                std::ptr::null_mut(),
                log_level,
            )
        };

        // llama_rs_fit_params returns common_params_fit_status: 0 = success, 1 = failure, 2 = error.
        match status {
            0 => {}
            1 => return Err(FitError::Failure),
            _ => return Err(FitError::Error),
        }

        // Wire the owned buffers into the raw params.
        self.params.tensor_split = self.tensor_split.as_ptr();
        self.params.tensor_buft_overrides = self.buft_overrides.as_ptr();

        Ok(FitResult {
            n_ctx: cparams.context_params.n_ctx,
        })
    }
}

impl LlamaModelParams {
    /// Get the number of layers to offload to the GPU.
    #[must_use]
    pub fn n_gpu_layers(&self) -> i32 {
        self.params.n_gpu_layers
    }

    /// The GPU that is used for scratch and small tensors
    #[must_use]
    pub fn main_gpu(&self) -> i32 {
        self.params.main_gpu
    }

    /// only load the vocabulary, no weights
    #[must_use]
    pub fn vocab_only(&self) -> bool {
        self.params.vocab_only
    }

    /// use mmap if possible
    ///
    /// `use_mmap`/`use_mlock` were replaced by a single `load_mode` enum
    /// (`args: refactor mlock/mmap/directio into load-mode`, #20834); this
    /// getter decodes the combined mode back into the two independent flags
    /// this crate's public API still exposes.
    #[must_use]
    pub fn use_mmap(&self) -> bool {
        // `LLAMA_LOAD_MODE_AUTO` generally translates to `use_mmap`.
        // FIXME(madsmtm): Change this API to expose the load mode directly,
        // instead of making it two methods.
        matches!(
            self.params.load_mode,
            llama_cpp_sys_2::LLAMA_LOAD_MODE_AUTO
                | llama_cpp_sys_2::LLAMA_LOAD_MODE_MMAP
                | llama_cpp_sys_2::LLAMA_LOAD_MODE_MMAP_MLOCK
        )
    }

    /// force system to keep model in RAM
    #[must_use]
    pub fn use_mlock(&self) -> bool {
        matches!(
            self.params.load_mode,
            llama_cpp_sys_2::LLAMA_LOAD_MODE_MLOCK | llama_cpp_sys_2::LLAMA_LOAD_MODE_MMAP_MLOCK
        )
    }

    /// get the split mode
    ///
    /// # Errors
    /// Returns `LlamaSplitModeParseError` if the unknown split mode is encountered.
    pub fn split_mode(&self) -> Result<LlamaSplitMode, LlamaSplitModeParseError> {
        LlamaSplitMode::try_from(self.params.split_mode)
    }

    /// get the devices
    #[must_use]
    pub fn devices(&self) -> Vec<usize> {
        let mut backend_devices = Vec::new();
        for i in 0..unsafe { llama_cpp_sys_2::ggml_backend_dev_count() } {
            let dev = unsafe { llama_cpp_sys_2::ggml_backend_dev_get(i) };
            backend_devices.push(dev);
        }
        let mut devices = Vec::new();
        for &dev in self.devices.iter() {
            if dev.is_null() {
                break;
            }
            if let Some((index, _)) = backend_devices
                .iter()
                .enumerate()
                .find(|&(_i, &d)| d == dev)
            {
                devices.push(index);
            }
        }
        devices
    }

    /// sets the number of gpu layers to offload to the GPU.
    /// ```
    /// # use llama_cpp_2::model::params::LlamaModelParams;
    /// let params = LlamaModelParams::default();
    /// let params = params.with_n_gpu_layers(1);
    /// assert_eq!(params.n_gpu_layers(), 1);
    /// ```
    #[must_use]
    pub fn with_n_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        // The only way this conversion can fail is if u32 overflows the i32 - in which case we set
        // to MAX
        let n_gpu_layers = i32::try_from(n_gpu_layers).unwrap_or(i32::MAX);
        self.params.n_gpu_layers = n_gpu_layers;
        self
    }

    /// sets the main GPU
    ///
    /// To enable this option, you must set `split_mode` to `LlamaSplitMode::None` to enable single GPU mode.
    #[must_use]
    pub fn with_main_gpu(mut self, main_gpu: i32) -> Self {
        self.params.main_gpu = main_gpu;
        self
    }

    /// sets `vocab_only`
    #[must_use]
    pub fn with_vocab_only(mut self, vocab_only: bool) -> Self {
        self.params.vocab_only = vocab_only;
        self
    }

    /// sets `use_mmap`
    ///
    /// Clearing this drops to [`llama_cpp_sys_2::LLAMA_LOAD_MODE_NONE`], which also clears
    /// mlock, since mlock cannot hold without the mapping it is defined in terms of.
    #[must_use]
    pub fn with_use_mmap(mut self, use_mmap: bool) -> Self {
        self.params.load_mode = load_mode_from_flags(use_mmap, self.use_mlock());
        self
    }

    /// sets `use_mlock`
    #[must_use]
    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.params.load_mode = load_mode_from_flags(self.use_mmap(), use_mlock);
        self
    }

    /// sets `split_mode`
    #[must_use]
    pub fn with_split_mode(mut self, split_mode: LlamaSplitMode) -> Self {
        self.params.split_mode = split_mode.into();
        self
    }

    /// Sets each device's share of the model, parallel to
    /// [`with_devices`](Self::with_devices).
    ///
    /// Shares are relative, so they need not sum to one. An empty slice clears the split and
    /// restores llama.cpp's default, which distributes in proportion to free memory. The
    /// slice is padded with zeroes to `llama_max_devices()`, as the C API requires.
    ///
    /// The pointer handed to llama.cpp addresses the `Vec`'s heap buffer, not this struct, so
    /// it survives the moves that builder chaining performs.
    ///
    /// # Errors
    /// Returns `LlamaCppError::MaxDevicesExceeded` if `split` is longer than
    /// `llama_max_devices()`.
    pub fn with_tensor_split(mut self, split: &[f32]) -> Result<Self, LlamaCppError> {
        if split.is_empty() {
            self.tensor_split.clear();
            self.params.tensor_split = null::<f32>();
            return Ok(self);
        }
        let max_devices = crate::max_devices();
        if split.len() > max_devices {
            return Err(LlamaCppError::MaxDevicesExceeded(max_devices));
        }
        self.tensor_split.clear();
        self.tensor_split.extend_from_slice(split);
        self.tensor_split.resize(max_devices, 0.0);
        self.params.tensor_split = self.tensor_split.as_ptr();
        Ok(self)
    }

    /// Each device's share of the model, as last set by
    /// [`with_tensor_split`](Self::with_tensor_split). Empty when the split is llama.cpp's
    /// free-memory default.
    #[must_use]
    pub fn tensor_split(&self) -> &[f32] {
        &self.tensor_split
    }

    /// sets `devices`
    ///
    /// The devices are specified as indices that correspond to the ggml backend device indices.
    ///
    /// The maximum number of devices is 16.
    ///
    /// You don't need to specify CPU or ACCEL devices.
    ///
    /// # Errors
    /// Returns `LlamaCppError::BackendDeviceNotFound` if any device index is invalid.
    pub fn with_devices(mut self, devices: &[usize]) -> Result<Self, LlamaCppError> {
        for dev in self.devices.iter_mut() {
            *dev = std::ptr::null_mut();
        }
        // Check device count
        let max_devices = crate::max_devices().min(LLAMA_CPP_MAX_DEVICES);
        if devices.len() > max_devices {
            return Err(LlamaCppError::MaxDevicesExceeded(max_devices));
        }
        for (i, &dev) in devices.iter().enumerate() {
            if dev >= unsafe { llama_cpp_sys_2::ggml_backend_dev_count() } {
                return Err(LlamaCppError::BackendDeviceNotFound(dev));
            }
            let backend_dev = unsafe { llama_cpp_sys_2::ggml_backend_dev_get(dev) };
            self.devices[i] = backend_dev;
        }
        if self.devices.is_empty() {
            self.params.devices = std::ptr::null_mut();
        } else {
            self.params.devices = self.devices.as_mut_ptr();
        }
        Ok(self)
    }

    /// Set `no_alloc`
    ///
    /// If this parameter is true, don't allocate memory for the tensor data
    ///
    /// You can't use `no_alloc` with `use_mmap`, so this also sets `use_mmap` to false.
    #[must_use]
    pub fn with_no_alloc(mut self, no_alloc: bool) -> Self {
        self.params.no_alloc = no_alloc;
        if no_alloc {
            self = self.with_use_mmap(false);
        }
        self
    }

    /// Get `no_alloc`
    ///
    /// If this parameter is true, don't allocate memory for the tensor data
    #[must_use]
    pub fn no_alloc(&self) -> bool {
        self.params.no_alloc
    }

    /// Sets whether to load bundled multi-token prediction (MTP) tensors.
    #[must_use]
    pub fn with_load_mtp(mut self, load_mtp: bool) -> Self {
        self.params.load_mtp = load_mtp;
        self
    }

    /// Returns whether bundled multi-token prediction (MTP) tensors are loaded.
    #[must_use]
    pub fn load_mtp(&self) -> bool {
        self.params.load_mtp
    }

    /// Sets a callback invoked during loading with progress in `0.0..=1.0`.
    /// Returning `false` aborts the load (it then fails with `NullResult`).
    #[must_use]
    pub fn with_progress_callback<F: FnMut(f32) -> bool + 'static>(mut self, callback: F) -> Self {
        unsafe extern "C" fn trampoline<F: FnMut(f32) -> bool>(
            progress: f32,
            user_data: *mut c_void,
        ) -> bool {
            let callback = unsafe { &mut *user_data.cast::<F>() };
            callback(progress)
        }

        let mut callback = Box::new(callback);
        self.params.progress_callback_user_data =
            std::ptr::from_mut(&mut *callback).cast::<c_void>();
        self.params.progress_callback = Some(trampoline::<F>);
        self.progress_callback = Some(callback);
        self
    }
}

/// Default parameters for `LlamaModel`. (as defined in llama.cpp by `llama_model_default_params`)
/// ```
/// # use llama_cpp_2::model::params::LlamaModelParams;
/// use llama_cpp_2::model::params::LlamaSplitMode;
/// let params = LlamaModelParams::default();
/// assert_eq!(params.n_gpu_layers(), -1, "n_gpu_layers should be -1");
/// assert_eq!(params.main_gpu(), 0, "main_gpu should be 0");
/// assert_eq!(params.vocab_only(), false, "vocab_only should be false");
/// assert_eq!(params.use_mmap(), true, "use_mmap should be true");
/// assert_eq!(params.use_mlock(), false, "use_mlock should be false");
/// assert_eq!(params.split_mode(), Ok(LlamaSplitMode::Layer), "split_mode should be LAYER");
/// assert_eq!(params.devices().len(), 0, "devices should be empty");
/// assert_eq!(params.no_alloc(), false, "no_alloc should be false");
/// assert_eq!(params.load_mtp(), false, "load_mtp should be false");
/// ```
impl Default for LlamaModelParams {
    fn default() -> Self {
        let default_params = unsafe { llama_cpp_sys_2::llama_model_default_params() };
        LlamaModelParams {
            params: default_params,
            // push the next one to ensure we maintain the iterator invariant of ending with a 0
            kv_overrides: vec![llama_cpp_sys_2::llama_model_kv_override {
                key: [0; 128],
                tag: 0,
                __bindgen_anon_1: llama_cpp_sys_2::llama_model_kv_override__bindgen_ty_1 {
                    val_i64: 0,
                },
            }],
            buft_overrides: vec![llama_cpp_sys_2::llama_model_tensor_buft_override {
                pattern: std::ptr::null(),
                buft: std::ptr::null_mut(),
            }],
            buft_patterns: Vec::new(),
            devices: Box::pin([std::ptr::null_mut(); 16]),
            tensor_split: Vec::new(),
            progress_callback: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::kv_overrides::ParamOverrideValue;
    use super::{LlamaModelParams, LlamaSplitMode};
    use std::pin::pin;

    #[test]
    fn tensor_buft_override_patterns_empty_by_default() {
        // Fresh params carry only the null-terminator entry, so no patterns are reported.
        assert!(LlamaModelParams::default()
            .tensor_buft_override_patterns()
            .is_empty());
    }

    #[test]
    fn tensor_buft_override_patterns_reads_back_added_override() {
        // The getter is the read-only counterpart to the setter: the override added is reported and
        // the trailing null terminator is skipped. This mirrors how `fit_params` populates the same
        // buffer for the auto-fit's MoE expert offload (`add_cpu_moe_override` uses the same expert
        // tensor pattern shape the fit emits).
        let mut params = pin!(LlamaModelParams::default());
        params.as_mut().add_cpu_moe_override();
        assert_eq!(
            params.tensor_buft_override_patterns(),
            vec![super::LLM_FFN_EXPS_REGEX.to_owned()],
        );
    }

    #[test]
    fn tensor_buft_override_patterns_outlive_the_callers_string_and_moves() {
        // The parameters own their copy of each pattern, so the caller's string may go before
        // the parameters are read or moved.
        let mut params = LlamaModelParams::default();
        {
            let pattern = std::ffi::CString::new("blk\\.7\\.ffn_up_exps").unwrap();
            params.add_cpu_buft_override(&pattern);
        }
        let moved = Box::new(params);
        let moved = *moved;
        assert_eq!(
            moved.tensor_buft_override_patterns(),
            vec!["blk\\.7\\.ffn_up_exps".to_owned()],
        );
        // The array llama.cpp reads starts at the owned entries and ends with a null terminator.
        assert_eq!(moved.params.tensor_buft_overrides, moved.buft_overrides.as_ptr());
        assert!(moved
            .buft_overrides
            .last()
            .is_some_and(|terminator| terminator.pattern.is_null()));
    }

    #[test]
    fn tensor_buft_overrides_keep_the_order_they_were_added_in() {
        let mut params = LlamaModelParams::default()
            .with_cpu_expert_prefix(2)
            .with_cpu_all_experts();
        params.add_cpu_buft_override(c"output\\.weight");

        assert_eq!(
            params.tensor_buft_override_patterns(),
            vec![
                format!("blk\\.(0|1){}", super::LLM_FFN_EXPS_REGEX),
                super::LLM_FFN_EXPS_REGEX.to_owned(),
                "output\\.weight".to_owned(),
            ],
        );
        assert_eq!(params.buft_overrides.len(), 4);
    }

    #[test]
    fn an_expert_prefix_of_no_blocks_adds_no_override() {
        assert!(LlamaModelParams::default()
            .with_cpu_expert_prefix(0)
            .tensor_buft_override_patterns()
            .is_empty());
    }

    #[test]
    fn expert_patterns_match_routed_experts_and_never_shared_ones() {
        // llama.cpp matches these with `std::regex_search`; the patterns use only syntax the
        // `regex` crate reads the same way.
        let prefix = regex::Regex::new(&super::expert_prefix_pattern(3)).unwrap();
        let all = regex::Regex::new(super::LLM_FFN_EXPS_REGEX).unwrap();
        for routed in [
            "blk.0.ffn_up_exps.weight",
            "blk.1.ffn_down_exps.weight",
            "blk.2.ffn_gate_exps.weight",
            "blk.2.ffn_gate_up_exps.weight",
            "blk.0.ffn_down_chexps.weight",
        ] {
            assert!(prefix.is_match(routed), "{routed}");
            assert!(all.is_match(routed), "{routed}");
        }
        // Blocks past the prefix, including ones whose index starts with a prefix block's.
        for later in ["blk.3.ffn_up_exps.weight", "blk.12.ffn_gate_up_exps.weight"] {
            assert!(!prefix.is_match(later), "{later}");
            assert!(all.is_match(later), "{later}");
        }
        for other in [
            "blk.0.ffn_up_shexp.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_up.weight",
            "blk.0.attn_q.weight",
        ] {
            assert!(!prefix.is_match(other), "{other}");
            assert!(!all.is_match(other), "{other}");
        }
    }

    #[cfg(feature = "common")]
    #[test]
    fn the_expert_pattern_is_llama_cpps() {
        // SAFETY: the shim returns a pointer to llama.cpp's static NUL-terminated pattern.
        let upstream = unsafe { std::ffi::CStr::from_ptr(llama_cpp_sys_2::llama_rs_ffn_exps_regex()) };
        assert_eq!(upstream.to_str(), Ok(super::LLM_FFN_EXPS_REGEX));
    }

    #[cfg(feature = "common")]
    #[test]
    fn a_model_without_routed_experts_reports_none() {
        // A vocabulary-only BERT file: a readable GGUF with no expert tensors in any block.
        let bytes = super::model_file_expert_bytes(std::path::Path::new(
            "src/gguf/ggml-vocab-bert-bge.gguf",
        ))
        .unwrap();
        assert!(bytes.block_bytes.iter().all(|&block| block == 0));
        assert_eq!(bytes.nextn_blocks, 0);
    }

    #[cfg(feature = "common")]
    #[test]
    fn a_file_that_cannot_be_read_has_no_expert_bytes() {
        assert_eq!(
            super::model_file_expert_bytes(std::path::Path::new("/nonexistent/model.gguf")),
            None
        );
    }

    #[test]
    fn tensor_buft_override_appends_on_second_call() {
        // add_cpu_buft_override is documented as appending. After the first call the
        // vector is [filled, null-terminator]; writing the second override into the
        // terminator (then pushing a new one) is what keeps the list null-terminated.
        let mut params = pin!(LlamaModelParams::default());
        params
            .as_mut()
            .add_cpu_buft_override(c"blk\\.0\\.ffn_(up|down|gate)_(ch|)exps");
        params
            .as_mut()
            .add_cpu_buft_override(c"blk\\.1\\.ffn_(up|down|gate)_(ch|)exps");

        assert_eq!(
            params.tensor_buft_override_patterns(),
            vec![
                "blk\\.0\\.ffn_(up|down|gate)_(ch|)exps".to_owned(),
                "blk\\.1\\.ffn_(up|down|gate)_(ch|)exps".to_owned(),
            ],
        );
    }

    #[test]
    fn kv_override_appends_on_second_call() {
        // Same [filled, null-terminator] layout as the buffer overrides: the second call has to
        // write into the terminator, not back into slot 0.
        let mut params = pin!(LlamaModelParams::default());
        params
            .as_mut()
            .append_kv_override(c"first", ParamOverrideValue::Int(1));
        params
            .as_mut()
            .append_kv_override(c"second", ParamOverrideValue::Int(2));

        let kv_overrides = params.kv_overrides().into_iter().collect::<Vec<_>>();
        assert_eq!(kv_overrides.len(), 2);
        assert_eq!(kv_overrides[0].0.to_bytes(), b"first");
        assert_eq!(kv_overrides[1].0.to_bytes(), b"second");
        assert_eq!(kv_overrides[1].1, ParamOverrideValue::Int(2));
    }

    #[test]
    fn tensor_split_round_trips_and_pads() {
        let max = crate::max_devices();
        let params = LlamaModelParams::default()
            .with_tensor_split(&[0.25, 0.75])
            .expect("two devices is under any max");
        assert_eq!(params.tensor_split().len(), max);
        assert_eq!(&params.tensor_split()[..2], &[0.25, 0.75]);
        assert!(params.tensor_split()[2..].iter().all(|share| *share == 0.0));

        // The pointer must survive the moves builder chaining performs.
        let params = params.with_n_gpu_layers(99);
        assert_eq!(&params.tensor_split()[..2], &[0.25, 0.75]);

        let cleared = params.with_tensor_split(&[]).expect("empty clears");
        assert!(cleared.tensor_split().is_empty());
    }

    #[test]
    fn tensor_split_rejects_more_devices_than_llama_supports() {
        let too_many = vec![1.0_f32; crate::max_devices() + 1];
        assert!(LlamaModelParams::default()
            .with_tensor_split(&too_many)
            .is_err());
    }

    #[test]
    fn tensor_split_mode_round_trips() {
        assert_eq!(
            LlamaSplitMode::try_from(llama_cpp_sys_2::LLAMA_SPLIT_MODE_TENSOR),
            Ok(LlamaSplitMode::Tensor)
        );
        assert_eq!(
            u32::from(LlamaSplitMode::Tensor),
            llama_cpp_sys_2::LLAMA_SPLIT_MODE_TENSOR as u32
        );
        assert_eq!(
            i32::from(LlamaSplitMode::Tensor),
            llama_cpp_sys_2::LLAMA_SPLIT_MODE_TENSOR as i32
        );
    }

    #[test]
    fn load_mtp_round_trips() {
        let params = LlamaModelParams::default();
        assert!(!params.load_mtp());

        let params = params.with_load_mtp(true);
        assert!(params.load_mtp());

        let params = params.with_load_mtp(false);
        assert!(!params.load_mtp());
    }

    #[test]
    fn progress_callback_round_trips_and_can_abort() {
        use super::LlamaModelParams;
        use std::cell::Cell;
        use std::rc::Rc;

        let calls = Rc::new(Cell::new(0_u32));
        let counter = Rc::clone(&calls);
        let params = LlamaModelParams::default().with_progress_callback(move |_progress| {
            counter.set(counter.get() + 1);
            false
        });

        assert!(params.params.progress_callback.is_some());
        assert!(!params.params.progress_callback_user_data.is_null());

        let trampoline = params.params.progress_callback.unwrap();
        let user_data = params.params.progress_callback_user_data;
        let first = unsafe { trampoline(0.5, user_data) };
        let second = unsafe { trampoline(1.0, user_data) };

        assert!(!first && !second, "returning false signals an abort");
        assert_eq!(calls.get(), 2);
    }
}
