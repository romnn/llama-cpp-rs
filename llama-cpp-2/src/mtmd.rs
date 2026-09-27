//! Safe wrapper around multimodal (MTMD) functionality in llama.cpp.
//!
//! This module provides Rust bindings for llama.cpp's multimodal support,
//! allowing processing of text, image, and audio inputs through a unified interface.
//!
//! # Warning
//! This API is experimental and subject to breaking changes.
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::slice;
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::context::LlamaContext;
use crate::model::LlamaModel;
use crate::token::LlamaToken;

/// Input chunk types for multimodal data
///
/// # Examples
///
/// ```
/// use llama_cpp_2::mtmd::MtmdInputChunkType;
///
/// let text_chunk = MtmdInputChunkType::Text;
/// let image_chunk = MtmdInputChunkType::Image;
/// let audio_chunk = MtmdInputChunkType::Audio;
///
/// assert_eq!(text_chunk, MtmdInputChunkType::Text);
/// assert_eq!(text_chunk, llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_TEXT.into());
/// assert_ne!(text_chunk, image_chunk);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MtmdInputChunkType {
    /// Text input chunk
    Text = llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_TEXT as _,
    /// Image input chunk
    Image = llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_IMAGE as _,
    /// Audio input chunk
    Audio = llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_AUDIO as _,
}

impl From<llama_cpp_sys_2::mtmd_input_chunk_type> for MtmdInputChunkType {
    fn from(chunk_type: llama_cpp_sys_2::mtmd_input_chunk_type) -> Self {
        match chunk_type {
            llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_TEXT => MtmdInputChunkType::Text,
            llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_IMAGE => MtmdInputChunkType::Image,
            llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_AUDIO => MtmdInputChunkType::Audio,
            _ => panic!("Unknown MTMD input chunk type: {chunk_type}"),
        }
    }
}

/// Configuration parameters for MTMD context
///
/// # Examples
///
/// ```
/// use llama_cpp_2::mtmd::{MtmdContextParams, mtmd_default_marker};
/// use std::ffi::CString;
///
/// let params = MtmdContextParams {
///     use_gpu: false,
///     device: None,
///     print_timings: true,
///     n_threads: 4,
///     media_marker: CString::new(mtmd_default_marker()).unwrap(),
///     image_min_tokens: -1,
///     image_max_tokens: -1,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MtmdContextParams {
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// Index of the ggml backend device the projector runs on when `use_gpu` is set, as
    /// [`crate::list_llama_ggml_backend_devices`] lists it.
    ///
    /// `None` lets llama.cpp pick the first GPU, or the first integrated GPU when there is none.
    pub device: Option<usize>,
    /// Whether to print timing information
    pub print_timings: bool,
    /// Number of threads to use for processing
    pub n_threads: i32,
    /// Media marker string used to identify media positions in text
    pub media_marker: CString,
    /// Minimum number of tokens used to represent an image.
    /// Controls the visual token budget lower bound. Use -1 for the model default.
    /// Gemma 4 supported budgets: 70, 140, 280, 560, 1120.
    pub image_min_tokens: i32,
    /// Maximum number of tokens used to represent an image.
    /// Controls the visual token budget upper bound. Use -1 for the model default.
    /// Lower values reduce memory and compute at the cost of visual detail.
    /// Gemma 4 supported budgets: 70, 140, 280, 560, 1120.
    pub image_max_tokens: i32,
}

impl Default for MtmdContextParams {
    fn default() -> Self {
        unsafe { llama_cpp_sys_2::mtmd_context_params_default() }.into()
    }
}

impl From<&MtmdContextParams> for llama_cpp_sys_2::mtmd_context_params {
    fn from(params: &MtmdContextParams) -> Self {
        let mut context = unsafe { llama_cpp_sys_2::mtmd_context_params_default() };
        let MtmdContextParams {
            use_gpu,
            device,
            print_timings,
            n_threads,
            media_marker,
            image_min_tokens,
            image_max_tokens,
        } = params;

        context.use_gpu = *use_gpu;
        // An index outside the registry leaves the choice to llama.cpp rather than tripping
        // `ggml_backend_dev_get`'s assertion; `MtmdContext::init_from_file` rejects it first.
        // SAFETY: both registry calls take no pointers, and the filter keeps the index in range.
        context.device = device
            .filter(|index| *index < unsafe { llama_cpp_sys_2::ggml_backend_dev_count() })
            .map_or(std::ptr::null_mut(), |index| unsafe {
                llama_cpp_sys_2::ggml_backend_dev_get(index)
            });
        context.print_timings = *print_timings;
        context.n_threads = *n_threads;
        context.media_marker = media_marker.as_ptr();
        context.image_min_tokens = *image_min_tokens;
        context.image_max_tokens = *image_max_tokens;

        context
    }
}

impl From<llama_cpp_sys_2::mtmd_context_params> for MtmdContextParams {
    fn from(params: llama_cpp_sys_2::mtmd_context_params) -> Self {
        Self {
            use_gpu: params.use_gpu,
            device: backend_device_index(params.device),
            print_timings: params.print_timings,
            n_threads: params.n_threads,
            media_marker: unsafe { CStr::from_ptr(params.media_marker) }.to_owned(),
            image_min_tokens: params.image_min_tokens,
            image_max_tokens: params.image_max_tokens,
        }
    }
}

/// Returns the registry index of a ggml backend device, or `None` for a null or unregistered one.
fn backend_device_index(device: llama_cpp_sys_2::ggml_backend_dev_t) -> Option<usize> {
    if device.is_null() {
        return None;
    }
    // SAFETY: the registry calls take no pointers, and every index is below the count.
    let count = unsafe { llama_cpp_sys_2::ggml_backend_dev_count() };
    (0..count).find(|index| unsafe { llama_cpp_sys_2::ggml_backend_dev_get(*index) } == device)
}

/// Memory a projector would hold on one ggml backend device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtmdDeviceMemory {
    /// Index of the device, as [`crate::list_llama_ggml_backend_devices`] lists it.
    pub device_index: usize,
    /// Bytes on this device: the weights plus the warmup compute buffers from
    /// [`MtmdContext::estimate_memory_usage`], or one input's compute buffers from
    /// [`MtmdContext::estimate_chunk_compute`].
    pub bytes: usize,
}

/// Text input configuration
///
/// # Examples
///
/// ```
/// use llama_cpp_2::mtmd::MtmdInputText;
///
/// let input = MtmdInputText {
///     text: "Describe this image.".to_string(),
///     add_special: true,
///     parse_special: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MtmdInputText {
    /// The input text string
    pub text: String,
    /// Whether to add special tokens
    pub add_special: bool,
    /// Whether to parse special tokens
    pub parse_special: bool,
}

/// Safe wrapper around `mtmd_context`.
///
/// This represents an initialized multimodal context that can process
/// text, images, and audio through llama.cpp's multimodal interface.
///
/// # Thread safety
///
/// A context may be shared across threads.
/// Upstream allows tokenizing and creating bitmaps on a shared context, and the decode queries
/// ([`Self::decode_use_mrope`], [`Self::decode_use_non_causal`], and the ones
/// [`MtmdInputChunk::decode_embeddings`] makes) only read what `mtmd_init_from_file` set up.
/// Encoding and projecting a chunk's compute write the context's compute graph, scheduler,
/// buffers, and output embeddings, so this wrapper runs one of them at a time: a second call waits
/// for the first, and embeddings are copied out before another encode can overwrite them.
/// A tokenization or decode on another thread proceeds while an encode runs.
#[derive(Debug)]
pub struct MtmdContext {
    pub(crate) context: NonNull<llama_cpp_sys_2::mtmd_context>,
    /// Width of one output embedding row, the text model's input embedding width, which
    /// `mtmd_init_from_file` checks the projector against.
    embedding_width: usize,
    /// Held by every call that writes the context's compute state.
    compute: Mutex<()>,
}

// SAFETY: the reads upstream allows on a shared context are the only unguarded calls, and
// `compute` serializes every call that writes to it.
unsafe impl Send for MtmdContext {}
unsafe impl Sync for MtmdContext {}

impl MtmdContext {
    /// Initialize MTMD context from a multimodal projection file.
    ///
    /// # Arguments
    ///
    /// * `mmproj_path` - Path to the multimodal projection file
    /// * `text_model` - Reference to the text model
    /// * `params` - Configuration parameters for the MTMD context
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdContext)` on success, or `Err(MtmdInitError)` on failure.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The path cannot be converted to a C string
    /// - `params.device` is not a registered ggml backend device
    /// - The underlying C function returns null (indicating initialization failure)
    pub fn init_from_file(
        mmproj_path: &str,
        text_model: &LlamaModel,
        params: &MtmdContextParams,
    ) -> Result<Self, MtmdInitError> {
        let path_cstr = CString::new(mmproj_path)?;
        if let Some(UnregisteredDevice { index, count }) = unregistered_device(params) {
            return Err(MtmdInitError::InvalidDevice { index, count });
        }
        let ctx_params = llama_cpp_sys_2::mtmd_context_params::from(params);

        let context = unsafe {
            llama_cpp_sys_2::mtmd_init_from_file(
                path_cstr.as_ptr(),
                text_model.model.as_ptr(),
                ctx_params,
            )
        };

        let context = NonNull::new(context).ok_or(MtmdInitError::NullResult)?;
        // SAFETY: `text_model` holds a loaded model for the duration of the borrow.
        let embedding_width =
            unsafe { llama_cpp_sys_2::llama_model_n_embd_inp(text_model.model.as_ptr()) };
        Ok(Self {
            context,
            // A model never reports a negative width; zero makes every readout empty.
            embedding_width: usize::try_from(embedding_width).unwrap_or(0),
            compute: Mutex::new(()),
        })
    }

    /// Waits for any other call that writes the context's compute state, and holds off the next
    /// one until the returned guard drops.
    fn lock_compute(&self) -> MutexGuard<'_, ()> {
        // The lock guards no data of its own, so a panic while it was held leaves nothing torn.
        self.compute.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Projects the memory [`Self::init_from_file`] would allocate for the projector at
    /// `mmproj_path` with `params`, per ggml backend device, without loading its weights.
    ///
    /// Each entry covers the weights plus the compute buffer the projector reserves for its warmup
    /// image, which some projector types cap below their largest input.
    /// This wraps upstream's `mtmd_get_memory_usage`, which llama-server uses to reserve the
    /// projector's memory before fitting a model.
    ///
    /// # Errors
    ///
    /// Returns an error when the path is not a valid C string, `params.device` is not a
    /// registered device, or llama.cpp cannot read the projector or plan its graph.
    pub fn estimate_memory_usage(
        mmproj_path: &str,
        params: &MtmdContextParams,
    ) -> Result<Vec<MtmdDeviceMemory>, MtmdMemoryEstimateError> {
        let path_cstr = CString::new(mmproj_path)?;
        if let Some(UnregisteredDevice { index, count }) = unregistered_device(params) {
            return Err(MtmdMemoryEstimateError::InvalidDevice { index, count });
        }
        let ctx_params = llama_cpp_sys_2::mtmd_context_params::from(params);
        // SAFETY: the registry count takes no arguments.
        let capacity = unsafe { llama_cpp_sys_2::ggml_backend_dev_count() };
        let mut devices = vec![
            llama_cpp_sys_2::llama_rs_mtmd_device_memory {
                device_index: 0,
                bytes: 0,
            };
            capacity
        ];
        let mut device_count = 0_usize;

        // SAFETY: the path and the media marker `ctx_params` points at outlive the call, and
        // `devices` holds as many entries as the length passed with it.
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_estimate_memory_usage(
                path_cstr.as_ptr(),
                &raw const ctx_params,
                devices.as_mut_ptr(),
                devices.len(),
                &raw mut device_count,
            )
        };
        match status {
            llama_cpp_sys_2::LLAMA_RS_MTMD_STATUS_OK => Ok(devices
                .iter()
                .take(device_count)
                .map(|device| MtmdDeviceMemory {
                    device_index: device.device_index,
                    bytes: device.bytes,
                })
                .collect()),
            llama_cpp_sys_2::LLAMA_RS_MTMD_STATUS_UNKNOWN_DEVICE => {
                Err(MtmdMemoryEstimateError::UnknownDevice)
            }
            _ => Err(MtmdMemoryEstimateError::ProjectionFailed),
        }
    }

    /// Projects the compute buffers encoding `chunk` with this context needs, per ggml backend
    /// device, without allocating them.
    ///
    /// An encode grows the projector's compute buffers to its input's graph, and they keep that
    /// size, which can be far beyond the warmup image [`Self::estimate_memory_usage`] covers.
    /// This is the size encoding `chunk` grows them to.
    /// The projection leaves the context's buffers as they are, and it covers only those buffers,
    /// not the scratch memory a backend such as CUDA takes from its own pool during the encode.
    /// A text chunk needs no compute buffers and projects to no devices.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdMemoryEstimateError::ProjectionFailed`] when llama.cpp cannot plan the
    /// chunk's graph, and [`MtmdMemoryEstimateError::UnknownDevice`] when it charges a device
    /// outside the ggml backend registry.
    pub fn estimate_chunk_compute(
        &self,
        chunk: &MtmdInputChunk<'_>,
    ) -> Result<Vec<MtmdDeviceMemory>, MtmdMemoryEstimateError> {
        if chunk.chunk_type() == MtmdInputChunkType::Text {
            return Ok(Vec::new());
        }
        // SAFETY: the registry count takes no arguments.
        let capacity = unsafe { llama_cpp_sys_2::ggml_backend_dev_count() };
        let mut devices = vec![
            llama_cpp_sys_2::llama_rs_mtmd_device_memory {
                device_index: 0,
                bytes: 0,
            };
            capacity
        ];
        let mut device_count = 0_usize;

        // The projection builds the chunk's graph in the context's graph memory.
        let _compute = self.lock_compute();
        // SAFETY: the context and the chunk are live for the duration of the call, and `devices`
        // holds as many entries as the length passed with it.
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_chunk_compute_usage(
                self.context.as_ptr(),
                chunk.chunk.as_ptr(),
                devices.as_mut_ptr(),
                devices.len(),
                &raw mut device_count,
            )
        };
        match status {
            llama_cpp_sys_2::LLAMA_RS_MTMD_STATUS_OK => Ok(devices
                .iter()
                .take(device_count)
                .map(|device| MtmdDeviceMemory {
                    device_index: device.device_index,
                    bytes: device.bytes,
                })
                .collect()),
            llama_cpp_sys_2::LLAMA_RS_MTMD_STATUS_UNKNOWN_DEVICE => {
                Err(MtmdMemoryEstimateError::UnknownDevice)
            }
            _ => Err(MtmdMemoryEstimateError::ProjectionFailed),
        }
    }

    /// Check whether non-causal attention mask is needed before `llama_decode`.
    #[must_use]
    pub fn decode_use_non_causal(&self) -> bool {
        unsafe {
            llama_cpp_sys_2::mtmd_decode_use_non_causal(self.context.as_ptr(), std::ptr::null())
        }
    }

    /// Check whether the current model uses M-RoPE for `llama_decode`.
    ///
    /// M-RoPE (Multimodal Rotary Position Embedding) affects how positions
    /// are calculated for multimodal inputs.
    #[must_use]
    pub fn decode_use_mrope(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_decode_use_mrope(self.context.as_ptr()) }
    }

    /// Check whether the current model supports vision input.
    #[must_use]
    pub fn support_vision(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_support_vision(self.context.as_ptr()) }
    }

    /// Check whether the current model supports audio input.
    #[must_use]
    pub fn support_audio(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_support_audio(self.context.as_ptr()) }
    }

    /// Get audio sample rate in Hz (e.g., 16000 for Whisper).
    /// Returns None if audio is not supported.
    #[must_use]
    pub fn get_audio_sample_rate(&self) -> Option<u32> {
        let rate = unsafe { llama_cpp_sys_2::mtmd_get_audio_sample_rate(self.context.as_ptr()) };
        (rate > 0).then_some(rate.unsigned_abs())
    }

    /// Backward-compatible alias for the audio sample rate getter.
    #[must_use]
    pub fn get_audio_bitrate(&self) -> Option<u32> {
        self.get_audio_sample_rate()
    }

    /// Tokenize input text and bitmaps into chunks.
    ///
    /// The input text must contain media markers (default: `<__media__>`) that will be
    /// replaced with the corresponding bitmap data from the `bitmaps` array.
    /// The number of bitmaps must equal the number of markers in the text.
    ///
    /// # Arguments
    ///
    /// * `text` - Text input configuration containing the text and tokenization options
    /// * `bitmaps` - Array of bitmaps (images/audio) to replace markers with
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdInputChunks)` containing the tokenized chunks on success.
    ///
    /// # Errors
    ///
    /// * `BitmapCountMismatch` - Number of bitmaps doesn't match number of markers
    /// * `ImagePreprocessingError` - Error occurred during image preprocessing
    /// * `UnknownError` - Other tokenization error occurred
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use llama_cpp_2::mtmd::*;
    /// # fn example(ctx: &MtmdContext, bitmap: &MtmdBitmap) -> Result<(), Box<dyn std::error::Error>> {
    /// let text = MtmdInputText {
    ///     text: "Here is an image: <__media__>\nDescribe it.".to_string(),
    ///     add_special: true,
    ///     parse_special: true,
    /// };
    /// let chunks = ctx.tokenize(text, &[bitmap])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn tokenize(
        &self,
        text: MtmdInputText,
        bitmaps: &[&MtmdBitmap],
    ) -> Result<MtmdInputChunks, MtmdTokenizeError> {
        let chunks = MtmdInputChunks::new();
        let text_cstring = CString::new(text.text)?;
        let input_text = llama_cpp_sys_2::mtmd_input_text {
            text: text_cstring.as_ptr(),
            // Byte length excluding the NUL terminator, matching what the C side
            // reads from the pointer above.
            text_len: text_cstring.as_bytes().len(),
            add_special: text.add_special,
            parse_special: text.parse_special,
        };

        // Create bitmap pointers
        let bitmap_ptrs: Vec<*const llama_cpp_sys_2::mtmd_bitmap> = bitmaps
            .iter()
            .map(|b| b.bitmap.as_ptr().cast_const())
            .collect();

        let result = unsafe {
            llama_cpp_sys_2::mtmd_tokenize(
                self.context.as_ptr(),
                chunks.chunks.as_ptr(),
                &raw const input_text,
                bitmap_ptrs.as_ptr().cast_mut(),
                bitmaps.len(),
            )
        };

        match result {
            0 => Ok(chunks),
            1 => Err(MtmdTokenizeError::BitmapCountMismatch),
            2 => Err(MtmdTokenizeError::ImagePreprocessingError),
            _ => Err(MtmdTokenizeError::UnknownError(result)),
        }
    }

    /// Encode a chunk for image/audio processing.
    ///
    /// This function processes image or audio chunks by encoding them into
    /// embeddings that can be used by the language model.
    /// Use [`Self::encode_chunk_embeddings`] to also read the embeddings back.
    ///
    /// # Arguments
    ///
    /// * `chunk` - The input chunk to encode (should be image or audio type)
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Returns `MtmdEncodeError::EncodeFailure` if encoding fails.
    pub fn encode_chunk(&self, chunk: &MtmdInputChunk<'_>) -> Result<(), MtmdEncodeError> {
        let _compute = self.lock_compute();
        self.encode_chunk_locked(chunk)
    }

    /// Encodes `chunk` while the caller holds [`Self::lock_compute`].
    fn encode_chunk_locked(&self, chunk: &MtmdInputChunk<'_>) -> Result<(), MtmdEncodeError> {
        // SAFETY: the context and the chunk are live for the duration of the call, and the
        // caller's compute lock keeps every other writer of the context out.
        let result = unsafe {
            llama_cpp_sys_2::mtmd_encode_chunk(self.context.as_ptr(), chunk.chunk.as_ptr())
        };

        if result == 0 {
            Ok(())
        } else {
            Err(MtmdEncodeError::EncodeFailure(result))
        }
    }

    /// Encode an image or audio chunk and return its embeddings.
    ///
    /// The result holds one row per chunk token, each as wide as the text model's input
    /// embeddings, which is what `mtmd_helper_eval_chunks` decodes into the text model.
    ///
    /// # Errors
    ///
    /// Returns `MtmdEncodeError::TextChunk` for a text chunk, which has no embeddings to encode,
    /// and `MtmdEncodeError::EncodeFailure` if encoding fails.
    pub fn encode_chunk_embeddings(
        &self,
        chunk: &MtmdInputChunk<'_>,
    ) -> Result<Vec<f32>, MtmdEncodeError> {
        // `mtmd_encode_chunk` leaves the output buffer untouched for a text chunk, so reading
        // it would return a previous chunk's embeddings, or read past their end.
        if chunk.chunk_type() == MtmdInputChunkType::Text {
            return Err(MtmdEncodeError::TextChunk);
        }
        // Held until the embeddings are copied out, so no other encode overwrites them first.
        let _compute = self.lock_compute();
        self.encode_chunk_locked(chunk)?;

        let len = chunk.n_tokens() * self.embedding_width;
        // SAFETY: `self.context` is a live context owned by `self`.
        let embeddings = unsafe { llama_cpp_sys_2::mtmd_get_output_embd(self.context.as_ptr()) };
        if embeddings.is_null() || len == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: a successful encode sized the context's output buffer to the chunk's tokens
        // times the projector's output width, which `mtmd_init_from_file` checked against the
        // text model's input width.
        // The buffer stays valid until the next encode, which the compute lock holds off until
        // the copy below has finished.
        Ok(unsafe { slice::from_raw_parts(embeddings, len) }.to_vec())
    }
}

/// A projector device index outside the ggml backend registry.
struct UnregisteredDevice {
    index: usize,
    count: usize,
}

/// Returns the requested projector device when the ggml backend registry does not hold it.
fn unregistered_device(params: &MtmdContextParams) -> Option<UnregisteredDevice> {
    let index = params.device?;
    // SAFETY: the registry count takes no arguments.
    let count = unsafe { llama_cpp_sys_2::ggml_backend_dev_count() };
    (index >= count).then_some(UnregisteredDevice { index, count })
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::mtmd_free(self.context.as_ptr()) }
    }
}

/// Safe wrapper around `mtmd_bitmap`.
///
/// Represents bitmap data for images or audio that can be processed
/// by the multimodal system. For images, data is stored in RGB format.
/// For audio, data is stored as PCM F32 samples.
///
/// Not `Clone`: the wrapper owns the bitmap and frees it on drop, so a copied pointer would be
/// freed twice.
#[derive(Debug)]
pub struct MtmdBitmap {
    pub(crate) bitmap: NonNull<llama_cpp_sys_2::mtmd_bitmap>,
}

// MtmdBitmap is thread safe
unsafe impl Send for MtmdBitmap {}
unsafe impl Sync for MtmdBitmap {}

impl MtmdBitmap {
    /// Create a bitmap from image data in RGB format.
    ///
    /// # Arguments
    ///
    /// * `nx` - Width of the image in pixels
    /// * `ny` - Height of the image in pixels
    /// * `data` - Image data in RGBRGBRGB... format (must be exactly `nx * ny * 3` bytes)
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdBitmap)` on success.
    ///
    /// # Errors
    ///
    /// * `InvalidDataSize` - Data length doesn't match `nx * ny * 3`
    /// * `NullResult` - Underlying C function returned null
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_2::mtmd::MtmdBitmap;
    ///
    /// // Create a 2x2 red image
    /// let red_pixel = [255, 0, 0]; // RGB values for red
    /// let image_data = red_pixel.repeat(4); // 2x2 = 4 pixels
    ///
    /// let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data);
    /// assert!(bitmap.is_ok());
    /// ```
    pub fn from_image_data(nx: u32, ny: u32, data: &[u8]) -> Result<Self, MtmdBitmapError> {
        if data.len() != (nx * ny * 3) as usize {
            return Err(MtmdBitmapError::InvalidDataSize);
        }

        let bitmap = unsafe { llama_cpp_sys_2::mtmd_bitmap_init(nx, ny, data.as_ptr()) };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from audio data in PCM F32 format.
    ///
    /// # Arguments
    ///
    /// * `data` - Audio samples as 32-bit floating point values
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdBitmap)` on success.
    ///
    /// # Errors
    ///
    /// * `NullResult` - Underlying C function returned null
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_2::mtmd::MtmdBitmap;
    ///
    /// // Create a simple sine wave audio sample
    /// let audio_data: Vec<f32> = (0..100)
    ///     .map(|i| (i as f32 * 0.1).sin())
    ///     .collect();
    ///
    /// let bitmap = MtmdBitmap::from_audio_data(&audio_data);
    /// // Note: This will likely fail without proper MTMD context setup
    /// ```
    pub fn from_audio_data(data: &[f32]) -> Result<Self, MtmdBitmapError> {
        let bitmap =
            unsafe { llama_cpp_sys_2::mtmd_bitmap_init_from_audio(data.len(), data.as_ptr()) };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from a file.
    ///
    /// Supported formats:
    /// - Images: formats supported by `stb_image` (jpg, png, bmp, gif, etc.)
    /// - Audio: formats supported by miniaudio (wav, mp3, flac)
    ///
    /// Audio files are auto-detected based on magic bytes.
    ///
    /// # Arguments
    ///
    /// * `ctx` - MTMD context for processing
    /// * `path` - Path to the image or audio file
    /// * `placeholder` - If `true`, build a data-less bitmap (dimensions/length only, with no
    ///   decoded pixels or audio samples) — useful for counting tokens without loading the media.
    ///   If `false`, decode and load the actual data.
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdBitmap)` on success.
    ///
    /// # Errors
    ///
    /// * `CStringError` - Path contains null bytes
    /// * `NullResult` - File could not be loaded or processed
    ///
    /// This function is thread-safe.
    pub fn from_file(
        ctx: &MtmdContext,
        path: &str,
        placeholder: bool,
    ) -> Result<Self, MtmdBitmapError> {
        let path_cstr = CString::new(path)?;
        // This helper now returns a wrapper struct (bitmap + an optional video
        // decoding context) instead of a bare pointer. `video_ctx` is only set
        // when MTMD_VIDEO is enabled at compile time; this crate does not build
        // with it on, so it is always null here and only the bitmap matters.
        let wrapper = unsafe {
            llama_cpp_sys_2::mtmd_helper_bitmap_init_from_file(
                ctx.context.as_ptr(),
                path_cstr.as_ptr(),
                placeholder,
                llama_cpp_sys_2::mtmd_helper_init_opt_default(),
            )
        };

        let bitmap = NonNull::new(wrapper.bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from a buffer containing file data.
    ///
    /// Supported formats:
    /// - Images: formats supported by `stb_image` (jpg, png, bmp, gif, etc.)
    /// - Audio: formats supported by miniaudio (wav, mp3, flac)
    ///
    /// Audio files are auto-detected based on magic bytes.
    ///
    /// # Arguments
    ///
    /// * `ctx` - MTMD context for processing
    /// * `data` - Buffer containing the file data
    /// * `placeholder` - If `true`, build a data-less bitmap (dimensions/length only, with no
    ///   decoded pixels or audio samples) — useful for counting tokens without loading the media.
    ///   If `false`, decode and load the actual data.
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdBitmap)` on success.
    ///
    /// # Errors
    ///
    /// * `NullResult` - Buffer could not be processed
    ///
    /// This function is thread-safe.
    pub fn from_buffer(
        ctx: &MtmdContext,
        data: &[u8],
        placeholder: bool,
    ) -> Result<Self, MtmdBitmapError> {
        // See the comment in `from_file`: this returns a wrapper struct now, and
        // `video_ctx` is always null since MTMD_VIDEO is not enabled in this build.
        let wrapper = unsafe {
            llama_cpp_sys_2::mtmd_helper_bitmap_init_from_buf(
                ctx.context.as_ptr(),
                data.as_ptr(),
                data.len(),
                placeholder,
                llama_cpp_sys_2::mtmd_helper_init_opt_default(),
            )
        };

        let bitmap = NonNull::new(wrapper.bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Get bitmap width in pixels.
    #[must_use]
    pub fn nx(&self) -> u32 {
        unsafe { llama_cpp_sys_2::mtmd_bitmap_get_nx(self.bitmap.as_ptr()) }
    }

    /// Get bitmap height in pixels.
    #[must_use]
    pub fn ny(&self) -> u32 {
        unsafe { llama_cpp_sys_2::mtmd_bitmap_get_ny(self.bitmap.as_ptr()) }
    }

    /// Get bitmap data as a byte slice.
    ///
    /// For images: RGB format with length `nx * ny * 3`
    /// For audio: PCM F32 format with length `n_samples * 4`
    #[must_use]
    pub fn data(&self) -> &[u8] {
        let ptr = unsafe { llama_cpp_sys_2::mtmd_bitmap_get_data(self.bitmap.as_ptr()) };
        let len = unsafe { llama_cpp_sys_2::mtmd_bitmap_get_n_bytes(self.bitmap.as_ptr()) };
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    /// Check if this bitmap contains audio data (vs image data).
    #[must_use]
    pub fn is_audio(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_bitmap_is_audio(self.bitmap.as_ptr()) }
    }

    /// Get the bitmap's optional ID string.
    ///
    /// Bitmap ID is useful for KV cache tracking and can e.g. be calculated
    /// based on a hash of the bitmap data.
    #[must_use]
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { llama_cpp_sys_2::mtmd_bitmap_get_id(self.bitmap.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            let id = unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned();
            Some(id)
        }
    }

    /// Set the bitmap's ID string.
    ///
    /// Bitmap ID is useful for KV cache tracking and can e.g. be calculated
    /// based on a hash of the bitmap data.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID string to set
    ///
    /// # Errors
    ///
    /// Returns an error if the ID string contains null bytes.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use llama_cpp_2::mtmd::MtmdBitmap;
    /// # fn example(bitmap: &MtmdBitmap) -> Result<(), Box<dyn std::error::Error>> {
    /// bitmap.set_id("image_001")?;
    /// assert_eq!(bitmap.id(), Some("image_001".to_string()));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_id(&self, id: &str) -> Result<(), std::ffi::NulError> {
        let id_cstr = CString::new(id)?;
        unsafe {
            llama_cpp_sys_2::mtmd_bitmap_set_id(self.bitmap.as_ptr(), id_cstr.as_ptr());
        }
        Ok(())
    }
}

impl Drop for MtmdBitmap {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::mtmd_bitmap_free(self.bitmap.as_ptr()) }
    }
}

/// Safe wrapper around `mtmd_input_chunks`.
///
/// This is a collection of input chunks created from tokenizing text and media.
/// The chunks represent the tokenized input that can be processed by the model,
/// with text chunks containing tokens and media chunks containing embeddings.
#[derive(Debug)]
pub struct MtmdInputChunks {
    pub(crate) chunks: NonNull<llama_cpp_sys_2::mtmd_input_chunks>,
}

impl Default for MtmdInputChunks {
    fn default() -> Self {
        Self::new()
    }
}

impl MtmdInputChunks {
    /// Create a new empty input chunks collection
    /// # Panics
    /// This function will panic if the underlying llama.cpp function returns null,
    /// which should not happen.
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_2::mtmd::MtmdInputChunks;
    ///
    /// let chunks = MtmdInputChunks::new();
    /// assert_eq!(chunks.len(), 0);
    /// assert!(chunks.is_empty());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        let chunks = unsafe { llama_cpp_sys_2::mtmd_input_chunks_init() };
        let chunks = NonNull::new(chunks).unwrap();
        Self { chunks }
    }

    /// Get the number of chunks
    #[must_use]
    pub fn len(&self) -> usize {
        unsafe { llama_cpp_sys_2::mtmd_input_chunks_size(self.chunks.as_ptr()) }
    }

    /// Check if chunks collection is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a chunk by index.
    ///
    /// The returned chunk borrows from this collection. Dropping the collection
    /// while a borrowed chunk is still in use is a compile error. Use
    /// [`MtmdInputChunk::copy`] if you need a chunk that outlives the collection.
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_2::mtmd::MtmdInputChunks;
    ///
    /// let chunks = MtmdInputChunks::new();
    /// assert!(chunks.get(0).is_none());
    /// ```
    ///
    /// ```compile_fail
    /// use llama_cpp_2::mtmd::MtmdInputChunks;
    ///
    /// let chunks = MtmdInputChunks::new();
    /// let chunk = chunks.get(0);
    /// drop(chunks);
    /// let _ = chunk;
    /// ```
    #[must_use]
    pub fn get(&self, index: usize) -> Option<MtmdInputChunk<'_>> {
        if index >= self.len() {
            return None;
        }

        let chunk_ptr =
            unsafe { llama_cpp_sys_2::mtmd_input_chunks_get(self.chunks.as_ptr(), index) };

        // Note: We don't own this chunk, it's owned by the chunks collection
        NonNull::new(chunk_ptr.cast_mut()).map(|ptr| MtmdInputChunk {
            chunk: ptr,
            owned: false,
            phantom: PhantomData,
        })
    }

    /// Get total number of tokens across all chunks.
    ///
    /// This is useful for keeping track of KV cache size.
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        unsafe { llama_cpp_sys_2::mtmd_helper_get_n_tokens(self.chunks.as_ptr()) }
    }

    /// Get total position count across all chunks.
    ///
    /// This is useful to keep track of `n_past`. Normally `n_pos` equals `n_tokens`,
    /// but for M-RoPE it is different.
    #[must_use]
    pub fn total_positions(&self) -> i32 {
        unsafe { llama_cpp_sys_2::mtmd_helper_get_n_pos(self.chunks.as_ptr()) }
    }

    /// Evaluate chunks using the multimodal context and LLAMA context.
    ///
    /// This helper function automatically:
    /// 1. Runs `llama_decode()` on text chunks
    /// 2. Runs `mtmd_encode()` on image chunks, then `mtmd_get_output_embd()` and then `llama_decode()`
    ///
    /// If any of the `mtmd_encode()` or `llama_decode()` calls return non-zero, the function
    /// stops and forwards the error.
    ///
    /// # Arguments
    ///
    /// * `mtmd_ctx` - The multimodal context
    /// * `llama_ctx` - The LLAMA context
    /// * `n_past` - Current position in the sequence
    /// * `seq_id` - Sequence ID for the batch
    /// * `n_batch` - Batch size for processing
    /// * `logits_last` - Whether to compute logits for the last token only
    ///
    /// # Returns
    ///
    /// Returns the new `n_past` value on success.
    ///
    /// # Errors
    ///
    /// Returns `MtmdEvalError::EvalFailure` if any encoding or decoding operation fails.
    ///
    /// This function is NOT thread-safe.
    pub fn eval_chunks(
        &self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &LlamaContext,
        n_past: llama_cpp_sys_2::llama_pos,
        seq_id: llama_cpp_sys_2::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_sys_2::llama_pos, MtmdEvalError> {
        let mut new_n_past: llama_cpp_sys_2::llama_pos = 0;

        // Media chunks are encoded with the context along the way.
        let _compute = mtmd_ctx.lock_compute();
        let result = unsafe {
            llama_cpp_sys_2::mtmd_helper_eval_chunks(
                mtmd_ctx.context.as_ptr(),
                llama_ctx.context.as_ptr(),
                self.chunks.as_ptr(),
                n_past,
                seq_id,
                n_batch,
                logits_last,
                &raw mut new_n_past,
            )
        };

        if result == 0 {
            Ok(new_n_past)
        } else {
            Err(MtmdEvalError::EvalFailure(result))
        }
    }
}

// SAFETY: the collection owns its chunks and no thread-local state, and every `&self` method only
// reads it.
unsafe impl Send for MtmdInputChunks {}
unsafe impl Sync for MtmdInputChunks {}

impl Drop for MtmdInputChunks {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::mtmd_input_chunks_free(self.chunks.as_ptr()) }
    }
}

/// Safe wrapper around `mtmd_input_chunk`.
///
/// Represents a single chunk of input data, which can be either text tokens,
/// image tokens, or audio tokens. The chunk type determines what kind of
/// data and operations are available.
///
/// `'a` is the lifetime of the [`MtmdInputChunks`] this chunk was borrowed
/// from via [`MtmdInputChunks::get`]. [`MtmdInputChunk::copy`] returns an
/// owned chunk with a `'static` lifetime that is freed independently.
#[derive(Debug)]
pub struct MtmdInputChunk<'a> {
    pub(crate) chunk: NonNull<llama_cpp_sys_2::mtmd_input_chunk>,
    owned: bool,
    phantom: PhantomData<&'a MtmdInputChunks>,
}

// SAFETY: every `&self` method only reads the chunk, an owned chunk is freed only by its own drop,
// and a borrowed one is tied to its collection, which is `Sync`.
unsafe impl Send for MtmdInputChunk<'_> {}
unsafe impl Sync for MtmdInputChunk<'_> {}

impl MtmdInputChunk<'_> {
    /// Get the type of this chunk
    #[must_use]
    pub fn chunk_type(&self) -> MtmdInputChunkType {
        let chunk_type = unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_type(self.chunk.as_ptr()) };
        MtmdInputChunkType::from(chunk_type)
    }

    /// Get text tokens from this chunk.
    ///
    /// Only valid for text chunks. Returns `None` for image or audio chunks.
    ///
    /// # Returns
    ///
    /// Returns `Some(&[LlamaToken])` for text chunks, `None` otherwise.
    #[must_use]
    pub fn text_tokens(&self) -> Option<&[LlamaToken]> {
        if self.chunk_type() != MtmdInputChunkType::Text {
            return None;
        }

        let mut n_tokens = 0usize;
        let tokens_ptr = unsafe {
            llama_cpp_sys_2::mtmd_input_chunk_get_tokens_text(
                self.chunk.as_ptr(),
                &raw mut n_tokens,
            )
        };

        if tokens_ptr.is_null() || n_tokens == 0 {
            None
        } else {
            unsafe {
                Some(slice::from_raw_parts(
                    tokens_ptr.cast::<LlamaToken>(),
                    n_tokens,
                ))
            }
        }
    }

    /// Get the number of tokens in this chunk
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_n_tokens(self.chunk.as_ptr()) }
    }

    /// Get the number of positions in this chunk.
    ///
    /// Returns the number of temporal positions (always 1 for M-RoPE, `n_tokens` otherwise).
    #[must_use]
    pub fn n_positions(&self) -> i32 {
        unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_n_pos(self.chunk.as_ptr()) }
    }

    /// Get chunk ID if available.
    ///
    /// Returns `None` for text chunks, may return an ID for image/audio chunks.
    #[must_use]
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_id(self.chunk.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
                .into()
        }
    }

    /// Evaluates this chunk into `llama_ctx` at position `n_past` of sequence `seq_id`, and returns
    /// the position after it.
    ///
    /// A text chunk is decoded in batches of at most `n_batch` tokens, with logits for its last
    /// token when `logits_last` is set.
    /// A media chunk is encoded with `mtmd_ctx` and its embeddings decoded.
    /// This wraps `mtmd_helper_eval_chunk_single`, the per-chunk step of
    /// [`MtmdInputChunks::eval_chunks`], for callers that treat chunks differently.
    ///
    /// `llama_ctx` is borrowed mutably, as [`LlamaContext::decode`] borrows it, because the decode
    /// overwrites the logits and embeddings that earlier reads of the context return.
    ///
    /// This function is NOT thread-safe.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdEvalError::InvalidBatchSize`] for a batch size below one, and
    /// [`MtmdEvalError::EvalFailure`] when encoding or decoding fails.
    pub fn eval(
        &self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &mut LlamaContext,
        n_past: llama_cpp_sys_2::llama_pos,
        seq_id: llama_cpp_sys_2::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_sys_2::llama_pos, MtmdEvalError> {
        // llama.cpp asserts on a batch size below one, which aborts the process.
        if n_batch < 1 {
            return Err(MtmdEvalError::InvalidBatchSize(n_batch));
        }
        // The helper advances a text chunk's position from the value it finds here.
        let mut new_n_past = n_past;
        // A media chunk is encoded with the context first; a text chunk leaves it untouched.
        let _compute =
            (self.chunk_type() != MtmdInputChunkType::Text).then(|| mtmd_ctx.lock_compute());
        // SAFETY: every pointer is live for the duration of the call, the compute lock keeps
        // other writers of the context out while a media chunk is encoded, and the helper writes
        // only `new_n_past`.
        let result = unsafe {
            llama_cpp_sys_2::mtmd_helper_eval_chunk_single(
                mtmd_ctx.context.as_ptr(),
                llama_ctx.context.as_ptr(),
                self.chunk.as_ptr(),
                n_past,
                seq_id,
                n_batch,
                logits_last,
                &raw mut new_n_past,
            )
        };
        if result == 0 {
            Ok(new_n_past)
        } else {
            Err(MtmdEvalError::EvalFailure(result))
        }
    }

    /// Decodes this media chunk from embeddings encoded beforehand into `llama_ctx` at position
    /// `n_past` of sequence `seq_id`, and returns the position after it.
    ///
    /// `embeddings` holds one row per chunk token, each as wide as the text model's input
    /// embeddings, as [`MtmdContext::encode_chunk_embeddings`] returns them.
    /// `mtmd_ctx` supplies the model-specific decoding, such as non-causal attention and M-RoPE
    /// positions, so any context of the projector that encoded the embeddings serves.
    /// Together with [`MtmdContext::encode_chunk_embeddings`] this splits [`Self::eval`] of a
    /// media chunk in two, so the encode can run on a different projector context than the one
    /// the chunk was tokenized with, or on another thread while this one decodes.
    /// This wraps `mtmd_helper_decode_image_chunk`, which only reads `mtmd_ctx`, so the decode
    /// does not wait for an encode running on the same context.
    ///
    /// `llama_ctx` is borrowed mutably, as [`LlamaContext::decode`] borrows it, because the decode
    /// overwrites the logits and embeddings that earlier reads of the context return.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdEvalError::TextChunk`] for a text chunk,
    /// [`MtmdEvalError::EmbeddingsLength`] when `embeddings` does not hold one row per token,
    /// [`MtmdEvalError::InvalidBatchSize`] for a batch size below one, and
    /// [`MtmdEvalError::EvalFailure`] when decoding fails.
    pub fn decode_embeddings(
        &self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &mut LlamaContext,
        embeddings: &[f32],
        n_past: llama_cpp_sys_2::llama_pos,
        seq_id: llama_cpp_sys_2::llama_seq_id,
        n_batch: i32,
    ) -> Result<llama_cpp_sys_2::llama_pos, MtmdEvalError> {
        if self.chunk_type() == MtmdInputChunkType::Text {
            return Err(MtmdEvalError::TextChunk);
        }
        // llama.cpp asserts on a batch size below one, which aborts the process.
        if n_batch < 1 {
            return Err(MtmdEvalError::InvalidBatchSize(n_batch));
        }
        // SAFETY: the context holds a loaded model for as long as it lives.
        let embedding_width = unsafe {
            llama_cpp_sys_2::llama_model_n_embd_inp(llama_cpp_sys_2::llama_get_model(
                llama_ctx.context.as_ptr(),
            ))
        };
        let expected = self
            .n_tokens()
            .saturating_mul(usize::try_from(embedding_width).unwrap_or(0));
        if embeddings.len() != expected {
            return Err(MtmdEvalError::EmbeddingsLength {
                expected,
                actual: embeddings.len(),
            });
        }
        let mut new_n_past = n_past;
        // SAFETY: every pointer is live for the duration of the call, `embeddings` holds the
        // `n_tokens * n_embd_inp` values the helper reads and never writes, and the helper writes
        // only `new_n_past`.
        let result = unsafe {
            llama_cpp_sys_2::mtmd_helper_decode_image_chunk(
                mtmd_ctx.context.as_ptr(),
                llama_ctx.context.as_ptr(),
                self.chunk.as_ptr(),
                embeddings.as_ptr().cast_mut(),
                n_past,
                seq_id,
                n_batch,
                &raw mut new_n_past,
                None,
                std::ptr::null_mut(),
            )
        };
        if result == 0 {
            Ok(new_n_past)
        } else {
            Err(MtmdEvalError::EvalFailure(result))
        }
    }

    /// Create a copy of this chunk that you own.
    ///
    /// This is useful if you want to use custom logic to handle the chunk
    /// (e.g., KV cache management) by moving the chunk ownership to your own code.
    /// Remember to ensure the copied chunk is properly freed when you're done with it.
    ///
    /// The copy does not borrow from the original collection, so it can
    /// outlive the [`MtmdInputChunks`] this chunk was taken from.
    ///
    /// # Returns
    ///
    /// Returns an owned copy of the chunk.
    ///
    /// # Errors
    ///
    /// Returns `MtmdInputChunkError::NullResult` if copying fails.
    pub fn copy(&self) -> Result<MtmdInputChunk<'static>, MtmdInputChunkError> {
        let chunk = unsafe { llama_cpp_sys_2::mtmd_input_chunk_copy(self.chunk.as_ptr()) };
        let chunk = NonNull::new(chunk).ok_or(MtmdInputChunkError::NullResult)?;
        Ok(MtmdInputChunk {
            chunk,
            owned: true,
            phantom: PhantomData,
        })
    }
}

impl Drop for MtmdInputChunk<'_> {
    fn drop(&mut self) {
        if self.owned {
            unsafe { llama_cpp_sys_2::mtmd_input_chunk_free(self.chunk.as_ptr()) }
        }
    }
}

/// Get the default media marker string.
///
/// Returns the default marker used to identify media positions in text
/// (typically `"<__media__>"`). This marker should be used in your input text
/// to indicate where media content should be inserted.
///
/// # Returns
///
/// Returns the default media marker as a string slice.
///
/// # Examples
///
/// ```
/// use llama_cpp_2::mtmd::mtmd_default_marker;
///
/// let marker = mtmd_default_marker();
/// assert!(!marker.is_empty());
///
/// let text = format!("Describe this image: {}", marker);
/// assert!(text.contains(marker));
/// ```
#[must_use]
pub fn mtmd_default_marker() -> &'static str {
    unsafe {
        let c_str = llama_cpp_sys_2::mtmd_default_marker();
        CStr::from_ptr(c_str).to_str().unwrap_or("<__media__>")
    }
}

// Error types
/// Errors that can occur when initializing MTMD context
#[derive(thiserror::Error, Debug)]
pub enum MtmdInitError {
    /// Failed to create `CString` from input
    #[error("Failed to create CString: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// MTMD context initialization returned null
    #[error("MTMD context initialization returned null")]
    NullResult,
    /// The requested projector device is not in the ggml backend registry
    #[error("ggml backend device {index} does not exist; {count} devices are registered")]
    InvalidDevice {
        /// The requested device index
        index: usize,
        /// Number of registered ggml backend devices
        count: usize,
    },
}

/// Errors that can occur when projecting a projector's memory
#[derive(thiserror::Error, Debug)]
pub enum MtmdMemoryEstimateError {
    /// Failed to create `CString` from input
    #[error("Failed to create CString: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// The requested projector device is not in the ggml backend registry
    #[error("ggml backend device {index} does not exist; {count} devices are registered")]
    InvalidDevice {
        /// The requested device index
        index: usize,
        /// Number of registered ggml backend devices
        count: usize,
    },
    /// llama.cpp could not read the projector or plan its graph
    #[error("llama.cpp could not project the projector's memory")]
    ProjectionFailed,
    /// llama.cpp charged memory to a device outside the ggml backend registry
    #[error("the projector's memory was charged to an unregistered ggml backend device")]
    UnknownDevice,
}

/// Errors that can occur when working with MTMD bitmaps
#[derive(thiserror::Error, Debug)]
pub enum MtmdBitmapError {
    /// Failed to create `CString` from input
    #[error("Failed to create CString: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// Invalid data size for bitmap
    #[error("Invalid data size for bitmap")]
    InvalidDataSize,
    /// Bitmap creation returned null
    #[error("Bitmap creation returned null")]
    NullResult,
}

/// Errors that can occur when working with MTMD input chunks collections
#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunksError {
    /// Input chunks creation returned null
    #[error("Input chunks creation returned null")]
    NullResult,
}

/// Errors that can occur when working with individual MTMD input chunks
#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunkError {
    /// Input chunk operation returned null
    #[error("Input chunk operation returned null")]
    NullResult,
}

/// Errors that can occur during tokenization
#[derive(thiserror::Error, Debug)]
pub enum MtmdTokenizeError {
    /// Number of bitmaps does not match number of markers in text
    #[error("Number of bitmaps does not match number of markers")]
    BitmapCountMismatch,
    /// Image preprocessing error occurred
    #[error("Image preprocessing error")]
    ImagePreprocessingError,
    /// Text contains characters that cannot be converted to C string
    #[error("Failed to create CString from text: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// Unknown error occurred during tokenization
    #[error("Unknown error: {0}")]
    UnknownError(i32),
}

/// Errors that can occur during encoding
#[derive(thiserror::Error, Debug)]
pub enum MtmdEncodeError {
    /// Encode operation failed
    #[error("Encode failed with code: {0}")]
    EncodeFailure(i32),
    /// Text chunks carry tokens, not embeddings to encode
    #[error("text chunks have no embeddings to encode")]
    TextChunk,
}

/// Errors that can occur during evaluation
#[derive(thiserror::Error, Debug)]
pub enum MtmdEvalError {
    /// Evaluation operation failed
    #[error("Eval failed with code: {0}")]
    EvalFailure(i32),
    /// Text chunks carry tokens, not embeddings to decode
    #[error("text chunks have no embeddings to decode")]
    TextChunk,
    /// The embeddings do not hold one row per chunk token
    #[error("expected {expected} embedding values, got {actual}")]
    EmbeddingsLength {
        /// Values one row per chunk token takes
        expected: usize,
        /// Values supplied
        actual: usize,
    },
    /// Chunks are decoded in batches of at least one token
    #[error("batch size {0} is below one")]
    InvalidBatchSize(i32),
}
