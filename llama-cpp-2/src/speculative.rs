//! Speculative decoding with a model's own multi-token-prediction (MTP) head.
//!
//! A model whose GGUF carries `NextN` layers can predict tokens beyond the next one.
//! llama.cpp runs that head in a second,
//! [`LlamaContextType::Mtp`](crate::context::params::LlamaContextType) context over the same
//! weights, fed with the hidden states the target context produces.
//! [`MtpDraftContext::new`] creates that context bound to its target, and this module drives
//! llama.cpp's MTP drafter for any number of sequences at once:
//!
//! - [`MtpSpeculative::process`] replays every batch the target decoded through the drafter, so its
//!   context holds the same positions as the target's.
//! - [`MtpSpeculative::draft`] proposes draft tokens for several sequences in one pass.
//! - [`MtpSpeculative::accept`] reports how many drafted tokens the target accepted.
//!
//! Verifying the drafts, sampling, and removing rejected tokens from both contexts is the caller's
//! job, because only the caller knows its sampler chains.

use std::ffi::CString;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::context::params::LlamaContextParams;
use crate::context::LlamaContext;
use crate::llama_backend::LlamaBackend;
use crate::llama_batch::LlamaBatch;
use crate::model::LlamaModel;
use crate::token::LlamaToken;
use crate::{status_is_ok, LlamaContextLoadError};

/// Whether the GGUF file at `path` carries multi-token-prediction layers llama.cpp can draft with.
///
/// Many quantizations strip those layers, so this is decided from the file rather than the
/// architecture, by llama.cpp's own detection.
/// Returns `false` for a path or file that cannot be read.
#[must_use]
pub fn model_file_has_mtp_layers(path: &Path) -> bool {
    let Some(path) = path.to_str().and_then(|path| CString::new(path).ok()) else {
        return false;
    };
    // SAFETY: `path` is a valid NUL-terminated string for the duration of the call.
    unsafe { llama_cpp_sys_2::llama_rs_model_file_has_mtp_layers(path.as_ptr()) }
}

/// How a context's memory can remove positions from a sequence.
///
/// Speculative decoding has to remove the draft tokens the target rejects, which only
/// [`Self::Any`] and, within its bound, [`Self::Newest`] allow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceRemoval {
    /// The context has no memory to remove from.
    Unsupported,
    /// Any range of positions.
    Any,
    /// Whole sequences only, as for a recurrent model without rollback snapshots.
    WholeSequenceOnly,
    /// The newest [`LlamaContext::n_rs_seq`] positions, through rollback snapshots.
    Newest,
}

/// Classifies how `context` can remove positions, the way llama.cpp's speculative decoding does.
///
/// The probe decodes into the context and then clears its memory, so it belongs right after the
/// context is created, before it holds anything worth keeping.
pub fn sequence_removal(context: &mut LlamaContext<'_>) -> SequenceRemoval {
    // SAFETY: A live `LlamaContext` owns a valid context pointer, and the exclusive borrow keeps
    // anything else from using it during the probe.
    let removal =
        unsafe { llama_cpp_sys_2::llama_rs_context_seq_rm_type(context.context.as_ptr()) };
    match removal {
        1 => SequenceRemoval::Any,
        2 => SequenceRemoval::WholeSequenceOnly,
        3 => SequenceRemoval::Newest,
        _ => SequenceRemoval::Unsupported,
    }
}

/// An MTP draft context bound to the target context it drafts for.
///
/// llama.cpp's own speculative setup (`common_speculative_init_result`) creates every draft context
/// with its target as `ctx_other`.
/// Architectures whose drafter works on the target's memory, such as the Gemma 4 assistant,
/// require that, and every other architecture ignores it.
/// A bound draft context may keep the target's address, so it decodes only inside the
/// [`MtpSpeculative`] that owns that very target, which [`MtpSpeculative::new`] checks.
#[derive(Debug)]
pub struct MtpDraftContext<'model> {
    context: LlamaContext<'model>,
    /// Address of the target's llama.cpp context, compared but never dereferenced.
    target: usize,
}

impl MtpDraftContext<'static> {
    /// Creates the draft context for `target` over `model`, which shares its weights.
    ///
    /// `params` should be an [`Mtp`](crate::context::params::LlamaContextType::Mtp) copy of the
    /// target's parameters, with the same context size and sequence count.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaContextLoadError::NullReturn`] when llama.cpp cannot create the context,
    /// for example because the model was loaded without its MTP layers.
    pub fn new(
        model: Arc<LlamaModel>,
        _backend: &LlamaBackend,
        target: &LlamaContext<'_>,
        params: &LlamaContextParams,
    ) -> Result<Self, LlamaContextLoadError> {
        let raw_params = bound_draft_params(params, target.context);
        // SAFETY: The model pointer is live for the call, and `ctx_other` points at the live
        // target context.
        // The returned context keeps the model alive through `model`, and it only ever decodes
        // inside an `MtpSpeculative` that owns the target it points at.
        let context = unsafe {
            llama_cpp_sys_2::llama_new_context_with_model(model.model.as_ptr(), raw_params)
        };
        let context = NonNull::new(context).ok_or(LlamaContextLoadError::NullReturn)?;
        Ok(Self {
            context: LlamaContext::new_owned(model, context, params.embeddings()),
            target: target.context.as_ptr().addr(),
        })
    }
}

impl<'model> MtpDraftContext<'model> {
    /// The draft context, for reading what it holds or occupies.
    #[must_use]
    pub fn context(&self) -> &LlamaContext<'model> {
        &self.context
    }

    /// Whether this draft context was created for `target`.
    #[must_use]
    pub fn is_bound_to(&self, target: &LlamaContext<'_>) -> bool {
        self.target == target.context.as_ptr().addr()
    }
}

/// The llama.cpp parameters of a draft context bound to `target`.
fn bound_draft_params(
    params: &LlamaContextParams,
    target: NonNull<llama_cpp_sys_2::llama_context>,
) -> llama_cpp_sys_2::llama_context_params {
    let mut raw_params = params.context_params;
    raw_params.ctx_other = target.as_ptr();
    raw_params
}

/// Parameters for same-model MTP speculative decoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtpSpeculativeParams {
    /// Maximum number of draft tokens to propose per sequence.
    pub n_max: i32,
    /// Minimum number of draft tokens required before returning a draft.
    pub n_min: i32,
    /// Minimum draft probability accepted by llama.cpp's MTP drafter.
    pub p_min: f32,
}

impl Default for MtpSpeculativeParams {
    fn default() -> Self {
        Self {
            n_max: 3,
            n_min: 0,
            p_min: 0.0,
        }
    }
}

/// Errors returned by the MTP speculative wrapper.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MtpSpeculativeError {
    /// Invalid parameters were provided.
    #[error("invalid MTP speculative parameters")]
    InvalidParams,
    /// llama.cpp returned a null speculative handle.
    #[error("llama.cpp failed to initialize MTP speculative decoding")]
    InitFailed,
    /// llama.cpp rejected a wrapper call.
    #[error("llama.cpp MTP speculative call failed with status {0}")]
    Status(i32),
}

/// A failed [`MtpSpeculative::new`], handing back the contexts it was given.
///
/// Creating a context is expensive, so a caller that falls back to decoding without speculation
/// keeps using the target context instead of building a new one.
#[derive(Debug)]
pub struct MtpSpeculativeInitError<'model> {
    /// Why the speculative state could not be created.
    pub error: MtpSpeculativeError,
    /// The draft context, unchanged.
    ///
    /// Declared before the target so that it is dropped first: it may point at the target.
    pub draft_context: MtpDraftContext<'model>,
    /// The target context, unchanged.
    pub target_context: LlamaContext<'model>,
}

/// One sequence's request to [`MtpSpeculative::draft`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpDraftRequest {
    /// Sequence to draft for.
    pub seq_id: i32,
    /// Position the last sampled token takes when the target decodes it.
    pub pos0: i32,
    /// Last sampled token, which the target has not decoded yet.
    pub id_last: LlamaToken,
    /// Most draft tokens for this sequence, at most [`MtpSpeculative::n_max`].
    ///
    /// The drafter stops decoding the sequence at this depth, so a shallower request also takes
    /// fewer draft steps.
    pub n_max: i32,
}

/// Drafts from one [`MtpSpeculative::draft`] call, in request order.
#[derive(Debug)]
pub struct MtpDrafts<'a> {
    tokens: &'a [LlamaToken],
    counts: &'a [usize],
    stride: usize,
}

impl MtpDrafts<'_> {
    /// Draft tokens for the request at `index`, empty when the drafter proposed none.
    #[must_use]
    pub fn get(&self, index: usize) -> &[LlamaToken] {
        let Some(count) = self.counts.get(index) else {
            return &[];
        };
        let start = index.saturating_mul(self.stride);
        self.tokens
            .get(start..start.saturating_add(*count))
            .unwrap_or(&[])
    }
}

/// RAII owner of a same-model MTP drafter and the two contexts it runs between.
///
/// The drafter and the draft context keep the target context's address, and the drafter reads its
/// hidden states on every [`Self::process`], which is why this owns that context rather than
/// borrowing it: nothing can drop or replace the target while either still points at it.
#[derive(Debug)]
pub struct MtpSpeculative<'model> {
    raw: NonNull<llama_cpp_sys_2::llama_rs_mtp_speculative>,
    // Declared before the target so that it is dropped first: it may point at the target.
    draft_context: MtpDraftContext<'model>,
    target_context: LlamaContext<'model>,
    n_max: usize,
    raw_requests: Vec<llama_cpp_sys_2::llama_rs_mtp_draft_request>,
    raw_tokens: Vec<llama_cpp_sys_2::llama_token>,
    draft_tokens: Vec<LlamaToken>,
    draft_counts: Vec<usize>,
}

impl<'model> MtpSpeculative<'model> {
    /// Creates an MTP drafter for sequences `0..n_seq` of `target_context`.
    ///
    /// `draft_context` must have been created for `target_context`, with at least `n_seq`
    /// sequences and a batch size no smaller than the target's.
    ///
    /// # Errors
    ///
    /// Returns [`MtpSpeculativeError::InvalidParams`] for out-of-range parameters or a draft
    /// context bound to another target, and [`MtpSpeculativeError::InitFailed`] when llama.cpp
    /// cannot create the drafter, together with both contexts.
    pub fn new(
        target_context: LlamaContext<'model>,
        draft_context: MtpDraftContext<'model>,
        params: MtpSpeculativeParams,
        n_seq: u32,
    ) -> Result<Self, Box<MtpSpeculativeInitError<'model>>> {
        let Some(n_max) = usize::try_from(params.n_max)
            .ok()
            .filter(|n_max| *n_max > 0)
            .filter(|_| params.n_min >= 0 && params.n_min <= params.n_max && n_seq > 0)
            .filter(|_| draft_context.is_bound_to(&target_context))
        else {
            return Err(Box::new(MtpSpeculativeInitError {
                error: MtpSpeculativeError::InvalidParams,
                target_context,
                draft_context,
            }));
        };

        let raw = unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_init(
                target_context.context.as_ptr(),
                draft_context.context.context.as_ptr(),
                params.n_max,
                params.n_min,
                params.p_min,
                n_seq,
            )
        };
        let Some(raw) = NonNull::new(raw) else {
            return Err(Box::new(MtpSpeculativeInitError {
                error: MtpSpeculativeError::InitFailed,
                target_context,
                draft_context,
            }));
        };

        Ok(Self {
            raw,
            target_context,
            draft_context,
            n_max,
            raw_requests: Vec::new(),
            raw_tokens: Vec::new(),
            draft_tokens: Vec::new(),
            draft_counts: Vec::new(),
        })
    }

    /// The most draft tokens one request can receive.
    #[must_use]
    pub fn n_max(&self) -> usize {
        self.n_max
    }

    /// Access the target context.
    #[must_use]
    pub fn target_context(&self) -> &LlamaContext<'model> {
        &self.target_context
    }

    /// Access the target context for decode and cache rollback operations.
    pub fn target_context_mut(&mut self) -> &mut LlamaContext<'model> {
        &mut self.target_context
    }

    /// Access the draft context for cache rollback operations.
    pub fn draft_context_mut(&mut self) -> &mut LlamaContext<'model> {
        &mut self.draft_context.context
    }

    /// Replays a batch the target context just decoded through the drafter.
    ///
    /// Every batch the target decodes must pass through here, in order, for the draft context to
    /// hold the same positions.
    /// Each token must belong to exactly one sequence, and each sequence's tokens must form one run
    /// of consecutive positions.
    ///
    /// # Errors
    ///
    /// Returns [`MtpSpeculativeError::Status`] when the batch breaks that layout, is larger than
    /// the draft context's batch size, or the draft decode fails.
    pub fn process(&mut self, batch: &LlamaBatch<'_>) -> Result<(), MtpSpeculativeError> {
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_process(
                self.raw.as_ptr(),
                std::ptr::from_ref(&batch.llama_batch),
            )
        };
        status_to_result(status)
    }

    /// Drafts tokens for every request in one pass.
    ///
    /// Drafting decodes into the draft context, and those positions are removed again before this
    /// returns, so the draft context still mirrors the target.
    ///
    /// # Errors
    ///
    /// Returns [`MtpSpeculativeError::InvalidParams`] for a request outside the configured
    /// sequences or limits, including two requests for one sequence, and
    /// [`MtpSpeculativeError::Status`] when llama.cpp rejects the call.
    pub fn draft(
        &mut self,
        requests: &[MtpDraftRequest],
    ) -> Result<MtpDrafts<'_>, MtpSpeculativeError> {
        let n_max = i32::try_from(self.n_max).map_err(|_| MtpSpeculativeError::InvalidParams)?;
        if requests.iter().any(|request| request.n_max > n_max) {
            return Err(MtpSpeculativeError::InvalidParams);
        }

        self.raw_requests.clear();
        self.raw_requests.extend(requests.iter().map(|request| {
            llama_cpp_sys_2::llama_rs_mtp_draft_request {
                seq_id: request.seq_id,
                pos0: request.pos0,
                id_last: request.id_last.0,
                n_max: request.n_max,
            }
        }));
        let capacity = requests.len().saturating_mul(self.n_max);
        self.raw_tokens.clear();
        self.raw_tokens.resize(capacity, 0);
        self.draft_counts.clear();
        self.draft_counts.resize(requests.len(), 0);

        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_draft(
                self.raw.as_ptr(),
                self.raw_requests.as_ptr(),
                self.raw_requests.len(),
                self.raw_tokens.as_mut_ptr(),
                self.n_max,
                self.draft_counts.as_mut_ptr(),
            )
        };
        status_to_result(status)?;

        self.draft_tokens.clear();
        self.draft_tokens
            .extend(self.raw_tokens.iter().copied().map(LlamaToken));
        Ok(MtpDrafts {
            tokens: &self.draft_tokens,
            counts: &self.draft_counts,
            stride: self.n_max,
        })
    }

    /// Reports how many tokens of the sequence's last draft the target accepted.
    ///
    /// The drafter continues from the hidden state of the last accepted token, so this is called
    /// after every verified draft, before the sequence is drafted for again.
    ///
    /// # Errors
    ///
    /// Returns [`MtpSpeculativeError::Status`] when `seq_id` is out of range or `n_accepted`
    /// exceeds the sequence's last draft.
    pub fn accept(&mut self, seq_id: i32, n_accepted: u16) -> Result<(), MtpSpeculativeError> {
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_accept(self.raw.as_ptr(), seq_id, n_accepted)
        };
        status_to_result(status)
    }
}

impl Drop for MtpSpeculative<'_> {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_free(self.raw.as_ptr());
        }
    }
}

fn status_to_result(status: llama_cpp_sys_2::llama_rs_status) -> Result<(), MtpSpeculativeError> {
    if status_is_ok(status) {
        Ok(())
    } else {
        Err(MtpSpeculativeError::Status(status as i32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_draft_context_is_created_with_its_target_as_ctx_other() {
        // Never dereferenced: only the address llama.cpp would receive is compared.
        let target = NonNull::<llama_cpp_sys_2::llama_context>::dangling();
        let params = LlamaContextParams::default();

        let raw_params = bound_draft_params(&params, target);

        assert_eq!(raw_params.ctx_other, target.as_ptr());
        // Everything else is the caller's parameters, unchanged.
        assert_eq!(raw_params.n_ctx, params.context_params.n_ctx);
        assert_eq!(raw_params.n_seq_max, params.context_params.n_seq_max);
    }
}
