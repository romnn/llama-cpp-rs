//! OpenAI-compatible utility methods.
use crate::model::{optional_ffi_string, ChatTemplateResult};
use crate::{status_is_ok, ChatParseError};
use std::ffi::CString;
use std::fmt;
use std::mem;
use std::ptr::{self, NonNull};
use std::slice;

/// Parameters for applying OpenAI-compatible chat templates.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAIChatTemplateParams<'a> {
    /// OpenAI-compatible messages JSON array.
    pub messages_json: &'a str,
    /// Optional OpenAI-compatible tools JSON array.
    pub tools_json: Option<&'a str>,
    /// Optional tool choice string.
    pub tool_choice: Option<&'a str>,
    /// Optional JSON schema string for tool grammar generation.
    pub json_schema: Option<&'a str>,
    /// Optional custom grammar string.
    pub grammar: Option<&'a str>,
    /// Optional reasoning format string.
    pub reasoning_format: Option<&'a str>,
    /// Optional chat template kwargs JSON object.
    pub chat_template_kwargs: Option<&'a str>,
    /// Whether to add the assistant generation prompt.
    pub add_generation_prompt: bool,
    /// Whether to render templates with Jinja.
    pub use_jinja: bool,
    /// Whether to allow parallel tool calls.
    pub parallel_tool_calls: bool,
    /// Whether thinking blocks are enabled.
    pub enable_thinking: bool,
    /// Whether to add BOS.
    pub add_bos: bool,
    /// Whether to add EOS.
    pub add_eos: bool,
    /// Whether to parse tool calls in responses.
    pub parse_tool_calls: bool,
}

/// One typed content part of a parsed OpenAI-compatible message.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatMessageContentPartOaicompat {
    /// Content part type reported by the parser (e.g. `text`).
    pub part_type: String,
    /// Text payload of the content part.
    pub text: String,
}

/// One tool call extracted from a parsed OpenAI-compatible message.
///
/// `id` is only populated when the model's output format carries explicit
/// call ids; callers are expected to assign their own ids otherwise.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatMessageToolCallOaicompat {
    /// Name of the called tool.
    pub name: String,
    /// Tool call arguments as a JSON string.
    pub arguments: String,
    /// Call id as parsed from the model output, when present.
    pub id: Option<String>,
}

/// A model response parsed into OpenAI-compatible message fields.
///
/// Produced by [`crate::model::ChatTemplateResult::parse_response_oaicompat`].
/// `content` and `content_parts` mirror llama.cpp's `common_chat_msg`: plain
/// responses populate `content`, while typed responses populate
/// `content_parts` instead.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatMessageOaicompat {
    /// Message role (usually `assistant`).
    pub role: String,
    /// Plain-text message content.
    pub content: Option<String>,
    /// Typed content parts, populated instead of `content` for typed responses.
    pub content_parts: Vec<ChatMessageContentPartOaicompat>,
    /// Reasoning content extracted from thinking blocks.
    pub reasoning_content: Option<String>,
    /// Tool name for tool-role messages.
    pub tool_name: Option<String>,
    /// Tool call id for tool-role messages.
    pub tool_call_id: Option<String>,
    /// Tool calls extracted from the response.
    pub tool_calls: Vec<ChatMessageToolCallOaicompat>,
}

impl ChatMessageOaicompat {
    /// Converts the FFI message into an owned value without taking ownership
    /// of the FFI allocations; the caller still frees `msg`.
    pub(crate) unsafe fn from_ffi(
        msg: &llama_cpp_sys_2::llama_rs_chat_msg_oaicompat,
    ) -> Result<Self, ChatParseError> {
        let content_parts = if msg.content_parts_count == 0 || msg.content_parts.is_null() {
            &[]
        } else {
            // SAFETY: The wrapper guarantees `content_parts` points at
            // `content_parts_count` initialized entries.
            unsafe { slice::from_raw_parts(msg.content_parts, msg.content_parts_count) }
        };
        let tool_calls = if msg.tool_calls_count == 0 || msg.tool_calls.is_null() {
            &[]
        } else {
            // SAFETY: The wrapper guarantees `tool_calls` points at
            // `tool_calls_count` initialized entries.
            unsafe { slice::from_raw_parts(msg.tool_calls, msg.tool_calls_count) }
        };

        Ok(Self {
            role: optional_ffi_string(msg.role)?.unwrap_or_default(),
            content: optional_ffi_string(msg.content)?,
            content_parts: content_parts
                .iter()
                .map(|part| {
                    Ok(ChatMessageContentPartOaicompat {
                        part_type: optional_ffi_string(part.type_)?.unwrap_or_default(),
                        text: optional_ffi_string(part.text)?.unwrap_or_default(),
                    })
                })
                .collect::<Result<_, ChatParseError>>()?,
            reasoning_content: optional_ffi_string(msg.reasoning_content)?,
            tool_name: optional_ffi_string(msg.tool_name)?,
            tool_call_id: optional_ffi_string(msg.tool_call_id)?,
            tool_calls: tool_calls
                .iter()
                .map(|tool_call| {
                    Ok(ChatMessageToolCallOaicompat {
                        name: optional_ffi_string(tool_call.name)?.unwrap_or_default(),
                        arguments: optional_ffi_string(tool_call.arguments)?.unwrap_or_default(),
                        id: optional_ffi_string(tool_call.id)?.filter(|id| !id.is_empty()),
                    })
                })
                .collect::<Result<_, ChatParseError>>()?,
        })
    }
}

/// One change between two successive parses of a streamed response.
///
/// Produced by [`ChatResponseParser::update`], which takes it from llama.cpp's
/// `common_chat_msg_diff::compute_diffs`.
/// Each diff carries one kind of change: reasoning text, content text, or one tool call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatMessageDiffOaicompat {
    /// Reasoning text appended since the previous update.
    pub reasoning_content_delta: String,
    /// Content text appended since the previous update.
    pub content_delta: String,
    /// The tool call this diff changes, when it concerns one.
    pub tool_call: Option<ToolCallDiffOaicompat>,
}

/// The change to one tool call between two successive parses of a streamed response.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCallDiffOaicompat {
    /// Position of the call among the response's tool calls.
    pub index: usize,
    /// The call's name.
    ///
    /// A call reported by an earlier update carries its name only when the name or id changed and
    /// is empty otherwise, while a newly found call carries its current name.
    pub name: String,
    /// Argument text appended since the previous update; a newly found call carries all of it.
    pub arguments: String,
    /// The call's id as parsed from the model output, under the same rule as `name`.
    pub id: String,
}

impl ChatMessageDiffOaicompat {
    /// Converts FFI diffs into owned values without taking ownership of the FFI allocations; the
    /// caller still frees `diffs`.
    ///
    /// # Safety
    ///
    /// Unless `count` is zero, `diffs` must point at `count` entries that
    /// `llama_rs_chat_parser_update` filled and that stay valid until this returns.
    unsafe fn vec_from_ffi(
        diffs: *const llama_cpp_sys_2::llama_rs_chat_msg_diff_oaicompat,
        count: usize,
    ) -> Result<Vec<Self>, ChatParseError> {
        if count == 0 || diffs.is_null() {
            return Ok(Vec::new());
        }
        // SAFETY: The caller guarantees `diffs` points at `count` initialized entries.
        let diffs = unsafe { slice::from_raw_parts(diffs, count) };
        diffs
            .iter()
            .map(|diff| {
                let tool_call = if diff.tool_call_index == usize::MAX {
                    None
                } else {
                    Some(ToolCallDiffOaicompat {
                        index: diff.tool_call_index,
                        name: optional_ffi_string(diff.tool_call_name)?.unwrap_or_default(),
                        arguments: optional_ffi_string(diff.tool_call_arguments)?
                            .unwrap_or_default(),
                        id: optional_ffi_string(diff.tool_call_id)?.unwrap_or_default(),
                    })
                };
                Ok(Self {
                    reasoning_content_delta: optional_ffi_string(diff.reasoning_content_delta)?
                        .unwrap_or_default(),
                    content_delta: optional_ffi_string(diff.content_delta)?.unwrap_or_default(),
                    tool_call,
                })
            })
            .collect()
    }
}

/// Parses the responses generated for one [`ChatTemplateResult`].
///
/// Creating the parser deserializes the template's PEG parser once, so later parses skip that
/// cost.
/// The parser also follows one streamed response: [`Self::update`] appends generated text and
/// reports what changed, as llama-server does for each generated token.
pub struct ChatResponseParser {
    parser: NonNull<llama_cpp_sys_2::llama_rs_chat_parser>,
}

// SAFETY: The native parser owns plain C++ values and has no thread affinity, so it may move to
// another thread.
unsafe impl Send for ChatResponseParser {}

impl fmt::Debug for ChatResponseParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatResponseParser").finish_non_exhaustive()
    }
}

impl ChatTemplateResult {
    /// Creates a parser for the responses generated with this template result.
    ///
    /// # Errors
    ///
    /// Returns [`ChatParseError`] when the serialized parser or the generation prompt contains a
    /// null byte, or when the native parser cannot be loaded.
    pub fn response_parser(&self) -> Result<ChatResponseParser, ChatParseError> {
        let parser_cstr = self.parser.as_deref().map(CString::new).transpose()?;
        let generation_prompt_cstr = if self.generation_prompt.is_empty() {
            None
        } else {
            Some(CString::new(self.generation_prompt.as_str())?)
        };
        let mut parser = ptr::null_mut();
        // SAFETY: Every string pointer is null or a C string that outlives the call, and `parser`
        // is a valid out-pointer.
        let rc = unsafe {
            llama_cpp_sys_2::llama_rs_chat_parser_init(
                self.chat_format,
                self.parse_tool_calls,
                parser_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                generation_prompt_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                &raw mut parser,
            )
        };
        if !status_is_ok(rc) {
            return Err(ChatParseError::FfiError(rc));
        }
        let parser = NonNull::new(parser).ok_or(ChatParseError::FfiError(
            llama_cpp_sys_2::LLAMA_RS_STATUS_ALLOCATION_FAILED,
        ))?;
        Ok(ChatResponseParser { parser })
    }
}

impl ChatResponseParser {
    /// Parses `text` into OpenAI-compatible message fields, independent of the streamed response.
    ///
    /// The result equals [`ChatTemplateResult::parse_response_oaicompat`] on the template result
    /// this parser was created from.
    ///
    /// # Errors
    ///
    /// Returns [`ChatParseError`] when `text` contains a null byte, the native parser rejects it,
    /// or a parsed field is not valid UTF-8.
    pub fn parse(
        &self,
        text: &str,
        is_partial: bool,
    ) -> Result<ChatMessageOaicompat, ChatParseError> {
        let text_cstr = CString::new(text)?;
        // SAFETY: The message is a plain C struct of pointers and counts, for which all zeroes is
        // the empty value the wrapper expects.
        let mut out_msg: llama_cpp_sys_2::llama_rs_chat_msg_oaicompat = unsafe { mem::zeroed() };
        // SAFETY: `self.parser` is a live parser, and the input and out-pointer outlive the call.
        let rc = unsafe {
            llama_cpp_sys_2::llama_rs_chat_parser_parse(
                self.parser.as_ptr(),
                text_cstr.as_ptr(),
                is_partial,
                &raw mut out_msg,
            )
        };
        if !status_is_ok(rc) {
            return Err(ChatParseError::FfiError(rc));
        }
        // SAFETY: On success the wrapper filled `out_msg` with owned allocations that stay valid
        // until the free call below.
        let result = unsafe { ChatMessageOaicompat::from_ffi(&out_msg) };
        // SAFETY: The wrapper allocated `out_msg`, and nothing reads it after this free.
        unsafe { llama_cpp_sys_2::llama_rs_chat_msg_free_oaicompat(&raw mut out_msg) };
        result
    }

    /// Appends generated text to the streamed response and reports what changed.
    ///
    /// Mirrors llama-server's `task_result_state::update_chat_msg`: the whole response so far is
    /// parsed again, since llama.cpp's PEG parser cannot resume an earlier parse, and the new
    /// message is diffed against the previous one with `common_chat_msg_diff::compute_diffs`.
    /// A parse that finds nothing yet reports no diffs and keeps the previous message.
    /// Tool-call ids are reported as parsed, which is usually empty; assigning ids is the caller's
    /// policy.
    ///
    /// # Errors
    ///
    /// Returns [`ChatParseError`] when `text_added` contains a null byte, the native parser rejects
    /// the response, the new message does not extend the previous one (for example a tool call
    /// disappeared), or a diff is not valid UTF-8.
    /// `text_added` with a null byte is not appended.
    pub fn update(
        &mut self,
        text_added: &str,
        is_partial: bool,
    ) -> Result<Vec<ChatMessageDiffOaicompat>, ChatParseError> {
        let text_cstr = CString::new(text_added)?;
        let mut diffs = ptr::null_mut();
        let mut count = 0;
        // SAFETY: `self.parser` is a live parser that `&mut self` gives this call alone, and the
        // input and out-pointers outlive the call.
        let rc = unsafe {
            llama_cpp_sys_2::llama_rs_chat_parser_update(
                self.parser.as_ptr(),
                text_cstr.as_ptr(),
                is_partial,
                &raw mut diffs,
                &raw mut count,
            )
        };
        if !status_is_ok(rc) {
            return Err(ChatParseError::FfiError(rc));
        }
        // SAFETY: On success the wrapper filled `diffs` with `count` owned entries that stay valid
        // until the free call below.
        let result = unsafe { ChatMessageDiffOaicompat::vec_from_ffi(diffs, count) };
        // SAFETY: The wrapper allocated `diffs`, and nothing reads them after this free.
        unsafe { llama_cpp_sys_2::llama_rs_chat_msg_diffs_free_oaicompat(diffs, count) };
        result
    }
}

impl Drop for ChatResponseParser {
    fn drop(&mut self) {
        // SAFETY: The parser came from `llama_rs_chat_parser_init` and is freed exactly once.
        unsafe { llama_cpp_sys_2::llama_rs_chat_parser_free(self.parser.as_ptr()) }
    }
}
