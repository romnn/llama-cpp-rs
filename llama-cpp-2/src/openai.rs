//! OpenAI-compatible utility methods.
use crate::ChatParseError;
use std::ffi::CStr;
use std::os::raw::c_char;
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

fn owned_ffi_string(value: *const c_char) -> Result<Option<String>, ChatParseError> {
    if value.is_null() {
        return Ok(None);
    }
    // SAFETY: The wrapper returns owned, null-terminated strings for every
    // non-null field, and they stay alive until the message is freed after
    // conversion.
    let bytes = unsafe { CStr::from_ptr(value) }.to_bytes().to_vec();
    Ok(Some(String::from_utf8(bytes)?))
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
            role: owned_ffi_string(msg.role)?.unwrap_or_default(),
            content: owned_ffi_string(msg.content)?,
            content_parts: content_parts
                .iter()
                .map(|part| {
                    Ok(ChatMessageContentPartOaicompat {
                        part_type: owned_ffi_string(part.type_)?.unwrap_or_default(),
                        text: owned_ffi_string(part.text)?.unwrap_or_default(),
                    })
                })
                .collect::<Result<_, ChatParseError>>()?,
            reasoning_content: owned_ffi_string(msg.reasoning_content)?,
            tool_name: owned_ffi_string(msg.tool_name)?,
            tool_call_id: owned_ffi_string(msg.tool_call_id)?,
            tool_calls: tool_calls
                .iter()
                .map(|tool_call| {
                    Ok(ChatMessageToolCallOaicompat {
                        name: owned_ffi_string(tool_call.name)?.unwrap_or_default(),
                        arguments: owned_ffi_string(tool_call.arguments)?.unwrap_or_default(),
                        id: owned_ffi_string(tool_call.id)?.filter(|id| !id.is_empty()),
                    })
                })
                .collect::<Result<_, ChatParseError>>()?,
        })
    }
}
