#pragma once

#include "wrapper_common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct llama_rs_chat_msg_content_part_oaicompat {
    char * type;
    char * text;
};

struct llama_rs_tool_call_oaicompat {
    char * name;
    char * arguments;
    char * id;
};

struct llama_rs_chat_msg_oaicompat {
    char * role;
    char * content;
    struct llama_rs_chat_msg_content_part_oaicompat * content_parts;
    size_t content_parts_count;
    char * reasoning_content;
    char * tool_name;
    char * tool_call_id;
    struct llama_rs_tool_call_oaicompat * tool_calls;
    size_t tool_calls_count;
};

struct llama_rs_chat_template_oaicompat_params {
    const char * messages;
    const char * tools;
    const char * tool_choice;
    const char * json_schema;
    const char * grammar;
    const char * reasoning_format;
    const char * chat_template_kwargs;
    bool add_generation_prompt;
    bool use_jinja;
    bool parallel_tool_calls;
    bool enable_thinking;
    bool add_bos;
    bool add_eos;
};

// Apply a chat template using the simple "tools + json_schema" inputs that
// upstream llama-server uses for the common case. The `reasoning_format` and
// `enable_thinking` parameters mirror llama-server's defaults; we expose them
// because Qwen3-style templates emit `<think>...</think>` markers in the
// generation_prompt regardless of `enable_thinking`, and the autoparser only
// builds a grammar that accepts those markers when `reasoning_format` is not
// `none`. Pass `reasoning_format = NULL` (or empty) to keep the upstream
// `COMMON_REASONING_FORMAT_NONE` default; pass `"auto"` / `"deepseek"` to
// match llama-server's default behaviour.
llama_rs_status llama_rs_apply_chat_template_with_tools_oaicompat(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_chat_message * messages,
    size_t message_count,
    const char * tools_json,
    const char * json_schema,
    const char * reasoning_format,
    bool enable_thinking,
    bool add_generation_prompt,
    struct llama_rs_chat_template_result * out_result);

llama_rs_status llama_rs_apply_chat_template_oaicompat(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_rs_chat_template_oaicompat_params * params,
    struct llama_rs_chat_template_result * out_result);

// Parse generated text into the structured message shape produced by
// `common_chat_parse`. Tool-call ids are returned exactly as parsed and may be
// empty; callers are expected to assign their own ids. On success the caller
// owns `out_msg` and must release it with `llama_rs_chat_msg_free_oaicompat`.
llama_rs_status llama_rs_chat_parse_to_oaicompat(
    const char * input,
    bool is_partial,
    int chat_format,
    bool parse_tool_calls,
    const char * parser_data,
    const char * generation_prompt,
    struct llama_rs_chat_msg_oaicompat * out_msg);

void llama_rs_chat_msg_free_oaicompat(struct llama_rs_chat_msg_oaicompat * msg);

// One change between two successive parses of a streamed response, as computed by
// `common_chat_msg_diff::compute_diffs`.
// No string is null.
// `tool_call_index` is `SIZE_MAX` when the diff concerns no tool call.
// A known call's `tool_call_name` and `tool_call_id` are empty unless they changed, while a newly
// found call carries its full current state.
struct llama_rs_chat_msg_diff_oaicompat {
    char * reasoning_content_delta;
    char * content_delta;
    size_t tool_call_index;
    char * tool_call_name;
    char * tool_call_arguments;
    char * tool_call_id;
};

// Parses the responses of one chat template: the serialized parser is loaded once, and the
// streamed response so far is kept to report what each update changed.
struct llama_rs_chat_parser;

// Creates a parser from the fields of a chat template result.
// On success the caller owns `out_parser` and must release it with `llama_rs_chat_parser_free`.
llama_rs_status llama_rs_chat_parser_init(
    int chat_format,
    bool parse_tool_calls,
    const char * parser_data,
    const char * generation_prompt,
    struct llama_rs_chat_parser ** out_parser);

void llama_rs_chat_parser_free(struct llama_rs_chat_parser * parser);

// Parses `input` on its own, independent of the streamed response.
// The result equals `llama_rs_chat_parse_to_oaicompat` with the parser's fields.
llama_rs_status llama_rs_chat_parser_parse(
    const struct llama_rs_chat_parser * parser,
    const char * input,
    bool is_partial,
    struct llama_rs_chat_msg_oaicompat * out_msg);

// Appends `text_added` to the streamed response, parses the whole response, and returns the diffs
// from the previous update, as llama-server's `task_result_state::update_chat_msg` does.
// A parse that finds nothing yet reports no diffs and keeps the previous message.
// On success the caller owns `out_diffs` and must release it with
// `llama_rs_chat_msg_diffs_free_oaicompat`.
llama_rs_status llama_rs_chat_parser_update(
    struct llama_rs_chat_parser * parser,
    const char * text_added,
    bool is_partial,
    struct llama_rs_chat_msg_diff_oaicompat ** out_diffs,
    size_t * out_diffs_count);

void llama_rs_chat_msg_diffs_free_oaicompat(
    struct llama_rs_chat_msg_diff_oaicompat * diffs,
    size_t count);

#ifdef __cplusplus
}
#endif
