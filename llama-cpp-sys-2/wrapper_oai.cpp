#include "wrapper_oai.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "llama.cpp/common/chat.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

static void init_chat_msg(struct llama_rs_chat_msg_oaicompat * out_msg) {
    if (!out_msg) {
        return;
    }
    out_msg->role = nullptr;
    out_msg->content = nullptr;
    out_msg->content_parts = nullptr;
    out_msg->content_parts_count = 0;
    out_msg->reasoning_content = nullptr;
    out_msg->tool_name = nullptr;
    out_msg->tool_call_id = nullptr;
    out_msg->tool_calls = nullptr;
    out_msg->tool_calls_count = 0;
}

static llama_rs_status dup_content_parts(
    const std::vector<common_chat_msg_content_part> & parts,
    struct llama_rs_chat_msg_content_part_oaicompat ** out_items,
    size_t * out_count) {
    if (!out_items || !out_count) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_items = nullptr;
    *out_count = 0;
    if (parts.empty()) {
        return LLAMA_RS_STATUS_OK;
    }

    auto * items = static_cast<struct llama_rs_chat_msg_content_part_oaicompat *>(
        std::malloc(sizeof(struct llama_rs_chat_msg_content_part_oaicompat) * parts.size()));
    if (!items) {
        return LLAMA_RS_STATUS_ALLOCATION_FAILED;
    }
    for (size_t i = 0; i < parts.size(); ++i) {
        items[i].type = llama_rs_dup_string(parts[i].type);
        items[i].text = llama_rs_dup_string(parts[i].text);
        if ((!items[i].type && !parts[i].type.empty())
            || (!items[i].text && !parts[i].text.empty())) {
            for (size_t j = 0; j <= i; ++j) {
                std::free(items[j].type);
                std::free(items[j].text);
            }
            std::free(items);
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    *out_items = items;
    *out_count = parts.size();
    return LLAMA_RS_STATUS_OK;
}

static llama_rs_status fill_chat_msg(
    const common_chat_msg & msg,
    struct llama_rs_chat_msg_oaicompat * out_msg) {
    if (!out_msg) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    init_chat_msg(out_msg);

    if (!msg.role.empty()) {
        out_msg->role = llama_rs_dup_string(msg.role);
        if (!out_msg->role) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    if (!msg.content.empty()) {
        out_msg->content = llama_rs_dup_string(msg.content);
        if (!out_msg->content) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    if (!msg.content_parts.empty()) {
        const auto status = dup_content_parts(
            msg.content_parts,
            &out_msg->content_parts,
            &out_msg->content_parts_count);
        if (status != LLAMA_RS_STATUS_OK) {
            return status;
        }
    }
    if (!msg.reasoning_content.empty()) {
        out_msg->reasoning_content = llama_rs_dup_string(msg.reasoning_content);
        if (!out_msg->reasoning_content) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    if (!msg.tool_name.empty()) {
        out_msg->tool_name = llama_rs_dup_string(msg.tool_name);
        if (!out_msg->tool_name) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    if (!msg.tool_call_id.empty()) {
        out_msg->tool_call_id = llama_rs_dup_string(msg.tool_call_id);
        if (!out_msg->tool_call_id) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }

    if (!msg.tool_calls.empty()) {
        auto * calls = static_cast<struct llama_rs_tool_call_oaicompat *>(
            std::malloc(sizeof(struct llama_rs_tool_call_oaicompat) * msg.tool_calls.size()));
        if (!calls) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
            calls[i].name = llama_rs_dup_string(msg.tool_calls[i].name);
            calls[i].arguments = llama_rs_dup_string(msg.tool_calls[i].arguments);
            calls[i].id = llama_rs_dup_string(msg.tool_calls[i].id);
            if ((!calls[i].name && !msg.tool_calls[i].name.empty())
                || (!calls[i].arguments && !msg.tool_calls[i].arguments.empty())
                || (!calls[i].id && !msg.tool_calls[i].id.empty())) {
                for (size_t j = 0; j <= i; ++j) {
                    std::free(calls[j].name);
                    std::free(calls[j].arguments);
                    std::free(calls[j].id);
                }
                std::free(calls);
                return LLAMA_RS_STATUS_ALLOCATION_FAILED;
            }
        }
        out_msg->tool_calls = calls;
        out_msg->tool_calls_count = msg.tool_calls.size();
    }

    return LLAMA_RS_STATUS_OK;
}

extern "C" llama_rs_status llama_rs_apply_chat_template_with_tools_oaicompat(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_chat_message * messages,
    size_t message_count,
    const char * tools_json,
    const char * json_schema,
    const char * reasoning_format,
    bool enable_thinking,
    bool add_generation_prompt,
    struct llama_rs_chat_template_result * out_result) {
    if (!chat_template || !out_result) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    out_result->prompt = nullptr;
    out_result->grammar = nullptr;
    out_result->parser = nullptr;
    out_result->generation_prompt = nullptr;
    out_result->thinking_start_tag = nullptr;
    out_result->thinking_end_tag = nullptr;
    out_result->supports_thinking = false;
    out_result->chat_format = 0;
    out_result->grammar_lazy = false;
    out_result->grammar_triggers = nullptr;
    out_result->grammar_triggers_count = 0;
    out_result->preserved_tokens = nullptr;
    out_result->preserved_tokens_count = 0;
    out_result->additional_stops = nullptr;
    out_result->additional_stops_count = 0;

    try {
        auto tmpls = common_chat_templates_init(model, chat_template);
        common_chat_templates_inputs inputs;
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.use_jinja = true;
        inputs.enable_thinking = enable_thinking;
        if (reasoning_format && std::strlen(reasoning_format) > 0) {
            inputs.reasoning_format = common_reasoning_format_from_name(reasoning_format);
        }

        inputs.messages.reserve(message_count);
        for (size_t i = 0; i < message_count; ++i) {
            common_chat_msg msg;
            msg.role = messages[i].role ? messages[i].role : "";
            msg.content = messages[i].content ? messages[i].content : "";
            inputs.messages.push_back(std::move(msg));
        }

        if (tools_json && std::strlen(tools_json) > 0) {
            inputs.tools = common_chat_tools_parse_oaicompat(json::parse(tools_json));
        }
        if (json_schema && std::strlen(json_schema) > 0) {
            inputs.json_schema = json_schema;
        }

        auto params = common_chat_templates_apply(tmpls.get(), inputs);
        out_result->prompt = llama_rs_dup_string(params.prompt);
        if (!params.grammar.empty()) {
            out_result->grammar = llama_rs_dup_string(params.grammar);
        }
        if (!params.parser.empty()) {
            out_result->parser = llama_rs_dup_string(params.parser);
        }
        if (!params.generation_prompt.empty()) {
            out_result->generation_prompt = llama_rs_dup_string(params.generation_prompt);
        }
        if (!params.thinking_start_tag.empty()) {
            out_result->thinking_start_tag = llama_rs_dup_string(params.thinking_start_tag);
        }
        if (!params.thinking_end_tag.empty()) {
            out_result->thinking_end_tag = llama_rs_dup_string(params.thinking_end_tag);
        }
        out_result->supports_thinking = params.supports_thinking;
        out_result->chat_format = static_cast<int>(params.format);
        out_result->grammar_lazy = params.grammar_lazy;
        const auto status_triggers = dup_trigger_array(
            params.grammar_triggers,
            &out_result->grammar_triggers,
            &out_result->grammar_triggers_count);
        if (status_triggers != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_template_result_free(out_result);
            return status_triggers;
        }
        const auto status_tokens = dup_string_array(
            params.preserved_tokens,
            &out_result->preserved_tokens,
            &out_result->preserved_tokens_count);
        if (status_tokens != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_template_result_free(out_result);
            return status_tokens;
        }
        const auto status_stops = dup_string_array(
            params.additional_stops,
            &out_result->additional_stops,
            &out_result->additional_stops_count);
        if (status_stops != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_template_result_free(out_result);
            return status_stops;
        }
        if (!out_result->prompt
            || (!params.thinking_start_tag.empty() && !out_result->thinking_start_tag)
            || (!params.thinking_end_tag.empty() && !out_result->thinking_end_tag)) {
            llama_rs_chat_template_result_free(out_result);
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        llama_rs_chat_template_result_free(out_result);
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        llama_rs_chat_template_result_free(out_result);
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_apply_chat_template_oaicompat(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_rs_chat_template_oaicompat_params * params,
    struct llama_rs_chat_template_result * out_result) {
    if (!chat_template || !params || !out_result) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    if (!params->messages) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    out_result->prompt = nullptr;
    out_result->grammar = nullptr;
    out_result->parser = nullptr;
    out_result->generation_prompt = nullptr;
    out_result->thinking_start_tag = nullptr;
    out_result->thinking_end_tag = nullptr;
    out_result->supports_thinking = false;
    out_result->chat_format = 0;
    out_result->grammar_lazy = false;
    out_result->grammar_triggers = nullptr;
    out_result->grammar_triggers_count = 0;
    out_result->preserved_tokens = nullptr;
    out_result->preserved_tokens_count = 0;
    out_result->additional_stops = nullptr;
    out_result->additional_stops_count = 0;

    try {
        auto tmpls = common_chat_templates_init(model, chat_template);
        common_chat_templates_inputs inputs;
        inputs.add_generation_prompt = params->add_generation_prompt;
        inputs.use_jinja = params->use_jinja;
        inputs.parallel_tool_calls = params->parallel_tool_calls;
        inputs.enable_thinking = params->enable_thinking;
        inputs.add_bos = params->add_bos;
        inputs.add_eos = params->add_eos;

        inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(params->messages));
        if (params->tools && std::strlen(params->tools) > 0) {
            inputs.tools = common_chat_tools_parse_oaicompat(json::parse(params->tools));
        }
        if (params->tool_choice && std::strlen(params->tool_choice) > 0) {
            inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(params->tool_choice);
        }
        if (params->json_schema && std::strlen(params->json_schema) > 0) {
            inputs.json_schema = params->json_schema;
        }
        if (params->grammar && std::strlen(params->grammar) > 0) {
            inputs.grammar = params->grammar;
        }
        if (params->reasoning_format && std::strlen(params->reasoning_format) > 0) {
            inputs.reasoning_format = common_reasoning_format_from_name(params->reasoning_format);
        }
        if (params->chat_template_kwargs && std::strlen(params->chat_template_kwargs) > 0) {
            auto kwargs = json::parse(params->chat_template_kwargs);
            if (!kwargs.is_object()) {
                throw std::invalid_argument("chat_template_kwargs must be a JSON object");
            }
            for (const auto & item : kwargs.items()) {
                inputs.chat_template_kwargs[item.key()] = item.value().dump();
            }
        }

        auto params_out = common_chat_templates_apply(tmpls.get(), inputs);
        out_result->prompt = llama_rs_dup_string(params_out.prompt);
        if (!params_out.grammar.empty()) {
            out_result->grammar = llama_rs_dup_string(params_out.grammar);
        }
        if (!params_out.parser.empty()) {
            out_result->parser = llama_rs_dup_string(params_out.parser);
        }
        if (!params_out.generation_prompt.empty()) {
            out_result->generation_prompt = llama_rs_dup_string(params_out.generation_prompt);
        }
        if (!params_out.thinking_start_tag.empty()) {
            out_result->thinking_start_tag = llama_rs_dup_string(params_out.thinking_start_tag);
        }
        if (!params_out.thinking_end_tag.empty()) {
            out_result->thinking_end_tag = llama_rs_dup_string(params_out.thinking_end_tag);
        }
        out_result->supports_thinking = params_out.supports_thinking;
        out_result->chat_format = static_cast<int>(params_out.format);
        out_result->grammar_lazy = params_out.grammar_lazy;

        const auto status_triggers = dup_trigger_array(
            params_out.grammar_triggers,
            &out_result->grammar_triggers,
            &out_result->grammar_triggers_count);
        if (status_triggers != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_template_result_free(out_result);
            return status_triggers;
        }
        const auto status_tokens = dup_string_array(
            params_out.preserved_tokens,
            &out_result->preserved_tokens,
            &out_result->preserved_tokens_count);
        if (status_tokens != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_template_result_free(out_result);
            return status_tokens;
        }
        const auto status_stops = dup_string_array(
            params_out.additional_stops,
            &out_result->additional_stops,
            &out_result->additional_stops_count);
        if (status_stops != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_template_result_free(out_result);
            return status_stops;
        }
        if (!out_result->prompt
            || (!params_out.thinking_start_tag.empty() && !out_result->thinking_start_tag)
            || (!params_out.thinking_end_tag.empty() && !out_result->thinking_end_tag)) {
            llama_rs_chat_template_result_free(out_result);
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        llama_rs_chat_template_result_free(out_result);
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        llama_rs_chat_template_result_free(out_result);
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_chat_parse_to_oaicompat(
    const char * input,
    bool is_partial,
    int chat_format,
    bool parse_tool_calls,
    const char * parser_data,
    const char * generation_prompt,
    struct llama_rs_chat_msg_oaicompat * out_msg) {
    if (!input || !out_msg) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    init_chat_msg(out_msg);

    try {
        common_chat_parser_params syntax;
        syntax.format = static_cast<common_chat_format>(chat_format);
        syntax.parse_tool_calls = parse_tool_calls;
        if (generation_prompt && std::strlen(generation_prompt) > 0) {
            syntax.generation_prompt = generation_prompt;
        }
        if (parser_data && std::strlen(parser_data) > 0) {
            syntax.parser.load(parser_data);
        }

        // Tool-call ids are intentionally left as parsed (usually empty): id
        // assignment is caller policy, and generating ids here would require
        // duplicating the id helpers llama.cpp keeps in its server tool.
        const auto msg = common_chat_parse(input, is_partial, syntax);
        const auto status = fill_chat_msg(msg, out_msg);
        if (status != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_msg_free_oaicompat(out_msg);
            return status;
        }
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        llama_rs_chat_msg_free_oaicompat(out_msg);
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_chat_msg_free_oaicompat(struct llama_rs_chat_msg_oaicompat * msg) {
    if (!msg) {
        return;
    }
    if (msg->role) {
        std::free(msg->role);
    }
    if (msg->content) {
        std::free(msg->content);
    }
    if (msg->content_parts) {
        for (size_t i = 0; i < msg->content_parts_count; ++i) {
            std::free(msg->content_parts[i].type);
            std::free(msg->content_parts[i].text);
        }
        std::free(msg->content_parts);
    }
    if (msg->reasoning_content) {
        std::free(msg->reasoning_content);
    }
    if (msg->tool_name) {
        std::free(msg->tool_name);
    }
    if (msg->tool_call_id) {
        std::free(msg->tool_call_id);
    }
    if (msg->tool_calls) {
        for (size_t i = 0; i < msg->tool_calls_count; ++i) {
            std::free(msg->tool_calls[i].name);
            std::free(msg->tool_calls[i].arguments);
            std::free(msg->tool_calls[i].id);
        }
        std::free(msg->tool_calls);
    }
    msg->role = nullptr;
    msg->content = nullptr;
    msg->content_parts = nullptr;
    msg->content_parts_count = 0;
    msg->reasoning_content = nullptr;
    msg->tool_name = nullptr;
    msg->tool_call_id = nullptr;
    msg->tool_calls = nullptr;
    msg->tool_calls_count = 0;
}
