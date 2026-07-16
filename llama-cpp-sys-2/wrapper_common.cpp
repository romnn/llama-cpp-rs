#include "wrapper_common.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <stdint.h>
#include <vector>

#include "llama.cpp/common/chat.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/fit.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/common/reasoning-budget.h"
#include "llama.cpp/common/speculative.h"
#include "llama.cpp/include/llama.h"
#include "llama.cpp/src/llama-ext.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

extern "C" llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar) {
    if (!schema_json || !out_grammar) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_grammar = nullptr;
    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);
        return *out_grammar ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_chat_template_get_caps(
    const struct llama_model * model,
    const char * chat_template,
    struct llama_rs_chat_template_caps * out_caps) {
    if (!model || !chat_template || !out_caps) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_caps = {};
    try {
        auto tmpls = common_chat_templates_init(model, chat_template);
        const auto caps = common_chat_templates_get_caps(tmpls.get());

        const auto read_cap = [&](const char * key) {
            const auto it = caps.find(key);
            return it != caps.end() ? it->second : false;
        };

        out_caps->supports_tools = read_cap("supports_tools");
        out_caps->supports_tool_calls = read_cap("supports_tool_calls");
        out_caps->supports_system_role = read_cap("supports_system_role");
        out_caps->supports_parallel_tool_calls = read_cap("supports_parallel_tool_calls");
        out_caps->supports_preserve_reasoning = read_cap("supports_preserve_reasoning");
        out_caps->supports_thinking = common_chat_templates_support_enable_thinking(tmpls.get());
        out_caps->supports_string_content = read_cap("supports_string_content");
        out_caps->supports_typed_content = read_cap("supports_typed_content");
        out_caps->supports_object_arguments = read_cap("supports_object_arguments");
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_backend_load_all(void) {
    ggml_backend_load_all();
}

extern "C" llama_rs_status llama_rs_backend_load_all_from_path(const char * dir_path) {
    if (!dir_path) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        ggml_backend_load_all_from_path(dir_path);
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_chat_template_result_free(struct llama_rs_chat_template_result * result) {
    if (!result) {
        return;
    }
    if (result->prompt) {
        std::free(result->prompt);
    }
    if (result->grammar) {
        std::free(result->grammar);
    }
    if (result->parser) {
        std::free(result->parser);
    }
    if (result->generation_prompt) {
        std::free(result->generation_prompt);
    }
    if (result->thinking_start_tag) {
        std::free(result->thinking_start_tag);
    }
    if (result->thinking_end_tag) {
        std::free(result->thinking_end_tag);
    }
    if (result->grammar_triggers) {
        for (size_t i = 0; i < result->grammar_triggers_count; ++i) {
            std::free(result->grammar_triggers[i].value);
        }
        std::free(result->grammar_triggers);
    }
    if (result->preserved_tokens) {
        for (size_t i = 0; i < result->preserved_tokens_count; ++i) {
            std::free(result->preserved_tokens[i]);
        }
        std::free(result->preserved_tokens);
    }
    if (result->additional_stops) {
        for (size_t i = 0; i < result->additional_stops_count; ++i) {
            std::free(result->additional_stops[i]);
        }
        std::free(result->additional_stops);
    }
    result->prompt = nullptr;
    result->grammar = nullptr;
    result->parser = nullptr;
    result->generation_prompt = nullptr;
    result->thinking_start_tag = nullptr;
    result->thinking_end_tag = nullptr;
    result->supports_thinking = false;
    result->chat_format = 0;
    result->grammar_lazy = false;
    result->grammar_triggers = nullptr;
    result->grammar_triggers_count = 0;
    result->preserved_tokens = nullptr;
    result->preserved_tokens_count = 0;
    result->additional_stops = nullptr;
    result->additional_stops_count = 0;
}

extern "C" struct llama_sampler * llama_rs_sampler_init_reasoning_budget(
    const struct llama_vocab * vocab,
    const llama_token * start_tokens,
    size_t start_tokens_count,
    const llama_token * end_tokens,
    size_t end_tokens_count,
    const llama_token * forced_tokens,
    size_t forced_tokens_count,
    int32_t budget) {
    if (!vocab
        || !start_tokens
        || start_tokens_count == 0
        || !end_tokens
        || end_tokens_count == 0
        || !forced_tokens
        || forced_tokens_count == 0
        || budget < 0) {
        return nullptr;
    }

    try {
        const std::vector<llama_token> start(start_tokens, start_tokens + start_tokens_count);
        const std::vector<llama_token> end(end_tokens, end_tokens + end_tokens_count);
        const std::vector<llama_token> forced(forced_tokens, forced_tokens + forced_tokens_count);
        return common_reasoning_budget_init(
            vocab,
            start,
            end,
            forced,
            budget,
            REASONING_BUDGET_IDLE);
    } catch (...) {
        return nullptr;
    }
}

extern "C" void llama_rs_string_free(char * ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root) {
    try {
        return llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        std::vector<std::string> trigger_patterns;
        trigger_patterns.reserve(num_trigger_words);
        for (size_t i = 0; i < num_trigger_words; ++i) {
            const char * word = trigger_words ? trigger_words[i] : nullptr;
            if (word && word[0] != '\0') {
                trigger_patterns.push_back(regex_escape(word));
            }
        }
        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & pattern : trigger_patterns) {
            trigger_patterns_c.push_back(pattern.c_str());
        }
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns_c.data(),
            trigger_patterns_c.size(),
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns,
            num_trigger_patterns,
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token) {
    if (!sampler) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        llama_sampler_accept(sampler, token);
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

// Thin pass-through to llama.cpp's common_fit_params (a C++ symbol in libcommon).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
extern "C" int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level) {
    return static_cast<int>(common_fit_params(
        path_model,
        mparams,
        cparams,
        tensor_split,
        tensor_buft_overrides,
        margins,
        n_ctx_min,
        log_level));
}

extern "C" void llama_rs_memory_breakdown_print(const struct llama_context * ctx) {
    common_memory_breakdown_print(ctx);
}

extern "C" llama_rs_status llama_rs_get_memory_breakdown(
    const struct llama_context * ctx,
    struct llama_rs_memory_usage * out_host,
    struct llama_rs_memory_usage * out_unattributed,
    struct llama_rs_device_memory_usage * out_devices,
    size_t device_capacity,
    size_t * out_device_count) {
    if (!ctx || !out_host || !out_unattributed || !out_device_count ||
        (!out_devices && device_capacity != 0)) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_host = {};
    *out_unattributed = {};
    *out_device_count = 0;

    try {
        const auto * model = llama_get_model(ctx);
        if (!model) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }

        const int32_t signed_device_count = llama_model_n_devices(model);
        if (signed_device_count < 0) {
            return LLAMA_RS_STATUS_EXCEPTION;
        }
        const size_t device_count = static_cast<size_t>(signed_device_count);
        *out_device_count = device_count;

        if (!out_devices) {
            return LLAMA_RS_STATUS_OK;
        }
        if (device_capacity < device_count) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }

        for (size_t i = 0; i < device_count; ++i) {
            out_devices[i] = {};
            out_devices[i].device_index = i;
        }

        const auto add_usage = [](
            struct llama_rs_memory_usage & destination,
            const llama_memory_breakdown_data & source) {
            destination.model_bytes += source.model;
            destination.context_bytes += source.context;
            destination.compute_bytes += source.compute;
        };

        const llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);
        for (const auto & [buffer_type, usage] : memory_breakdown) {
            if (ggml_backend_buft_is_host(buffer_type)) {
                add_usage(*out_host, usage);
                continue;
            }

            const ggml_backend_dev_t device = ggml_backend_buft_get_device(buffer_type);
            bool matched_device = false;
            if (device) {
                for (size_t i = 0; i < device_count; ++i) {
                    if (device == llama_model_get_device(model, static_cast<int>(i))) {
                        add_usage(out_devices[i].usage, usage);
                        matched_device = true;
                        break;
                    }
                }
            }

            if (!matched_device) {
                add_usage(*out_unattributed, usage);
            }
        }

        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

struct llama_rs_mtp_speculative {
    common_params_speculative params;
    common_speculative * spec = nullptr;
    std::vector<llama_token> prompt;
    std::vector<llama_token> draft;
    size_t last_draft_len = 0;
    bool draft_pending = false;
};

static constexpr llama_seq_id LLAMA_RS_MTP_SEQ_ID = 0;

static bool llama_rs_mtp_batch_compatible(const struct llama_batch & batch) {
    if (batch.n_tokens <= 0 || !batch.token || batch.embd || !batch.pos || !batch.n_seq_id ||
        !batch.seq_id) {
        return false;
    }
    for (int32_t k = 0; k < batch.n_tokens; ++k) {
        if (batch.n_seq_id[k] != 1 || !batch.seq_id[k] ||
            batch.seq_id[k][0] != LLAMA_RS_MTP_SEQ_ID) {
            return false;
        }
    }
    return true;
}

static void llama_rs_assign_tokens(
    std::vector<llama_token> & dst,
    const llama_token * tokens,
    size_t count) {
    if (count == 0) {
        dst.clear();
        return;
    }
    dst.assign(tokens, tokens + count);
}

extern "C" struct llama_rs_mtp_speculative * llama_rs_mtp_speculative_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float p_min) {
    if (!ctx_tgt || !ctx_dft || n_max <= 0 || n_min < 0 || n_min > n_max) {
        return nullptr;
    }

    try {
        auto wrapper = std::make_unique<llama_rs_mtp_speculative>();
        wrapper->params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        wrapper->params.draft.ctx_tgt = ctx_tgt;
        wrapper->params.draft.ctx_dft = ctx_dft;
        wrapper->params.draft.n_max = n_max;
        wrapper->params.draft.n_min = n_min;
        wrapper->params.draft.p_min = p_min;

        wrapper->spec = common_speculative_init(wrapper->params, 1);
        if (!wrapper->spec) {
            return nullptr;
        }

        return wrapper.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void llama_rs_mtp_speculative_free(struct llama_rs_mtp_speculative * spec) {
    if (!spec) {
        return;
    }
    if (spec->spec) {
        common_speculative_free(spec->spec);
        spec->spec = nullptr;
    }
    delete spec;
}

extern "C" llama_rs_status llama_rs_mtp_speculative_begin(
    struct llama_rs_mtp_speculative * spec,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count) {
    if (!spec || !spec->spec || (!prompt_tokens && prompt_tokens_count > 0)) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        llama_rs_assign_tokens(spec->prompt, prompt_tokens, prompt_tokens_count);
        spec->last_draft_len = 0;
        spec->draft_pending = false;
        common_speculative_begin(spec->spec, LLAMA_RS_MTP_SEQ_ID, spec->prompt);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_process(
    struct llama_rs_mtp_speculative * spec,
    const struct llama_batch * batch) {
    if (!spec || !spec->spec || !batch) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    if (!llama_rs_mtp_batch_compatible(*batch)) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        return common_speculative_process(spec->spec, *batch)
            ? LLAMA_RS_STATUS_OK
            : LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_draft(
    struct llama_rs_mtp_speculative * spec,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count) {
    if (!spec || !spec->spec || (!prompt_tokens && prompt_tokens_count > 0) ||
        !out_tokens_count || n_past < 0) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        if (spec->draft_pending) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }
        llama_rs_assign_tokens(spec->prompt, prompt_tokens, prompt_tokens_count);
        spec->draft.clear();
        spec->last_draft_len = 0;

        auto & params = common_speculative_get_draft_params(spec->spec, LLAMA_RS_MTP_SEQ_ID);
        params = {
            true,
            spec->params.draft.n_max,
            n_past,
            id_last,
            &spec->prompt,
            &spec->draft,
        };

        common_speculative_draft(spec->spec);

        *out_tokens_count = spec->draft.size();
        if (spec->draft.size() > out_tokens_capacity) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        if (!spec->draft.empty() && !out_tokens) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }
        if (!spec->draft.empty()) {
            std::memcpy(out_tokens, spec->draft.data(), spec->draft.size() * sizeof(llama_token));
        }
        spec->last_draft_len = spec->draft.size();
        spec->draft_pending = !spec->draft.empty();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_accept(
    struct llama_rs_mtp_speculative * spec,
    uint16_t n_accepted) {
    if (!spec || !spec->spec) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    if (!spec->draft_pending || n_accepted > spec->last_draft_len) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        common_speculative_accept(spec->spec, LLAMA_RS_MTP_SEQ_ID, n_accepted);
        spec->last_draft_len = 0;
        spec->draft_pending = false;
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}
