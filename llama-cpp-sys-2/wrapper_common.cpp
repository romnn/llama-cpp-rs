#include "wrapper_common.h"

#include <algorithm>
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
#include "llama.cpp/common/json.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/common/reasoning-budget.h"
#include "llama.cpp/common/speculative.h"
#include "llama.cpp/include/llama.h"
#include "llama.cpp/src/llama-ext.h"
#include "wrapper_utils.h"

extern "C" llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar) {
    if (!schema_json || !out_grammar) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_grammar = nullptr;
    try {
        const auto schema = common_json::parse(schema_json);
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
    if (result->thinking_end_tags) {
        for (size_t i = 0; i < result->thinking_end_tags_count; ++i) {
            std::free(result->thinking_end_tags[i]);
        }
        std::free(result->thinking_end_tags);
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
    result->thinking_end_tags = nullptr;
    result->thinking_end_tags_count = 0;
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
    const size_t * end_sequence_lengths,
    size_t end_sequences_count,
    const llama_token * forced_tokens,
    size_t forced_tokens_count,
    int32_t budget) {
    if (!vocab
        || !start_tokens
        || start_tokens_count == 0
        || !end_tokens
        || !end_sequence_lengths
        || end_sequences_count == 0
        || !forced_tokens
        || forced_tokens_count == 0
        || budget < 0) {
        return nullptr;
    }

    try {
        const std::vector<llama_tokens> starts = {
            llama_tokens(start_tokens, start_tokens + start_tokens_count)};
        // End sequences arrive concatenated, delimited by their lengths.
        std::vector<llama_tokens> ends;
        ends.reserve(end_sequences_count);
        const llama_token * cursor = end_tokens;
        for (size_t i = 0; i < end_sequences_count; ++i) {
            if (end_sequence_lengths[i] == 0) {
                return nullptr;
            }
            ends.emplace_back(cursor, cursor + end_sequence_lengths[i]);
            cursor += end_sequence_lengths[i];
        }
        const llama_tokens forced(forced_tokens, forced_tokens + forced_tokens_count);
        return common_reasoning_budget_init(
            vocab,
            starts,
            ends,
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
    const void * extra,
    enum ggml_log_level log_level) {
    return static_cast<int>(common_fit_params(
        path_model,
        mparams,
        cparams,
        tensor_split,
        tensor_buft_overrides,
        margins,
        n_ctx_min,
        static_cast<const common_fit_extra_model *>(extra),
        log_level));
}

extern "C" void llama_rs_memory_breakdown_print(const struct llama_context * ctx) {
    common_memory_breakdown_print(ctx);
}

static llama_rs_status llama_rs_collect_memory_breakdown(
    const struct llama_model * model,
    const llama_memory_breakdown & memory_breakdown,
    struct llama_rs_memory_usage * out_host,
    struct llama_rs_memory_usage * out_unattributed,
    struct llama_rs_device_memory_usage * out_devices,
    size_t device_capacity,
    size_t * out_device_count) {
    if (!model || !out_host || !out_unattributed || !out_device_count ||
        (!out_devices && device_capacity != 0)) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_host = {};
    *out_unattributed = {};
    *out_device_count = 0;

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
}

extern "C" llama_rs_status llama_rs_get_memory_breakdown(
    const struct llama_context * ctx,
    struct llama_rs_memory_usage * out_host,
    struct llama_rs_memory_usage * out_unattributed,
    struct llama_rs_device_memory_usage * out_devices,
    size_t device_capacity,
    size_t * out_device_count) {
    if (!ctx) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        const auto * model = llama_get_model(ctx);
        const llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);
        return llama_rs_collect_memory_breakdown(
            model,
            memory_breakdown,
            out_host,
            out_unattributed,
            out_devices,
            device_capacity,
            out_device_count);
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_estimate_memory_breakdown(
    const char * model_path,
    const struct llama_model_params * model_params,
    const struct llama_context_params * context_params,
    struct llama_rs_memory_usage * out_host,
    struct llama_rs_memory_usage * out_unattributed,
    struct llama_rs_device_memory_usage * out_devices,
    size_t device_capacity,
    size_t * out_device_count) {
    if (!model_path || !model_params || !context_params) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        llama_model_params projected_model_params = *model_params;
        projected_model_params.no_alloc = true;
        projected_model_params.load_mode = LLAMA_LOAD_MODE_NONE;

        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path, projected_model_params),
            llama_model_free);
        if (!model) {
            return LLAMA_RS_STATUS_EXCEPTION;
        }

        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), *context_params),
            llama_free);
        if (!context) {
            return LLAMA_RS_STATUS_EXCEPTION;
        }

        const llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(context.get());
        return llama_rs_collect_memory_breakdown(
            model.get(),
            memory_breakdown,
            out_host,
            out_unattributed,
            out_devices,
            device_capacity,
            out_device_count);
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" bool llama_rs_model_file_has_mtp_layers(const char * path) {
    if (!path) {
        return false;
    }

    try {
        const auto types = common_speculative_types_from_gguf(path);
        return std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != types.end();
    } catch (...) {
        return false;
    }
}

extern "C" int32_t llama_rs_context_seq_rm_type(struct llama_context * ctx) {
    if (!ctx) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }

    try {
        return common_context_can_seq_rm(ctx);
    } catch (...) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }
}

struct llama_rs_mtp_speculative {
    common_params_speculative params;
    common_speculative * spec = nullptr;
    llama_context * ctx_dft = nullptr;
    uint32_t n_seq = 0;
    // Per-sequence draft output buffers that llama.cpp's draft parameters point into.
    std::vector<llama_tokens> drafts;
    // Per-sequence length of the draft awaiting `accept`, bounding what may be accepted.
    std::vector<size_t> pending_draft_len;
    // The MTP drafter reads no token history, but every draft request must point at one.
    llama_tokens no_history;
};

// Every token belongs to exactly one sequence below `n_seq`, and each sequence's tokens form one
// contiguous run of consecutive positions.
// The MTP drafter pairs each token with the target hidden state of the token before it by shifting
// rows within the batch, so any other layout would pair tokens with the wrong states; a token with
// several sequences aborts inside llama.cpp.
static bool llama_rs_mtp_batch_compatible(
    const struct llama_batch & batch,
    uint32_t n_seq,
    std::vector<int32_t> & last_index) {
    if (batch.n_tokens <= 0 || !batch.token || batch.embd || !batch.pos || !batch.n_seq_id ||
        !batch.seq_id) {
        return false;
    }
    std::fill(last_index.begin(), last_index.end(), -1);
    for (int32_t k = 0; k < batch.n_tokens; ++k) {
        if (batch.n_seq_id[k] != 1 || !batch.seq_id[k]) {
            return false;
        }
        const llama_seq_id seq_id = batch.seq_id[k][0];
        if (seq_id < 0 || (uint32_t) seq_id >= n_seq) {
            return false;
        }
        const int32_t previous = last_index[seq_id];
        if (previous >= 0 && (previous != k - 1 || batch.pos[k] != batch.pos[previous] + 1)) {
            return false;
        }
        last_index[seq_id] = k;
    }
    return true;
}

extern "C" struct llama_rs_mtp_speculative * llama_rs_mtp_speculative_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float p_min,
    uint32_t n_seq) {
    if (!ctx_tgt || !ctx_dft || n_max <= 0 || n_min < 0 || n_min > n_max || n_seq == 0 ||
        n_seq > llama_n_seq_max(ctx_tgt) || n_seq > llama_n_seq_max(ctx_dft)) {
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

        wrapper->spec = common_speculative_init(wrapper->params, n_seq);
        if (!wrapper->spec) {
            return nullptr;
        }
        wrapper->ctx_dft = ctx_dft;
        wrapper->n_seq = n_seq;
        wrapper->drafts.resize(n_seq);
        wrapper->pending_draft_len.assign(n_seq, 0);

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

extern "C" int32_t llama_rs_mtp_speculative_n_max(const struct llama_rs_mtp_speculative * spec) {
    if (!spec || !spec->spec) {
        return 0;
    }
    return common_speculative_n_max(spec->spec);
}

extern "C" llama_rs_status llama_rs_mtp_speculative_process(
    struct llama_rs_mtp_speculative * spec,
    const struct llama_batch * batch) {
    if (!spec || !spec->spec || !batch) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    // The drafter replays the batch through the draft context, whose batch holds `n_batch` tokens.
    if (batch->n_tokens > (int32_t) llama_n_batch(spec->ctx_dft)) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        std::vector<int32_t> last_index(spec->n_seq, -1);
        if (!llama_rs_mtp_batch_compatible(*batch, spec->n_seq, last_index)) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }
        return common_speculative_process(spec->spec, *batch)
            ? LLAMA_RS_STATUS_OK
            : LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_draft(
    struct llama_rs_mtp_speculative * spec,
    const struct llama_rs_mtp_draft_request * requests,
    size_t requests_count,
    llama_token * out_tokens,
    size_t out_tokens_stride,
    size_t * out_tokens_counts) {
    if (!spec || !spec->spec || (!requests && requests_count > 0) ||
        (requests_count > 0 && (!out_tokens || !out_tokens_counts)) ||
        requests_count > spec->n_seq) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        std::vector<bool> requested(spec->n_seq, false);
        for (size_t i = 0; i < requests_count; ++i) {
            const auto & request = requests[i];
            if (request.seq_id < 0 || (uint32_t) request.seq_id >= spec->n_seq ||
                requested[request.seq_id] || request.pos0 < 0 || request.n_max <= 0 ||
                (size_t) request.n_max > out_tokens_stride) {
                return LLAMA_RS_STATUS_INVALID_ARGUMENT;
            }
            requested[request.seq_id] = true;
        }

        for (size_t i = 0; i < requests_count; ++i) {
            const auto & request = requests[i];
            auto & draft = spec->drafts[request.seq_id];
            draft.clear();
            common_speculative_get_draft_params(spec->spec, request.seq_id) = {
                /* .drafting = */ true,
                /* .n_max    = */ request.n_max,
                /* .pos0     = */ request.pos0,
                /* .id_last  = */ request.id_last,
                /* .prompt   = */ &spec->no_history,
                /* .result   = */ &draft,
            };
        }

        if (requests_count > 0) {
            common_speculative_draft(spec->spec);
        }

        auto * memory_dft = llama_get_memory(spec->ctx_dft);
        for (size_t i = 0; i < requests_count; ++i) {
            const auto & request = requests[i];
            // Drafting decoded `id_last` and every draft token into the draft context.
            // Those positions hold what the verification batch is about to decode for real, so
            // they are dropped here: the draft context again mirrors exactly what the target has
            // decoded.
            llama_memory_seq_rm(memory_dft, request.seq_id, request.pos0, -1);

            const auto & draft = spec->drafts[request.seq_id];
            const size_t count = std::min(draft.size(), (size_t) request.n_max);
            if (count > 0) {
                std::memcpy(out_tokens + i * out_tokens_stride, draft.data(), count * sizeof(llama_token));
            }
            out_tokens_counts[i] = count;
            spec->pending_draft_len[request.seq_id] = count;
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_accept(
    struct llama_rs_mtp_speculative * spec,
    llama_seq_id seq_id,
    uint16_t n_accepted) {
    if (!spec || !spec->spec || seq_id < 0 || (uint32_t) seq_id >= spec->n_seq) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    if (n_accepted > spec->pending_draft_len[seq_id]) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        common_speculative_accept(spec->spec, seq_id, n_accepted);
        spec->pending_draft_len[seq_id] = 0;
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}
