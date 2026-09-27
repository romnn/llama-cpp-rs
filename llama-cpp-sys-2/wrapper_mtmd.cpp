#include "wrapper_mtmd.h"

#include <exception>
#include <map>

// Defined by patches/0004-mtmd-project-the-compute-buffer-one-image-needs.patch. The pristine
// headers the shims compile against do not declare it.
std::map<ggml_backend_dev_t, size_t> mtmd_get_chunk_compute_usage(
    mtmd_context * ctx, const mtmd_input_chunk * chunk);

// Writes a per-device projection keyed by device handle as registry indices.
static llama_rs_mtmd_status write_device_memory(
    const std::map<ggml_backend_dev_t, size_t> & usage,
    struct llama_rs_mtmd_device_memory * out_devices,
    size_t device_capacity,
    size_t * out_device_count) {
    if (usage.size() > device_capacity) {
        return LLAMA_RS_MTMD_STATUS_CAPACITY_EXCEEDED;
    }

    const size_t registered_devices = ggml_backend_dev_count();
    size_t written = 0;
    for (const auto & [device, bytes] : usage) {
        size_t device_index = registered_devices;
        for (size_t index = 0; index < registered_devices; ++index) {
            if (ggml_backend_dev_get(index) == device) {
                device_index = index;
                break;
            }
        }
        if (device_index == registered_devices) {
            return LLAMA_RS_MTMD_STATUS_UNKNOWN_DEVICE;
        }
        out_devices[written] = llama_rs_mtmd_device_memory{device_index, bytes};
        ++written;
    }
    *out_device_count = written;
    return LLAMA_RS_MTMD_STATUS_OK;
}

extern "C" llama_rs_mtmd_status llama_rs_mtmd_estimate_memory_usage(
    const char * mmproj_path,
    const struct mtmd_context_params * params,
    struct llama_rs_mtmd_device_memory * out_devices,
    size_t device_capacity,
    size_t * out_device_count) {
    if (!mmproj_path || !params || !out_device_count || (!out_devices && device_capacity > 0)) {
        return LLAMA_RS_MTMD_STATUS_INVALID_ARGUMENT;
    }
    *out_device_count = 0;

    try {
        const std::map<ggml_backend_dev_t, size_t> usage = mtmd_get_memory_usage(mmproj_path, *params);
        // Upstream reports a failed projection as an empty map; a successful one always charges
        // the weights to at least one device.
        if (usage.empty()) {
            return LLAMA_RS_MTMD_STATUS_PROJECTION_FAILED;
        }
        return write_device_memory(usage, out_devices, device_capacity, out_device_count);
    } catch (const std::exception &) {
        return LLAMA_RS_MTMD_STATUS_PROJECTION_FAILED;
    } catch (...) {
        return LLAMA_RS_MTMD_STATUS_PROJECTION_FAILED;
    }
}

extern "C" llama_rs_mtmd_status llama_rs_mtmd_chunk_compute_usage(
    struct mtmd_context * ctx,
    const struct mtmd_input_chunk * chunk,
    struct llama_rs_mtmd_device_memory * out_devices,
    size_t device_capacity,
    size_t * out_device_count) {
    if (!ctx || !chunk || !out_device_count || (!out_devices && device_capacity > 0)) {
        return LLAMA_RS_MTMD_STATUS_INVALID_ARGUMENT;
    }
    *out_device_count = 0;

    try {
        const std::map<ggml_backend_dev_t, size_t> usage = mtmd_get_chunk_compute_usage(ctx, chunk);
        // A media chunk always needs compute buffers, so an empty projection of one failed; a
        // text chunk needs none.
        if (usage.empty() && mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
            return LLAMA_RS_MTMD_STATUS_PROJECTION_FAILED;
        }
        return write_device_memory(usage, out_devices, device_capacity, out_device_count);
    } catch (const std::exception &) {
        return LLAMA_RS_MTMD_STATUS_PROJECTION_FAILED;
    } catch (...) {
        return LLAMA_RS_MTMD_STATUS_PROJECTION_FAILED;
    }
}
