#pragma once

#include "llama.cpp/tools/mtmd/mtmd.h"
#include "llama.cpp/tools/mtmd/mtmd-helper.h"

#include <stddef.h>
#include <stdint.h>

// Memory a projector would hold on one ggml backend device.
struct llama_rs_mtmd_device_memory {
    // Index of the device in the ggml backend registry, as `ggml_backend_dev_get` takes it.
    size_t device_index;
    // Bytes the projection reports on this device.
    size_t bytes;
};

typedef enum llama_rs_mtmd_status {
    LLAMA_RS_MTMD_STATUS_OK = 0,
    LLAMA_RS_MTMD_STATUS_INVALID_ARGUMENT = -1,
    // The projection could not load the projector's metadata or plan its graph.
    LLAMA_RS_MTMD_STATUS_PROJECTION_FAILED = -2,
    // The projector reported more devices than the output buffer holds.
    LLAMA_RS_MTMD_STATUS_CAPACITY_EXCEEDED = -3,
    // The projector reported a device that is not in the ggml backend registry.
    LLAMA_RS_MTMD_STATUS_UNKNOWN_DEVICE = -4
} llama_rs_mtmd_status;

#ifdef __cplusplus
extern "C" {
#endif

// Projects the memory `mtmd_init_from_file` would allocate for the projector at `mmproj_path`
// with `params`, per ggml backend device, without loading its weights or allocating buffers.
// This wraps upstream's C++-only `mtmd_get_memory_usage`, which llama-server uses to reserve the
// projector's memory before fitting a model.
// Callers should provide `ggml_backend_dev_count()` entries.
llama_rs_mtmd_status llama_rs_mtmd_estimate_memory_usage(
    const char * mmproj_path,
    const struct mtmd_context_params * params,
    struct llama_rs_mtmd_device_memory * out_devices,
    size_t device_capacity,
    size_t * out_device_count);

// Projects the compute buffers encoding `chunk` with `ctx` needs, per ggml backend device, without
// allocating them.
// An encode grows a projector's compute buffers to its image's graph, beyond the warmup image
// `llama_rs_mtmd_estimate_memory_usage` covers.
// The projection measures on a scheduler of its own, so it leaves the context's buffers as they
// are, and it only covers those buffers, not the backend's scratch memory an encode also takes.
// A text chunk needs no compute buffers and reports no devices; a media chunk the projection fails
// for reports `LLAMA_RS_MTMD_STATUS_PROJECTION_FAILED`.
// Callers should provide `ggml_backend_dev_count()` entries.
llama_rs_mtmd_status llama_rs_mtmd_chunk_compute_usage(
    struct mtmd_context * ctx,
    const struct mtmd_input_chunk * chunk,
    struct llama_rs_mtmd_device_memory * out_devices,
    size_t device_capacity,
    size_t * out_device_count);

#ifdef __cplusplus
}
#endif
