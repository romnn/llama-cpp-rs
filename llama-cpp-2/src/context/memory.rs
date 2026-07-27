//! Memory attribution for a loaded llama.cpp context.

use std::collections::TryReserveError;
use std::ffi::{CString, NulError};
use std::path::{Path, PathBuf};

use crate::context::params::LlamaContextParams;
use crate::context::LlamaContext;
use crate::model::params::LlamaModelParams;

/// Memory bytes grouped by the allocation's role.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LlamaMemoryUsage {
    /// Bytes allocated for model tensors.
    pub model_bytes: usize,
    /// Bytes allocated for persistent context state, including KV caches.
    pub context_bytes: usize,
    /// Bytes allocated for temporary compute buffers.
    pub compute_bytes: usize,
}

impl From<llama_cpp_sys_2::llama_rs_memory_usage> for LlamaMemoryUsage {
    fn from(value: llama_cpp_sys_2::llama_rs_memory_usage) -> Self {
        Self {
            model_bytes: value.model_bytes,
            context_bytes: value.context_bytes,
            compute_bytes: value.compute_bytes,
        }
    }
}

/// Memory attributed to one device in the model's llama.cpp device order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LlamaDeviceMemoryUsage {
    /// Zero-based index in the loaded model's device list.
    pub device_index: usize,
    /// Bytes allocated on the device.
    pub usage: LlamaMemoryUsage,
}

/// Point-in-time memory attributed to host buffers, model devices, and unknown buffer types.
///
/// Values sum buffers owned by llama.cpp when queried. They do not include backend or allocator
/// caches, driver reservations, or other memory owned by the process.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LlamaMemoryBreakdown {
    /// Memory backed by host buffer types.
    pub host: LlamaMemoryUsage,
    /// Memory backed by each device in the loaded model's device list.
    pub devices: Vec<LlamaDeviceMemoryUsage>,
    /// Memory whose buffer type cannot be associated with the host or a model device.
    pub unattributed: LlamaMemoryUsage,
}

/// Errors returned while collecting llama.cpp memory attribution.
#[derive(Debug, thiserror::Error)]
pub enum LlamaMemoryBreakdownError {
    /// The model path cannot be passed to llama.cpp as UTF-8.
    #[error("model path is not valid utf-8: {path}")]
    PathToStr {
        /// Path that could not be represented.
        path: PathBuf,
    },
    /// The model path contains an interior NUL byte.
    #[error("model path contains a nul byte")]
    PathContainsNul(#[source] NulError),
    /// The native wrapper rejected or failed the query.
    #[error("llama.cpp memory breakdown query failed with status {status}")]
    Native {
        /// Status code returned by the native wrapper.
        status: i32,
    },
    /// Rust could not reserve the device result buffer.
    #[error("failed to allocate memory breakdown device entries")]
    Allocation(#[source] TryReserveError),
}

impl LlamaContext<'_> {
    /// Returns a point-in-time sum of llama-owned model, context, and compute buffers by location.
    ///
    /// Buffer types that llama.cpp cannot associate with either host memory or a model device are
    /// preserved in [`LlamaMemoryBreakdown::unattributed`].
    ///
    /// # Errors
    ///
    /// Returns an error if the native query fails or the device result buffer cannot be allocated.
    pub fn memory_breakdown(&self) -> Result<LlamaMemoryBreakdown, LlamaMemoryBreakdownError> {
        let mut raw_host = raw_memory_usage();
        let mut raw_unattributed = raw_memory_usage();
        let mut device_count = 0_usize;

        // SAFETY: The context pointer remains valid for this shared borrow, and all output pointers
        // reference initialized writable values. A null device buffer with zero capacity is the
        // wrapper's documented size-query mode.
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_get_memory_breakdown(
                self.context.as_ptr(),
                &raw mut raw_host,
                &raw mut raw_unattributed,
                std::ptr::null_mut(),
                0,
                &raw mut device_count,
            )
        };
        status_to_result(status)?;

        let mut raw_devices = Vec::new();
        raw_devices
            .try_reserve_exact(device_count)
            .map_err(LlamaMemoryBreakdownError::Allocation)?;
        for device_index in 0..device_count {
            raw_devices.push(llama_cpp_sys_2::llama_rs_device_memory_usage {
                device_index,
                usage: raw_memory_usage(),
            });
        }

        // SAFETY: The context remains valid, and the device buffer has initialized capacity and
        // length for every entry reported by the size query. The wrapper validates the capacity
        // before writing.
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_get_memory_breakdown(
                self.context.as_ptr(),
                &raw mut raw_host,
                &raw mut raw_unattributed,
                raw_devices.as_mut_ptr(),
                raw_devices.len(),
                &raw mut device_count,
            )
        };
        status_to_result(status)?;

        let devices = raw_devices
            .into_iter()
            .take(device_count)
            .map(|device| LlamaDeviceMemoryUsage {
                device_index: device.device_index,
                usage: device.usage.into(),
            })
            .collect();

        Ok(LlamaMemoryBreakdown {
            host: raw_host.into(),
            devices,
            unattributed: raw_unattributed.into(),
        })
    }
}

/// Projects model, context, and compute-buffer memory without allocating their backing storage.
///
/// The model and context parameters must be the same values the eventual load will use. Device
/// entries follow the model's llama.cpp device order, just like [`LlamaContext::memory_breakdown`].
/// The projection can omit backend buffers created only while model weights are materialized, such
/// as CPU repacking buffers, and does not include driver or allocator caches. Prefer a measured
/// breakdown from the loaded context whenever one is available.
///
/// # Errors
///
/// Returns an error when the path cannot be passed to llama.cpp, the native projection fails, or
/// Rust cannot reserve its device result buffer.
pub fn estimate_memory_breakdown(
    model_path: &Path,
    model_params: &LlamaModelParams,
    context_params: &LlamaContextParams,
) -> Result<LlamaMemoryBreakdown, LlamaMemoryBreakdownError> {
    let model_path_string =
        model_path
            .to_str()
            .ok_or_else(|| LlamaMemoryBreakdownError::PathToStr {
                path: model_path.to_path_buf(),
            })?;
    let model_path =
        CString::new(model_path_string).map_err(LlamaMemoryBreakdownError::PathContainsNul)?;
    let device_capacity = crate::max_devices();
    let mut raw_host = raw_memory_usage();
    let mut raw_unattributed = raw_memory_usage();
    let mut raw_devices = raw_device_buffer(device_capacity)?;
    let mut device_count = 0_usize;

    // SAFETY: All parameter and output pointers remain valid for the call. The initialized device
    // buffer has one entry per device llama.cpp can represent, and the wrapper validates its
    // capacity before writing.
    let status = unsafe {
        llama_cpp_sys_2::llama_rs_estimate_memory_breakdown(
            model_path.as_ptr(),
            &raw const model_params.params,
            &raw const context_params.context_params,
            &raw mut raw_host,
            &raw mut raw_unattributed,
            raw_devices.as_mut_ptr(),
            raw_devices.len(),
            &raw mut device_count,
        )
    };
    status_to_result(status)?;

    Ok(memory_breakdown_from_raw(
        raw_host,
        raw_unattributed,
        raw_devices,
        device_count,
    ))
}

fn raw_memory_usage() -> llama_cpp_sys_2::llama_rs_memory_usage {
    llama_cpp_sys_2::llama_rs_memory_usage {
        model_bytes: 0,
        context_bytes: 0,
        compute_bytes: 0,
    }
}

fn raw_device_buffer(
    device_count: usize,
) -> Result<Vec<llama_cpp_sys_2::llama_rs_device_memory_usage>, LlamaMemoryBreakdownError> {
    let mut raw_devices = Vec::new();
    raw_devices
        .try_reserve_exact(device_count)
        .map_err(LlamaMemoryBreakdownError::Allocation)?;
    for device_index in 0..device_count {
        raw_devices.push(llama_cpp_sys_2::llama_rs_device_memory_usage {
            device_index,
            usage: raw_memory_usage(),
        });
    }
    Ok(raw_devices)
}

fn memory_breakdown_from_raw(
    raw_host: llama_cpp_sys_2::llama_rs_memory_usage,
    raw_unattributed: llama_cpp_sys_2::llama_rs_memory_usage,
    raw_devices: Vec<llama_cpp_sys_2::llama_rs_device_memory_usage>,
    device_count: usize,
) -> LlamaMemoryBreakdown {
    let devices = raw_devices
        .into_iter()
        .take(device_count)
        .map(|device| LlamaDeviceMemoryUsage {
            device_index: device.device_index,
            usage: device.usage.into(),
        })
        .collect();

    LlamaMemoryBreakdown {
        host: raw_host.into(),
        devices,
        unattributed: raw_unattributed.into(),
    }
}

fn status_to_result(
    status: llama_cpp_sys_2::llama_rs_status,
) -> Result<(), LlamaMemoryBreakdownError> {
    if status == llama_cpp_sys_2::LLAMA_RS_STATUS_OK {
        Ok(())
    } else {
        Err(LlamaMemoryBreakdownError::Native {
            status: status as i32,
        })
    }
}
