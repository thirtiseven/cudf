/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/utilities/bit.hpp>

#include <cuda/atomic>

extern "C" __device__ int cudf_row_ir_fixed_transform(int row,
                                                      unsigned active,
                                                      void const* inputs,
                                                      void const* outputs);

extern "C" __global__ void cudf_kernel_entry(
  cudf::size_type row_size,
  cudf::bitmask_type const* stencil,
  void* user_data,
  cudf::column_device_view_core const* input_cols,
  cudf::mutable_column_device_view_core const* output_cols,
  int32_t* max_error)
{
  auto start       = cudf::detail::grid_1d::global_thread_id();
  auto stride      = cudf::detail::grid_1d::grid_stride();
  int thread_error = 0;
  for (auto row = start; row < row_size; row += stride) {
    auto active =
      __ballot_sync(__activemask(), stencil == nullptr || cudf::bit_is_set(stencil, row));
    if (stencil != nullptr && !cudf::bit_is_set(stencil, row)) { continue; }
    auto error   = cudf_row_ir_fixed_transform(row, active, input_cols, output_cols);
    thread_error = error > thread_error ? error : thread_error;
  }
  if (thread_error != 0) {
    cuda::atomic_ref ref(*max_error);
    ref.fetch_max(thread_error, cuda::std::memory_order_relaxed);
  }
}
