/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <jit/column_accessor.cuh>
#include <jit/row_ir_lto.cuh>
#include <jit/sync.cuh>

namespace {
template <typename T>
using input_accessor = cudf::jit::column_accessor<0, cudf::column_device_view_core, T, false, 0>;
template <typename T>
using output_accessor =
  cudf::jit::column_accessor<0, cudf::mutable_column_device_view_core, T, false, 0>;
}  // namespace

extern "C" {
__device__ uint32_t cudf_row_ir_load_u32(void const* inputs, int index, int row)
{
  auto cols = static_cast<cudf::column_device_view_core const*>(inputs) + index;
  return input_accessor<uint32_t>::element(cols, row);
}
__device__ void cudf_row_ir_store_u32(
  void const* outputs, int index, int row, unsigned active, uint32_t value)
{
  auto cols = static_cast<cudf::mutable_column_device_view_core const*>(outputs) + index;
  output_accessor<uint32_t>::assign(cols, row, value);
}
__device__ cuda::std::optional<uint32_t> cudf_row_ir_load_u32_nullable(void const* inputs,
                                                                       int index,
                                                                       int row)
{
  auto cols = static_cast<cudf::column_device_view_core const*>(inputs) + index;
  return input_accessor<uint32_t>::nullable_element(cols, row);
}
__device__ void cudf_row_ir_store_u32_nullable(
  void const* outputs, int index, int row, unsigned active, cuda::std::optional<uint32_t> value)
{
  auto cols = static_cast<cudf::mutable_column_device_view_core const*>(outputs) + index;
  output_accessor<uint32_t>::assign(cols, row, value.value_or(0));
  cudf::jit::warp_compact_validity<output_accessor<uint32_t>>(active, cols, row, value.has_value());
}
__device__ uint64_t cudf_row_ir_load_u64(void const* inputs, int index, int row)
{
  auto cols = static_cast<cudf::column_device_view_core const*>(inputs) + index;
  return input_accessor<uint64_t>::element(cols, row);
}
__device__ void cudf_row_ir_store_u64(
  void const* outputs, int index, int row, unsigned active, uint64_t value)
{
  auto cols = static_cast<cudf::mutable_column_device_view_core const*>(outputs) + index;
  output_accessor<uint64_t>::assign(cols, row, value);
}
__device__ cuda::std::optional<uint64_t> cudf_row_ir_load_u64_nullable(void const* inputs,
                                                                       int index,
                                                                       int row)
{
  auto cols = static_cast<cudf::column_device_view_core const*>(inputs) + index;
  return input_accessor<uint64_t>::nullable_element(cols, row);
}
__device__ void cudf_row_ir_store_u64_nullable(
  void const* outputs, int index, int row, unsigned active, cuda::std::optional<uint64_t> value)
{
  auto cols = static_cast<cudf::mutable_column_device_view_core const*>(outputs) + index;
  output_accessor<uint64_t>::assign(cols, row, value.value_or(0));
  cudf::jit::warp_compact_validity<output_accessor<uint64_t>>(active, cols, row, value.has_value());
}
}
