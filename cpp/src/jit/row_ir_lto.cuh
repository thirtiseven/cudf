/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstdint>
#include <cuda/std/optional>

using cuda::std::uint32_t;
using cuda::std::uint64_t;

// Keep this ABI header independent of the operator implementation and its template dependencies.
extern "C" {
__device__ uint32_t cudf_row_ir_add_u32(uint32_t, uint32_t);
__device__ uint32_t cudf_row_ir_mul_u32(uint32_t, uint32_t);
__device__ uint64_t cudf_row_ir_add_u64(uint64_t, uint64_t);
__device__ uint64_t cudf_row_ir_mul_u64(uint64_t, uint64_t);
__device__ cuda::std::optional<uint32_t> cudf_row_ir_add_u32_nullable(
  cuda::std::optional<uint32_t>, cuda::std::optional<uint32_t>);
__device__ cuda::std::optional<uint32_t> cudf_row_ir_mul_u32_nullable(
  cuda::std::optional<uint32_t>, cuda::std::optional<uint32_t>);
__device__ cuda::std::optional<uint64_t> cudf_row_ir_add_u64_nullable(
  cuda::std::optional<uint64_t>, cuda::std::optional<uint64_t>);
__device__ cuda::std::optional<uint64_t> cudf_row_ir_mul_u64_nullable(
  cuda::std::optional<uint64_t>, cuda::std::optional<uint64_t>);
__device__ uint32_t cudf_row_ir_load_u32(void const*, int, int);
__device__ void cudf_row_ir_store_u32(void const*, int, int, unsigned, uint32_t);
__device__ cuda::std::optional<uint32_t> cudf_row_ir_load_u32_nullable(void const*, int, int);
__device__ void cudf_row_ir_store_u32_nullable(
  void const*, int, int, unsigned, cuda::std::optional<uint32_t>);
__device__ uint64_t cudf_row_ir_load_u64(void const*, int, int);
__device__ void cudf_row_ir_store_u64(void const*, int, int, unsigned, uint64_t);
__device__ cuda::std::optional<uint64_t> cudf_row_ir_load_u64_nullable(void const*, int, int);
__device__ void cudf_row_ir_store_u64_nullable(
  void const*, int, int, unsigned, cuda::std::optional<uint64_t>);
}
