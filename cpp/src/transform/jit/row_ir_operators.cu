/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/row_ir/opcode.hpp>

#include <jit/row_ir_lto.cuh>

using cudf::detail::row_ir::evaluate;
using cudf::detail::row_ir::opcode;

// Unsigned physical types preserve Spark's non-ANSI, same-width integer wraparound.
extern "C" __device__ uint32_t cudf_row_ir_add_u32(uint32_t a, uint32_t b)
{
  return evaluate<opcode::ADD, cudf::error_policy::PROPAGATE>(a, b);
}

extern "C" __device__ uint32_t cudf_row_ir_mul_u32(uint32_t a, uint32_t b)
{
  return evaluate<opcode::MUL, cudf::error_policy::PROPAGATE>(a, b);
}

extern "C" __device__ uint64_t cudf_row_ir_add_u64(uint64_t a, uint64_t b)
{
  return evaluate<opcode::ADD, cudf::error_policy::PROPAGATE>(a, b);
}

extern "C" __device__ uint64_t cudf_row_ir_mul_u64(uint64_t a, uint64_t b)
{
  return evaluate<opcode::MUL, cudf::error_policy::PROPAGATE>(a, b);
}

extern "C" __device__ cuda::std::optional<uint32_t> cudf_row_ir_add_u32_nullable(
  cuda::std::optional<uint32_t> a, cuda::std::optional<uint32_t> b)
{
  return evaluate<opcode::ADD, cudf::error_policy::PROPAGATE>(a, b);
}

extern "C" __device__ cuda::std::optional<uint32_t> cudf_row_ir_mul_u32_nullable(
  cuda::std::optional<uint32_t> a, cuda::std::optional<uint32_t> b)
{
  return evaluate<opcode::MUL, cudf::error_policy::PROPAGATE>(a, b);
}

extern "C" __device__ cuda::std::optional<uint64_t> cudf_row_ir_add_u64_nullable(
  cuda::std::optional<uint64_t> a, cuda::std::optional<uint64_t> b)
{
  return evaluate<opcode::ADD, cudf::error_policy::PROPAGATE>(a, b);
}

extern "C" __device__ cuda::std::optional<uint64_t> cudf_row_ir_mul_u64_nullable(
  cuda::std::optional<uint64_t> a, cuda::std::optional<uint64_t> b)
{
  return evaluate<opcode::MUL, cudf::error_policy::PROPAGATE>(a, b);
}
