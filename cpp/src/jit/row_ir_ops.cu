/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir_ops.cuh"

#include <cudf/detail/row_ir/opcode.hpp>

#include <cuda/std/bit>

namespace {

template <cudf::detail::row_ir::opcode Op, typename Semantic, typename Physical>
__device__ Physical evaluate_binary(Physical lhs, Physical rhs)
{
  auto const semantic_lhs = cuda::std::bit_cast<Semantic>(lhs);
  auto const semantic_rhs = cuda::std::bit_cast<Semantic>(rhs);
  auto const result =
    cudf::detail::row_ir::evaluate<Op, cudf::error_policy::PROPAGATE>(semantic_lhs, semantic_rhs);
  return cuda::std::bit_cast<Physical>(result);
}

}  // namespace

extern "C" __device__ uint32_t cudf_row_ir_add_i32(uint32_t lhs, uint32_t rhs)
{
  return evaluate_binary<cudf::detail::row_ir::opcode::ADD, int32_t>(lhs, rhs);
}

extern "C" __device__ uint64_t cudf_row_ir_add_i64(uint64_t lhs, uint64_t rhs)
{
  return evaluate_binary<cudf::detail::row_ir::opcode::ADD, int64_t>(lhs, rhs);
}

extern "C" __device__ uint32_t cudf_row_ir_mul_i32(uint32_t lhs, uint32_t rhs)
{
  return evaluate_binary<cudf::detail::row_ir::opcode::MUL, int32_t>(lhs, rhs);
}

extern "C" __device__ uint64_t cudf_row_ir_mul_i64(uint64_t lhs, uint64_t rhs)
{
  return evaluate_binary<cudf::detail::row_ir::opcode::MUL, int64_t>(lhs, rhs);
}
