/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir_arithmetic_ops.cuh"

extern "C" __device__ uint32_t cudf_row_ir_add_u32(uint32_t lhs, uint32_t rhs)
{
  return lhs + rhs;
}

extern "C" __device__ uint64_t cudf_row_ir_add_u64(uint64_t lhs, uint64_t rhs)
{
  return lhs + rhs;
}

extern "C" __device__ uint32_t cudf_row_ir_mul_u32(uint32_t lhs, uint32_t rhs)
{
  return lhs * rhs;
}

extern "C" __device__ uint64_t cudf_row_ir_mul_u64(uint64_t lhs, uint64_t rhs)
{
  return lhs * rhs;
}
