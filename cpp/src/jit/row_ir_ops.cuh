/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

extern "C" __device__ uint32_t cudf_row_ir_add_i32(uint32_t lhs, uint32_t rhs);
extern "C" __device__ uint64_t cudf_row_ir_add_i64(uint64_t lhs, uint64_t rhs);
extern "C" __device__ uint32_t cudf_row_ir_mul_i32(uint32_t lhs, uint32_t rhs);
extern "C" __device__ uint64_t cudf_row_ir_mul_i64(uint64_t lhs, uint64_t rhs);
