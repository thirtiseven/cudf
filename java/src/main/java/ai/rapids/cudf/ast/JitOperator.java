/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;

/**
 * Enumeration of AST JIT operators backed by libcudf row IR opcodes.
 * NOTE: This must be kept in sync with `jni_to_jit_operator` in CompiledExpression.cpp!
 */
public enum JitOperator {
  ANSI_ADD(0, 2),
  ANSI_SUB(1, 2),
  ANSI_MUL(2, 2),
  ANSI_ABS(3, 1),
  ANSI_NEG(4, 1),
  BIT_SHIFT_LEFT(5, 2),
  BIT_SHIFT_RIGHT(6, 2),
  COALESCE(7, 2),
  NULLIFY_IF(8, 2),
  PREDICATE(9, 1),
  ANSI_PRECISION_CHECK(10, 2),
  ANSI_TRY_PRECISION_CHECK(11, 2),
  CAST_TO_DEC32(12, 1),
  CAST_TO_DEC64(13, 1),
  CAST_TO_DEC128(14, 1),
  RESCALE(15, 1),
  ANSI_DIV(16, 2),
  ANSI_MOD(17, 2),
  CAST_TO_I64(18, 1),
  ANSI_TRY_ADD(19, 2),
  ANSI_TRY_SUB(20, 2),
  ANSI_TRY_MUL(21, 2),
  ANSI_TRY_DIV(22, 2),
  ANSI_TRY_MOD(23, 2),
  ANSI_TRY_ABS(24, 1),
  ANSI_TRY_NEG(25, 1),
  CAST_TO_B8(26, 1),
  CAST_TO_I8(27, 1),
  CAST_TO_I16(28, 1),
  CAST_TO_I32(29, 1),
  CAST_TO_U8(30, 1),
  CAST_TO_U16(31, 1),
  CAST_TO_U32(32, 1),
  CAST_TO_U64(33, 1),
  CAST_TO_F32(34, 1),
  CAST_TO_F64(35, 1),
  IF_ELSE(36, 3);

  private final byte nativeId;
  private final int arity;

  JitOperator(int nativeId, int arity) {
    this.nativeId = (byte) nativeId;
    this.arity = arity;
    assert this.nativeId == nativeId;
  }

  int getArity() {
    return arity;
  }

  /** Get the size in bytes to serialize this operator */
  int getSerializedSize() {
    return Byte.BYTES;
  }

  /** Serialize this operator to the specified buffer */
  void serialize(ByteBuffer bb) {
    bb.put(nativeId);
  }
}
