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
  COALESCE(7, 2);

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
