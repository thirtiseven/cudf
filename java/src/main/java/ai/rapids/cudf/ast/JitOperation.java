/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;
import java.util.Objects;

/** A JIT operation consisting of a row IR opcode and operands. */
public final class JitOperation extends AstExpression {
  private final JitOperator op;
  private final AstExpression[] inputs;

  public JitOperation(JitOperator op, AstExpression... inputs) {
    this.op = Objects.requireNonNull(op, "op is null");
    this.inputs = Objects.requireNonNull(inputs, "inputs is null").clone();
    if (this.inputs.length != op.getArity()) {
      throw new IllegalArgumentException(
          op + " requires " + op.getArity() + " inputs, found " + this.inputs.length);
    }
    for (AstExpression input : this.inputs) {
      Objects.requireNonNull(input, "input is null");
    }
  }

  @Override
  int getSerializedSize() {
    int size = ExpressionType.JIT_EXPRESSION.getSerializedSize() +
        op.getSerializedSize() +
        Byte.BYTES;
    for (AstExpression input : inputs) {
      size += input.getSerializedSize();
    }
    return size;
  }

  @Override
  void serialize(ByteBuffer bb) {
    ExpressionType.JIT_EXPRESSION.serialize(bb);
    op.serialize(bb);
    bb.put((byte) inputs.length);
    for (AstExpression input : inputs) {
      input.serialize(bb);
    }
  }

  @Override
  public String toString() {
    StringBuilder ret = new StringBuilder(op.toString()).append("(");
    for (int i = 0; i < inputs.length; i++) {
      if (i > 0) {
        ret.append(", ");
      }
      ret.append(inputs[i]);
    }
    return ret.append(")").toString();
  }
}
