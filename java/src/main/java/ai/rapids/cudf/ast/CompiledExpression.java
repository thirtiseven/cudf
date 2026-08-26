/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.MemoryCleaner;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

/** This class wraps a native compiled AST and must be closed to avoid native memory leaks. */
public class CompiledExpression implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger log = LoggerFactory.getLogger(CompiledExpression.class);

  private static class CompiledExpressionCleaner extends MemoryCleaner.Cleaner {
    private long nativeHandle;

    CompiledExpressionCleaner(long nativeHandle) {
      this.nativeHandle = nativeHandle;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      long origAddress = nativeHandle;
      boolean neededCleanup = nativeHandle != 0;
      if (neededCleanup) {
        try {
          destroy(nativeHandle);
        } finally {
          nativeHandle = 0;
        }
        if (logErrorIfNotClean) {
          log.error("AN AST COMPILED EXPRESSION WAS LEAKED (ID: " +
              id + " " + Long.toHexString(origAddress));
        }
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return nativeHandle == 0;
    }
  }

  private final CompiledExpressionCleaner cleaner;
  private boolean isClosed = false;

  /** Construct a compiled expression from a serialized AST */
  CompiledExpression(byte[] serializedExpression) {
    this(compile(serializedExpression));
  }

  /** Construct a compiled expression from a native compiled AST pointer */
  CompiledExpression(long nativeHandle) {
    this.cleaner = new CompiledExpressionCleaner(nativeHandle);
    MemoryCleaner.register(this, cleaner);
    cleaner.addRef();
  }

  /**
   * Compute a new column by applying this AST expression to the specified table. All
   * {@link ColumnReference} instances within the expression will use the sole input table,
   * even if they try to specify a non-existent table, e.g.: {@link TableReference#RIGHT}.
   * @param table input table for this expression
   * @return new column computed from this expression applied to the input table
   */
  public ColumnVector computeColumn(Table table) {
    long result;
    try {
      result = computeColumn(cleaner.nativeHandle, table.getNativeView());
    } finally {
      reachabilityFence(this);
      reachabilityFence(table);
    }
    return new ColumnVector(result);
  }

  /**
   * Compute a new column by applying this expression with the libcudf JIT executor, independent
   * of the process-level backend selected for {@link #computeColumn}.
   *
   * @param table input table for this expression
   * @return new column computed from this expression applied to the input table
   * @throws ai.rapids.cudf.CudfException if the expression refers to
   *         {@link TableReference#RIGHT}, or if JIT compilation or evaluation fails
   */
  public ColumnVector computeColumnJit(Table table) {
    long result;
    try {
      result = computeColumnJit(cleaner.nativeHandle, table.getNativeView());
    } finally {
      reachabilityFence(this);
      reachabilityFence(table);
    }
    return new ColumnVector(result);
  }

  /**
   * Compute a new table by applying expressions to the input table in one multi-output JIT
   * transform. Output column {@code i} contains the result of {@code expressions[i]}.
   *
   * @param table input table for the expressions
   * @param expressions non-empty expressions to evaluate in output order
   * @return table containing one output column per expression
   * @throws NullPointerException if the table, expression array, or an expression is null
   * @throws IllegalArgumentException if no expressions are provided
   * @throws IllegalStateException if the table or an expression is closed
   * @throws ai.rapids.cudf.CudfException if JIT compilation or evaluation fails
   */
  public static Table computeTableJit(Table table, CompiledExpression... expressions) {
    Objects.requireNonNull(table, "table");
    Objects.requireNonNull(expressions, "expressions");
    if (expressions.length == 0) {
      throw new IllegalArgumentException("At least one expression is required");
    }

    long tableHandle = table.getNativeView();
    if (tableHandle == 0) {
      throw new IllegalStateException("Table is closed");
    }

    CompiledExpression[] expressionRefs = expressions.clone();
    long[] nativeHandles = new long[expressionRefs.length];
    for (int i = 0; i < expressionRefs.length; i++) {
      CompiledExpression expression = Objects.requireNonNull(
          expressionRefs[i], "expression " + i + " is null");
      nativeHandles[i] = expression.cleaner.nativeHandle;
      if (nativeHandles[i] == 0) {
        throw new IllegalStateException("Expression " + i + " is closed");
      }
    }

    long[] result;
    try {
      result = computeTableJitNative(nativeHandles, tableHandle);
    } finally {
      reachabilityFence(table);
      reachabilityFence(expressionRefs);
    }
    return new Table(result);
  }

  private static void reachabilityFence(Object object) {
    if (object != null) {
      synchronized (object) {
        // The monitor operation is a Java 8 reachability fence.
      }
    }
  }

  @Override
  public synchronized void close() {
    cleaner.delRef();
    if (isClosed) {
      cleaner.logRefCountDebug("double free " + this);
      throw new IllegalStateException("Close called too many times " + this);
    }
    cleaner.clean(false);
    isClosed = true;
  }

  /** Returns the native address of a compiled expression. Intended for internal cudf use only. */
  public long getNativeHandle() {
    return cleaner.nativeHandle;
  }

  private static native long compile(byte[] serializedExpression);
  private static native long computeColumn(long astHandle, long tableHandle);
  private static native long computeColumnJit(long astHandle, long tableHandle);
  private static native long[] computeTableJitNative(long[] astHandles, long tableHandle);
  private static native void destroy(long handle);
}
