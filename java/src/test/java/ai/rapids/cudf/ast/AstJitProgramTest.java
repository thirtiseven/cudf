/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.CudfTestBase;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class AstJitProgramTest extends CudfTestBase {
  private static AstJitProgram compile(Table input, int backend, CompiledExpression... expressions) {
    return AstJitProgram.compile(input, backend == 1, expressions);
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1})
  void testReusesMultiOutputProgramAndOwnsLiterals(int backend) {
    AstExpression shared = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), new ColumnReference(1));
    AstExpression multiply = new JitOperation(JitOperator.MUL, shared, Literal.ofInt(2));
    AstExpression sum = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), new ColumnReference(1));

    AstJitProgram program;
    try (Table schemaTable = new Table.TestBuilder()
             .column(1, 2, 3)
             .column(10, 20, 30)
             .column(100, 200, 300)
             .build();
         CompiledExpression multiplyCompiled = multiply.compileJit();
         CompiledExpression sumCompiled = sum.compileJit()) {
      program = compile(
          schemaTable, backend, multiplyCompiled, sumCompiled, sumCompiled);
    }

    try (AstJitProgram closeableProgram = program;
         Table firstInput = new Table.TestBuilder()
             .column(4, 5)
             .column(40, 50)
             .build();
         Table firstResult = closeableProgram.computeTable(firstInput);
         ColumnVector firstMultiply = ColumnVector.fromInts(88, 110);
         ColumnVector firstSum = ColumnVector.fromInts(44, 55);
         Table secondInput = new Table.TestBuilder()
             .column(6, 7, 8, 9)
             .column(60, 70, 80, 90)
             .column(600L, 700L, 800L, 900L)
             .build();
         Table secondResult = closeableProgram.computeTable(secondInput);
         ColumnVector secondMultiply = ColumnVector.fromInts(132, 154, 176, 198);
         ColumnVector secondSum = ColumnVector.fromInts(66, 77, 88, 99)) {
      Assertions.assertEquals(backend != 0, closeableProgram.usesLto());
      Assertions.assertEquals(3, firstResult.getNumberOfColumns());
      assertColumnsAreEqual(firstMultiply, firstResult.getColumn(0));
      assertColumnsAreEqual(firstSum, firstResult.getColumn(1));
      assertColumnsAreEqual(firstSum, firstResult.getColumn(2));
      Assertions.assertEquals(3, secondResult.getNumberOfColumns());
      assertColumnsAreEqual(secondMultiply, secondResult.getColumn(0));
      assertColumnsAreEqual(secondSum, secondResult.getColumn(1));
      assertColumnsAreEqual(secondSum, secondResult.getColumn(2));
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1})
  void testReusesSingleOutputProgram(int backend) {
    AstExpression expression = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), Literal.ofInt(1));
    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = expression.compileJit();
         AstJitProgram program = compile(schemaTable, backend, compiled);
         Table firstInput = new Table.TestBuilder().column(4, 5).build();
         Table firstResult = program.computeTable(firstInput);
         ColumnVector firstExpected = ColumnVector.fromInts(5, 6);
         Table secondInput = new Table.TestBuilder().column(6, 7, 8, 9).build();
         Table secondResult = program.computeTable(secondInput);
         ColumnVector secondExpected = ColumnVector.fromInts(7, 8, 9, 10)) {
      Assertions.assertEquals(backend != 0, program.usesLto());
      Assertions.assertEquals(1, firstResult.getNumberOfColumns());
      assertColumnsAreEqual(firstExpected, firstResult.getColumn(0));
      Assertions.assertEquals(1, secondResult.getNumberOfColumns());
      assertColumnsAreEqual(secondExpected, secondResult.getColumn(0));
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1})
  void testReusesProgramWithStringLiteral(int backend) {
    AstExpression expression = new BinaryOperation(BinaryOperator.LESS,
        new ColumnReference(0), Literal.ofString("ccc"));

    AstJitProgram program;
    try (Table schemaTable = new Table.TestBuilder().column("a", "ccc").build();
         CompiledExpression compiled = expression.compileJit()) {
      program = compile(schemaTable, backend, compiled);
    }

    try (AstJitProgram closeableProgram = program;
         Table input = new Table.TestBuilder().column("a", "ccc", "dddd").build();
         Table result = closeableProgram.computeTable(input);
         ColumnVector expected = ColumnVector.fromBooleans(true, false, false);
         Table secondInput = new Table.TestBuilder().column("zzz", "b").build();
         Table secondResult = closeableProgram.computeTable(secondInput);
         ColumnVector secondExpected = ColumnVector.fromBooleans(false, true)) {
      Assertions.assertFalse(closeableProgram.usesLto());
      assertColumnsAreEqual(expected, result.getColumn(0));
      assertColumnsAreEqual(secondExpected, secondResult.getColumn(0));
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1})
  void testRejectsIncompatibleReferencedColumnSchema(int backend) {
    AstExpression expression = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), Literal.ofInt(1));
    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = expression.compileJit();
         AstJitProgram program = compile(schemaTable, backend, compiled);
         Table wrongType = new Table.TestBuilder().column(1L, 2L, 3L).build();
         Table nullable = new Table.TestBuilder().column(1, null, 3).build()) {
      Assertions.assertThrows(CudfException.class, () -> program.computeTable(wrongType).close());
      Assertions.assertThrows(CudfException.class, () -> program.computeTable(nullable).close());
    }
  }

  @Test
  void testValidation() {
    AstExpression expression = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), Literal.ofInt(1));
    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = expression.compileJit()) {
      Assertions.assertThrows(NullPointerException.class,
          () -> AstJitProgram.compile(null, compiled));
      Assertions.assertThrows(NullPointerException.class,
          () -> AstJitProgram.compile(schemaTable, (CompiledExpression[]) null));
      Assertions.assertThrows(IllegalArgumentException.class,
          () -> AstJitProgram.compile(schemaTable));
      Assertions.assertThrows(NullPointerException.class,
          () -> AstJitProgram.compile(schemaTable, compiled, null));
    }

    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build()) {
      CompiledExpression closedExpression = expression.compileJit();
      closedExpression.close();
      Assertions.assertThrows(IllegalStateException.class,
          () -> AstJitProgram.compile(schemaTable, closedExpression));
    }

    AstExpression defaultExpression = new BinaryOperation(BinaryOperator.ADD,
        new ColumnReference(0), Literal.ofInt(1));
    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression nonJitExpression = defaultExpression.compile()) {
      Assertions.assertThrows(IllegalArgumentException.class,
          () -> AstJitProgram.compile(schemaTable, nonJitExpression));
    }

    try (Table closedTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = expression.compileJit()) {
      closedTable.close();  // idempotent
      Assertions.assertThrows(IllegalStateException.class,
          () -> AstJitProgram.compile(closedTable, compiled));
      Assertions.assertThrows(NullPointerException.class,
          () -> AstJitProgram.compile(closedTable, (CompiledExpression[]) null));
      Assertions.assertThrows(IllegalArgumentException.class,
          () -> AstJitProgram.compile(closedTable));
      Assertions.assertThrows(IllegalStateException.class,
          () -> AstJitProgram.compile(closedTable, compiled, null));
      try (CompiledExpression nonJitExpression = defaultExpression.compile()) {
        Assertions.assertThrows(IllegalStateException.class,
            () -> AstJitProgram.compile(closedTable, nonJitExpression));
      }
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1})
  void testLtoWraparoundAndIndependentOutputNulls(int variant) {
    boolean useLong = variant % 2 != 0;
    int backend = 1;
    AstExpression shared = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), new ColumnReference(1));
    AstExpression product = new JitOperation(JitOperator.MUL, shared, new ColumnReference(2));
    try (Table input = useLong
             ? new Table.TestBuilder()
                 .column(Long.MAX_VALUE, null, 3L, Long.MIN_VALUE, 4L)
                 .column(1L, 2L, null, -1L, 5L)
                 .column(2L, 3L, 4L, 2L, null).build()
             : new Table.TestBuilder()
                 .column(Integer.MAX_VALUE, null, 3, Integer.MIN_VALUE, 4)
                 .column(1, 2, null, -1, 5)
                 .column(2, 3, 4, 2, null).build();
         CompiledExpression sumCompiled = shared.compileJit();
         CompiledExpression productCompiled = product.compileJit();
         AstJitProgram program = compile(input, backend, sumCompiled, productCompiled);
         Table result = program.computeTable(input);
         ColumnVector sum = useLong
             ? ColumnVector.fromBoxedLongs(Long.MIN_VALUE, null, null, Long.MAX_VALUE, 9L)
             : ColumnVector.fromBoxedInts(Integer.MIN_VALUE, null, null, Integer.MAX_VALUE, 9);
         ColumnVector multiplied = useLong
             ? ColumnVector.fromBoxedLongs(0L, null, null, -2L, null)
             : ColumnVector.fromBoxedInts(0, null, null, -2, null)) {
      Assertions.assertTrue(program.usesLto());
      assertColumnsAreEqual(sum, result.getColumn(0));
      assertColumnsAreEqual(multiplied, result.getColumn(1));
    }
  }

  @Test
  void testLtoUnsupportedOperatorFallsBackAndPreservesErrors() {
    int backend = 1;
    AstExpression sub = new JitOperation(JitOperator.SUB,
        new ColumnReference(0), Literal.ofInt(1));
    AstExpression checked = new JitOperation(JitOperator.ADD_OVERFLOW,
        new ColumnReference(0), Literal.ofInt(1));
    try (Table input = new Table.TestBuilder().column(Integer.MAX_VALUE).build();
         CompiledExpression subCompiled = sub.compileJit();
         CompiledExpression checkedCompiled = checked.compileJit();
         AstJitProgram subProgram = compile(input, backend, subCompiled);
         AstJitProgram checkedProgram = compile(input, backend, checkedCompiled);
         Table result = subProgram.computeTable(input);
         ColumnVector expected = ColumnVector.fromInts(Integer.MAX_VALUE - 1)) {
      Assertions.assertFalse(subProgram.usesLto());
      Assertions.assertFalse(checkedProgram.usesLto());
      assertColumnsAreEqual(expected, result.getColumn(0));
      Assertions.assertThrows(CudfException.class, () -> checkedProgram.computeTable(input).close());
    }
  }

  @Test
  void testClosedInputs() {
    AstExpression expression = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), Literal.ofInt(1));
    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = expression.compileJit()) {
      AstJitProgram program = AstJitProgram.compile(schemaTable, compiled);
      Assertions.assertThrows(NullPointerException.class, () -> program.computeTable(null));
      program.close();
      Assertions.assertThrows(IllegalStateException.class, program::usesLto);
      Assertions.assertThrows(IllegalStateException.class, () -> program.computeTable(schemaTable));
      Assertions.assertThrows(IllegalStateException.class, program::close);
    }

    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = expression.compileJit();
         AstJitProgram program = AstJitProgram.compile(schemaTable, compiled);
         Table closedTable = new Table.TestBuilder().column(4, 5, 6).build()) {
      closedTable.close();
      Assertions.assertThrows(IllegalStateException.class,
          () -> program.computeTable(closedTable));
    }
  }
  @ParameterizedTest
  @ValueSource(ints = {1, 3, 4, 16, 32, 33, 64})
  void testFixedAbiMixedSignatures(int outputCount) {
    // Wrapper reuse must not depend on the number or types of outputs.
    for (int rows : new int[]{0, 1, 31, 32, 33, 65, 65537}) {
      for (boolean nullable : new boolean[]{false, true}) {
        Integer[] ints = new Integer[rows];
        Long[] longs = new Long[rows];
        for (int row = 0; row < rows; row++) {
          ints[row] = nullable && row % 7 == 0 ? null : Integer.MAX_VALUE - row;
          longs[row] = nullable && row % 11 == 0 ? null : Long.MAX_VALUE - row;
        }
        CompiledExpression[] expressions = new CompiledExpression[outputCount];
        try (ColumnVector intColumn = nullable ? ColumnVector.fromBoxedInts(ints)
                 : ColumnVector.fromInts(java.util.Arrays.stream(ints).mapToInt(Integer::intValue).toArray());
             ColumnVector longColumn = nullable ? ColumnVector.fromBoxedLongs(longs)
                 : ColumnVector.fromLongs(java.util.Arrays.stream(longs).mapToLong(Long::longValue).toArray());
             Table input = new Table(intColumn, longColumn)) {
          try {
            for (int i = 0; i < outputCount; i++) {
              AstExpression column = new ColumnReference(i % 2);
              AstExpression literal = i % 2 == 0 ? Literal.ofInt(i + 1) : Literal.ofLong(i + 1);
              expressions[i] = new JitOperation(JitOperator.MUL,
                  i % 3 == 0 ? literal : column, i % 3 == 0 ? column : literal).compileJit();
            }
            try (AstJitProgram program = AstJitProgram.compile(input, true, expressions);
                 Table result = program.computeTable(input)) {
              Assertions.assertTrue(program.usesLto());
              Assertions.assertEquals(outputCount, result.getNumberOfColumns());
              Assertions.assertEquals(rows, result.getRowCount());
              for (int i = 0; i < outputCount; i++) {
                if (i % 2 == 0) {
                  Integer[] values = new Integer[rows];
                  for (int row = 0; row < rows; row++) {
                    values[row] = ints[row] == null ? null : ints[row] * (i + 1);
                  }
                  try (ColumnVector expected = ColumnVector.fromBoxedInts(values)) {
                    assertColumnsAreEqual(expected, result.getColumn(i));
                  }
                } else {
                  Long[] values = new Long[rows];
                  for (int row = 0; row < rows; row++) {
                    values[row] = longs[row] == null ? null : longs[row] * (i + 1);
                  }
                  try (ColumnVector expected = ColumnVector.fromBoxedLongs(values)) {
                    assertColumnsAreEqual(expected, result.getColumn(i));
                  }
                }
              }
            }
          } finally {
            for (CompiledExpression expression : expressions) {
              if (expression != null) { expression.close(); }
            }
          }
        }
      }
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {1, 2})
  void testFixedAbiNullScalar(int outputCount) {
    AstExpression nullSum = new JitOperation(JitOperator.ADD,
        Literal.ofInt((Integer) null), new ColumnReference(0));
    AstExpression identity = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), Literal.ofInt(0));
    try (Table input = new Table.TestBuilder().column(3, null, 5).build();
         CompiledExpression first = nullSum.compileJit();
         CompiledExpression second = identity.compileJit();
         AstJitProgram program = outputCount == 1 ? AstJitProgram.compile(input, true, first)
             : AstJitProgram.compile(input, true, first, second);
         Table result = program.computeTable(input);
         ColumnVector expected = ColumnVector.fromBoxedInts(null, null, null)) {
      Assertions.assertTrue(program.usesLto());
      assertColumnsAreEqual(expected, result.getColumn(0));
      if (outputCount == 2) { assertColumnsAreEqual(input.getColumn(0), result.getColumn(1)); }
    }
  }

  @Test
  void testFixedAbiInterleavedPrograms() throws Exception {
    String cacheDir = System.getenv("LIBCUDF_KERNEL_CACHE_PATH");
    java.util.Set<String> firstPassCache = null;
    for (int pass = 0; pass < 3; pass++) {
      System.err.println("FIXED_ABI_CACHE_PASS_BEGIN " + pass);
      for (int step = 0; step < 16; step++) {
        int variant = pass == 1 ? 15 - step : step;
        int columns = 1 + variant / 4;
        boolean useLong = variant % 2 != 0;
        boolean nullable = variant % 4 >= 2;
        boolean multiply = variant % 3 == 0;
        int rows = 65;
        ColumnVector[] inputs = new ColumnVector[columns];
        Long[] expected = new Long[rows];
        AstExpression expression = null;
        try {
          for (int i = 0; i < columns; i++) {
            int ordinal = pass == 1 ? columns - i - 1 : i;
            Long[] values = new Long[rows];
            Integer[] intValues = new Integer[rows];
            for (int row = 0; row < rows; row++) {
              Long value = nullable && (row + ordinal) % 7 == 0 ? null
                  : (useLong ? Long.MAX_VALUE : Integer.MAX_VALUE) - row - ordinal * 3L;
              values[row] = value;
              intValues[row] = value == null ? null : value.intValue();
              if (value != null) {
                long weighted = value * (i + 1);
                value = useLong ? weighted : (long) (int) weighted;
              }
              if (i == 0) {
                expected[row] = value;
              } else if (expected[row] == null || value == null) {
                expected[row] = null;
              } else {
                long next = multiply ? expected[row] * value : expected[row] + value;
                expected[row] = useLong ? next : (long) (int) next;
              }
            }
            inputs[i] = useLong ? (nullable ? ColumnVector.fromBoxedLongs(values)
                : ColumnVector.fromLongs(java.util.Arrays.stream(values).mapToLong(Long::longValue).toArray()))
                : (nullable ? ColumnVector.fromBoxedInts(intValues)
                : ColumnVector.fromInts(java.util.Arrays.stream(intValues).mapToInt(Integer::intValue).toArray()));
            AstExpression factor = useLong ? Literal.ofLong(i + 1) : Literal.ofInt(i + 1);
            AstExpression ref = new JitOperation(JitOperator.MUL, new ColumnReference(i), factor);
            expression = expression == null ? ref : new JitOperation(
                multiply ? JitOperator.MUL : JitOperator.ADD, expression, ref);
          }
          AstExpression literal = useLong ? Literal.ofLong(variant + 2) : Literal.ofInt(variant + 2);
          expression = new JitOperation(multiply ? JitOperator.ADD : JitOperator.MUL,
              expression, literal);
          for (int row = 0; row < rows; row++) {
            if (expected[row] != null) {
              long next = multiply ? expected[row] + variant + 2 : expected[row] * (variant + 2);
              expected[row] = useLong ? next : (long) (int) next;
            }
          }
          try (Table input = new Table(inputs);
               CompiledExpression compiled = expression.compileJit();
               AstJitProgram program = AstJitProgram.compile(input, true, compiled);
               Table result = program.computeTable(input);
               ColumnVector expectedColumn = useLong ? ColumnVector.fromBoxedLongs(expected)
                   : ColumnVector.fromBoxedInts(java.util.Arrays.stream(expected)
                       .map(value -> value == null ? null : value.intValue()).toArray(Integer[]::new))) {
            Assertions.assertTrue(program.usesLto());
            assertColumnsAreEqual(expectedColumn, result.getColumn(0));
          }
        } finally {
          for (ColumnVector input : inputs) {
            if (input != null) { input.close(); }
          }
        }
      }
      if (cacheDir != null) {
        java.util.Set<String> cacheFiles;
        try (java.util.stream.Stream<java.nio.file.Path> files =
                 java.nio.file.Files.walk(java.nio.file.Paths.get(cacheDir, "rtcx_cache"))) {
          cacheFiles = files.filter(file -> file.toString().endsWith(".bin"))
              .map(Object::toString).collect(java.util.stream.Collectors.toSet());
        }
        Assertions.assertFalse(cacheFiles.isEmpty());
        if (pass == 0) {
          firstPassCache = cacheFiles;
        } else {
          Assertions.assertEquals(firstPassCache, cacheFiles,
              "Revisiting an expression must not add cached glue or kernels");
        }
        System.err.println("FIXED_ABI_CACHE_FILES " + pass + " " + cacheFiles.size());
      }
      System.err.println("FIXED_ABI_CACHE_PASS_END " + pass);
    }
  }

}
