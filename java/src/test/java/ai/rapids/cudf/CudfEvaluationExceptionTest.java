/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class CudfEvaluationExceptionTest {
  @Test
  public void testKnownErrorCode() {
    CudfEvaluationException exception =
        new CudfEvaluationException("overflow", "native stack", 1);

    Assertions.assertEquals(
        CudfEvaluationException.ErrorCode.OVERFLOW, exception.getErrorCode());
  }

  @Test
  public void testUnknownErrorCode() {
    CudfEvaluationException exception =
        new CudfEvaluationException("unknown", "native stack", 100);

    Assertions.assertEquals(
        CudfEvaluationException.ErrorCode.UNKNOWN, exception.getErrorCode());
    Assertions.assertEquals(100, exception.getNativeErrorCode());
  }
}
