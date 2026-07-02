/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/** Exception thrown when evaluating a CUDF expression reports a data error. */
public class CudfEvaluationException extends CudfException {
  private final int nativeErrorCode;
  private final ErrorCode errorCode;

  CudfEvaluationException(String message, int errorCode) {
    this(message, "No native stacktrace is available.", errorCode);
  }

  CudfEvaluationException(String message, String nativeStacktrace, int errorCode) {
    super(message, nativeStacktrace);
    this.nativeErrorCode = errorCode;
    this.errorCode = ErrorCode.fromNative(errorCode);
  }

  CudfEvaluationException(
      String message, String nativeStacktrace, int errorCode, Throwable cause) {
    super(message, nativeStacktrace, cause);
    this.nativeErrorCode = errorCode;
    this.errorCode = ErrorCode.fromNative(errorCode);
  }

  public ErrorCode getErrorCode() {
    return errorCode;
  }

  public int getNativeErrorCode() {
    return nativeErrorCode;
  }

  /** Java mirror of cudf::errc. */
  public enum ErrorCode {
    UNKNOWN(-1),
    SUCCESS(0),
    OVERFLOW(1),
    DIVISION_BY_ZERO(2);

    private final int nativeId;

    ErrorCode(int nativeId) {
      this.nativeId = nativeId;
    }

    public int getNativeId() {
      return nativeId;
    }

    static ErrorCode fromNative(int nativeId) {
      for (ErrorCode errorCode : values()) {
        if (errorCode.nativeId == nativeId) {
          return errorCode;
        }
      }
      return UNKNOWN;
    }
  }
}
