# AST JIT LTO Design {#AST_JIT_LTO_DESIGN}

## Status

This document records the development design for reducing the cold-start cost of libcudf AST JIT
expressions with precompiled operator fragments and runtime LTO linking.

The current prototype proves the path for `ADD` and `MUL` with `INT32` and `INT64` operands. It
generates a thin expression topology, links that topology with an AOT-compiled operator fragment and
the existing transform kernel fragment, and falls back to CUDA source JIT for unsupported
expressions.

The target implementation described below is not complete yet. In particular, the current
prototype does not cover all Row IR operators, null-dependent operations, fallible operations, or
all transform kernel shapes.

## Goals

- Make every semantic signature accepted by the current Row IR operator registry linkable from
  precompiled operator fragments.
- Preserve the existing `compute_column_jit` API and its CUDA source fallback.
- Reuse `cudf::detail::row_ir::evaluate` as the only implementation of operator semantics.
- Keep the handwritten production and test diff small enough for one focused upstream review.
- Reduce cold compilation without changing hot execution semantics or performance.
- Keep operator coverage metadata, generated symbols, topology selection, and coverage tests in
  sync.

## Non-goals

- Multi-output execution or common subexpression elimination.
- Precompiling every possible transform kernel shape.
- Removing runtime compilation for arbitrary input arity, type combinations, or scalar placement.
- Adding LTO consumers to filter or join-filter execution in the same change.
- Changing Java, JNI, or cudf-spark APIs.
- Replacing the persistent JIT cache.

## Operator scope

The coverage target is the 76 semantic Row IR operators, not only the 50 legacy
`ast::ast_operator` values. The public `ast::jit::op` values map one-to-one to these 76 operators.
The internal Row IR enum contains two additional structural opcodes, `GET_INPUT` and `SET_OUTPUT`,
which do not need precompiled operator implementations.

| Family | Operators | Type signatures |
|---|---:|---:|
| Identity and null handling | 4 | 76 |
| Arithmetic | 10 | 124 |
| Checked arithmetic | 8 | 94 |
| Bitwise | 6 | 48 |
| Cast and rescale | 15 | 166 |
| Comparison | 7 | 175 |
| Logic and conditional | 6 | 295 |
| Mathematical | 8 | 24 |
| Trigonometric | 12 | 24 |
| Total | 76 | 1,026 |

The signature count comes from the 25 concrete type IDs and the argument and output rules in
`get_op_info`. Decimal scale is metadata in addition to the type ID and must be preserved by the
LTO ABI.

`PROPAGATE` and `NULLIFY` must not produce duplicate AOT operator specializations. A fallible
operator reports an error through the shared ABI. The thin topology either propagates that error or
converts it to an invalid result according to the expression's policy.

## Architecture

### Operator and type registries

Add a compact C++ `.inc` or `.def` registry containing, for each Row IR operator:

- operator and evaluator name;
- argument type schema and output type rule;
- arity;
- null behavior;
- fallibility;
- fragment family.

A separate short type registry maps each semantic type to:

- its stable type tag;
- its topology ABI type;
- its operator semantic type;
- its type categories.

The registries generate the LTO declarations, AOT definitions, signature lookup, and test
parameters. `get_op_info` should consume the same signature metadata or be exhaustively checked
against it. Public enums do not need to be refactored as part of this change.

The libcudf C++ build has no general Python or Jinja source generator for this purpose. The
registries should use C++ templates and X-macro expansion and continue to use the existing
`add_fragment` CMake support.

### Normalized operator ABI

Each legal semantic signature has one AOT specialization. Inputs and outputs use a normalized
value-and-validity representation conceptually equivalent to:

```cpp
template <typename T>
struct lto_value {
  T value;
  bool valid;
};
```

The exported device function writes an `lto_value<Out>` and returns an `errc`. Internally it
converts physical values to semantic values and invokes the existing
`cudf::detail::row_ir::evaluate` implementation.

This representation allows the same specialization to serve:

- nullable and non-nullable expression trees;
- null-propagating and null-dependent operators;
- successful and fallible evaluation;
- both `PROPAGATE` and `NULLIFY` policies.

The operator fragment implementation may use templates internally, but the generated external
device symbols should initially use stable C linkage. Direct linkage to C++ mangled template
symbols is smaller but requires an explicit NVCC-to-NVRTC compatibility test and should only be
adopted if it provides a measured benefit.

Physical type conversion must follow the existing transform reflection rules:

- booleans, integers, and floating-point values may use unsigned storage types of the same width;
- chrono values may use their representation type;
- decimal values must retain `numeric::decimal*` semantics and scale;
- strings must retain `string_view` semantics.

### Thin topology

The AST converter continues to instantiate and type-check the full Row IR tree on the host. For an
LTO-capable tree it emits only:

- input unpacking;
- calls to precompiled operator symbols;
- validity and error-policy control flow;
- output packing.

Eligibility becomes a generic recursive signature lookup. It must not contain an operator-specific
switch such as the current `ADD` and `MUL` prototype.

The generated topology remains a small runtime-compiled LTO fragment. Literal values remain runtime
transform inputs and are not embedded in cache keys.

### Fragment families

Do not put the entire operator matrix into every LTO link by default. Compile approximately five
operator fragment families:

- arithmetic and bitwise;
- checked arithmetic;
- casts, rescale, and decimal support;
- comparison, null handling, and control flow;
- mathematical and trigonometric operators.

String specializations may be split into a separate fragment if their size is material.

During AST conversion, collect the fragment families used by the expression. The transform LTO
path accepts a span of additional fragments and links only those families. The persistent kernel
cache key must continue to include all linked fragment contents.

The current RTCX key treats a named memory fragment as its name rather than its bytes. Operator
families must therefore either include their binary type and contents in the cache hash or carry an
explicit content/ABI version in the name. A changed AOT fragment must never reuse a linked cubin
from an older fragment.

### Transform kernel wrappers

Operator fragment coverage and transform kernel wrapper coverage are separate concerns.

The current precompiled transform wrapper set covers common non-null, single-output unary and
homogeneous binary shapes, including a right-hand scalar variant. The wrapper specialization also
contains input count, physical types, scalar placement, output types, and null-awareness.

The following shapes can therefore still require runtime wrapper compilation after all operators
are precompiled:

- heterogeneous casts and comparisons;
- three-input `IF_ELSE`;
- expressions with three or more unique leaves;
- uncommon scalar placement;
- null-aware execution;
- string and uncommon chrono combinations.

The number of possible wrapper shapes is unbounded with AST leaf count, so this design does not
enumerate them all. Common shapes should be added only from benchmark or workload evidence.
Unmatched shapes compile and cache a transform wrapper, then link it with the precompiled topology
and operator fragments.

If wrapper compilation remains the dominant cold cost, a later design can evaluate a type-erased
or partially dynamic transform wrapper ABI. That is a separate architectural change.

## Consumer scope

`compute_column_jit` is the first LTO consumer. It uses `lto_udf_source` when the complete
expression tree has registered operator signatures and otherwise uses the existing CUDA source
path.

The stream-compaction filter and join-filter paths currently generate Row IR CUDA source but do
not consume `lto_udf_source`. `PREDICATE` can be present in the operator registry, but this change
must not claim filter cold-start improvement until those consumers are wired separately.

## cudf-spark integration contract

The backend change is intentionally transparent to cudf-spark. The plugin continues to construct
the existing Java AST and compile it as a `CompiledExpression`. Row IR expressions may invoke
`CompiledExpression.computeColumnJit(Table)` explicitly; `computeColumn(Table)` also reaches the
same path when process-level JIT execution is enabled. Java serialization, JNI operator IDs, and
the native `compute_column_jit` signature remain unchanged. libcudf alone selects the
precompiled-fragment path or the CUDA source fallback.

Consequently, cudf-spark expression coverage can be developed and validated against the current
source JIT in parallel with this work. Picking up an LTO-enabled libcudf build should require only
the normal cudf/spark-rapids-jni dependency update; it must not require planner branches for
individual fragment signatures.

The Java AST can already express the complete semantic Row IR set through the union of its legacy
unary and binary nodes and `JitOperation`. The latter is intentionally only the additional subset
needed for checked operations, narrow casts, rescale, and related Row IR features. LTO coverage
must not duplicate all C++ operators in `JitOperator` or alter the existing wire format.

The integration still has the following release and validation requirements:

- cudf-spark, spark-rapids-jni, and libcudf artifacts must be upgraded as a compatible set when
  Java-visible AST operators are added. This LTO change itself must not renumber or reinterpret
  existing operators. Development builds using identical snapshot version strings must additionally
  verify their source revisions.
- Falling back because a complete expression is not LTO-capable changes cold latency, not results.
  Tests and benchmarks therefore need an internal strict or observable mode that distinguishes an
  LTO hit from a source fallback. This does not need to be a public Spark-facing API.
- The existing backend environment selection is initialized once per process. Comparisons between
  the legacy evaluator and JIT execution therefore require separate processes; a per-query Spark
  configuration must not be treated as an equivalent backend selector.
- An explicit `computeColumnJit` call is independent of the process-level selector. Until there is
  an LTO-only policy, the cudf-spark Project AST JIT feature gate remains the rollout kill switch.
- A failure after LTO compilation or linking has started is an execution error, not an eligibility
  fallback. Production rollout should retain the existing Spark feature gate and first validate
  every registered signature with strict LTO linkage tests.
- Spark error translation, retry behavior, nullability, decimal semantics, and expression
  eligibility remain plugin responsibilities. They can be implemented independently because both
  source and LTO paths execute the same typed Row IR.
- This contract applies to Project expressions evaluated by `compute_column_jit`. Filter and
  join-filter consumers require separate libcudf wiring before they receive the same cold-start
  benefit.

## Testing

Existing semantic tests should remain the primary correctness oracle:

- `TransformTest` already runs common legacy AST cases through both the interpreter and JIT
  executors.
- `jit_expressions_tests.cpp` already covers checked arithmetic with `PROPAGATE` and `NULLIFY`,
  casts, decimal rescale, shifts, coalesce, and fused error trees.

New tests should focus on LTO selection and linkage:

1. Generate a parameter matrix from the production registry and verify every legal signature has a
   symbol and fragment family.
2. Verify representative generated topologies for homogeneous, heterogeneous, nullable, fallible,
   decimal, chrono, string, and conditional ABI families.
3. Execute representative expressions from each ABI family through `compute_column_jit`. Missing
   symbols or mismatched signatures must fail the test rather than silently retry source JIT.
4. Add small table-driven semantic cases only for operators not already exercised end-to-end.
5. Fix the existing null string-literal comparison assertion before enabling string LTO coverage.

The generic `transform_lto` tests validate the linking engine but do not validate production Row IR
symbol generation, so registry-driven production tests are still required.

## Benchmarking

Cold and hot behavior are different acceptance criteria.

The source and LTO backends must evaluate the exact same AST in separate processes with separate
fresh `LIBCUDF_KERNEL_CACHE_PATH` directories. Cold results use CPU wall time reported by NVBench,
which includes runtime compilation and linking. Hot results use GPU execution time after explicit
warmup.

Track at least:

- thin-topology compilation time;
- transform wrapper compile hit or miss;
- LTO link time;
- operator fragment size by family;
- end-to-end cold latency;
- hot kernel time.

Adding all operator specializations is acceptable only if family-based linking retains the cold
latency benefit and hot execution remains equivalent to source JIT.

## Implementation checkpoints

1. Preserve the current four-symbol prototype and exact-backend benchmark as a local checkpoint.
2. Introduce the shared registries and normalized ABI without expanding public APIs.
3. Generate all non-null fixed-width specializations and validate fragment size and link time.
4. Add fallible, nullable, decimal, chrono, string, and control-flow support.
5. Replace operator-specific eligibility with registry lookup and add generated coverage tests.
6. Make named operator fragment identity content-safe in the persistent linked-kernel cache.
7. Measure common transform wrapper misses and add only evidence-backed wrapper fragments.
8. Replay the final net change onto a clean upstream base before requesting review.

## Review boundaries

The intended upstream change is one focused operator-fragment PR of approximately 1,500 to 2,000
lines, including tests and the exact-backend benchmark. If the normalized ABI materially exceeds
that range, if fragment-family linking loses the measured cold-start benefit, or if stable device
linkage requires broader runtime changes, discuss the split with the Row IR maintainers before
publishing.
