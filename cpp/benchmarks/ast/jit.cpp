/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace {

enum class jit_backend : uint8_t { SOURCE, LTO };

jit_backend backend_from_string(std::string_view backend)
{
  if (backend == "source") { return jit_backend::SOURCE; }
  if (backend == "lto") { return jit_backend::LTO; }
  CUDF_FAIL("Unrecognized JIT backend: " + std::string{backend});
}

std::unique_ptr<cudf::column> compute_column_source_jit(cudf::table_view const& table,
                                                        cudf::ast::expression const& expression,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  auto args = cudf::detail::row_ir::ast_converter::compute_column(
    cudf::detail::row_ir::target::CUDA, expression, table, {}, "compute_operation", stream, mr);

  CUDF_EXPECTS(args.lto_udf_source.has_value(), "Benchmark expression is not supported by LTO");

  auto result  = cudf::multi_transform(args.udf,
                                      args.source_type,
                                      args.is_null_aware,
                                      args.user_data,
                                      args.inputs,
                                      args.outputs,
                                      std::move(args.string_offsets),
                                      args.row_size,
                                      stream,
                                      mr);
  auto columns = result->release();
  return std::move(columns.front());
}

std::unique_ptr<cudf::column> compute_column_jit(jit_backend backend,
                                                 cudf::table_view const& table,
                                                 cudf::ast::expression const& expression,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  switch (backend) {
    case jit_backend::SOURCE: return compute_column_source_jit(table, expression, stream, mr);
    case jit_backend::LTO: return cudf::compute_column_jit(table, expression, stream, mr);
  }
  CUDF_FAIL("Invalid JIT backend");
}

void BM_ast_jit(nvbench::state& state, bool cold)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const tree_levels = static_cast<cudf::size_type>(state.get_int64("tree_levels"));
  auto const backend     = backend_from_string(state.get_string("backend"));

  CUDF_EXPECTS(tree_levels > 0, "Benchmark requires at least one operator");

  auto source_table =
    create_sequence_table({cudf::type_id::INT32, cudf::type_id::INT32}, row_count{num_rows});
  auto const table = source_table->view();

  cudf::ast::tree tree;
  auto const& lhs = tree.push(cudf::ast::column_reference{0});
  auto const& rhs = tree.push(cudf::ast::column_reference{1});
  tree.push(cudf::ast::operation{cudf::ast::ast_operator::ADD, lhs, rhs});

  for (cudf::size_type level = 1; level < tree_levels; ++level) {
    auto const op = level % 2 == 0 ? cudf::ast::ast_operator::ADD : cudf::ast::ast_operator::MUL;
    auto const& input = level % 2 == 0 ? lhs : rhs;
    tree.push(cudf::ast::operation{op, tree.back(), input});
  }

  auto const& expression = tree.back();
  auto const stream      = cudf::get_default_stream();
  auto const mr          = cudf::get_current_device_resource_ref();

  auto eligibility = cudf::detail::row_ir::ast_converter::compute_column(
    cudf::detail::row_ir::target::CUDA, expression, table, {}, "compute_operation", stream, mr);
  CUDF_EXPECTS(eligibility.lto_udf_source.has_value(),
               "Benchmark expression is not supported by LTO");

  if (!cold) {
    [[maybe_unused]] auto result = compute_column_jit(backend, table, expression, stream, mr);
  }

  state.add_global_memory_reads<int32_t>(static_cast<std::size_t>(num_rows) * 2);
  state.add_global_memory_writes<int32_t>(num_rows);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto const benchmark_stream = rmm::cuda_stream_view{launch.get_stream().get_stream()};
    [[maybe_unused]] auto result =
      compute_column_jit(backend, table, expression, benchmark_stream, mr);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void BM_ast_jit_cold(nvbench::state& state) { BM_ast_jit(state, true); }

void BM_ast_jit_hot(nvbench::state& state) { BM_ast_jit(state, false); }

}  // namespace

NVBENCH_BENCH(BM_ast_jit_cold)
  .set_name("ast_jit_cold")
  .set_run_once(true)
  .add_int64_axis("num_rows", {100'000})
  .add_int64_axis("tree_levels", {4})
  .add_string_axis("backend", {"source", "lto"});

NVBENCH_BENCH(BM_ast_jit_hot)
  .set_name("ast_jit_hot")
  .add_int64_axis("num_rows", {100'000'000})
  .add_int64_axis("tree_levels", {4})
  .add_string_axis("backend", {"source", "lto"});
