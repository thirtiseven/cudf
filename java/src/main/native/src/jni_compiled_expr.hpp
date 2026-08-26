/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/scalar_column_view.hpp>
#include <cudf/scalar/scalar.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cudf {
namespace jni {
namespace ast {

struct expression_pair {
  std::reference_wrapper<cudf::ast::expression const> regular;
  std::reference_wrapper<cudf::ast::expression const> jit;

  expression_pair(cudf::ast::expression const& regular, cudf::ast::expression const& jit)
    : regular{regular}, jit{jit}
  {
  }
};

/** A class to capture all resources associated with a compiled AST expression. */
class compiled_expr {
  // Keep literal owners before both trees so their non-owning nodes are destroyed first.
  /** GPU scalar instances that correspond to literal nodes */
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  /** One-row columns backing literals in the JIT expression tree */
  std::vector<std::unique_ptr<cudf::column>> scalar_columns;

  /** All expression nodes within the regular expression tree */
  cudf::ast::tree expressions;

  /** All expression nodes within the JIT expression tree */
  cudf::ast::tree jit_expressions;

 public:
  template <typename ScalarType>
  expression_pair add_literal(ScalarType& scalar, std::unique_ptr<cudf::scalar> scalar_ptr)
  {
    auto scalar_column = cudf::make_column_from_scalar(scalar, 1);
    scalars.push_back(std::move(scalar_ptr));
    scalar_columns.push_back(std::move(scalar_column));
    return {expressions.emplace<cudf::ast::literal>(scalar),
            jit_expressions.emplace<cudf::ast::literal>(
              cudf::scalar_column_view{scalar_columns.back()->view()})};
  }

  expression_pair add_column_ref(cudf::size_type column_index, cudf::ast::table_reference table_ref)
  {
    return {expressions.emplace<cudf::ast::column_reference>(column_index, table_ref),
            jit_expressions.emplace<cudf::ast::column_reference>(column_index, table_ref)};
  }

  expression_pair add_column_name_ref(std::string column_name)
  {
    return {expressions.emplace<cudf::ast::column_name_reference>(column_name),
            jit_expressions.emplace<cudf::ast::column_name_reference>(std::move(column_name))};
  }

  expression_pair add_operation(cudf::ast::ast_operator op, expression_pair const& child)
  {
    return {expressions.emplace<cudf::ast::operation>(op, child.regular.get()),
            jit_expressions.emplace<cudf::ast::operation>(op, child.jit.get())};
  }

  expression_pair add_operation(cudf::ast::ast_operator op,
                                expression_pair const& left,
                                expression_pair const& right)
  {
    return {expressions.emplace<cudf::ast::operation>(op, left.regular.get(), right.regular.get()),
            jit_expressions.emplace<cudf::ast::operation>(op, left.jit.get(), right.jit.get())};
  }

  expression_pair add_jit_operation(cudf::ast::jit::op op,
                                    std::vector<expression_pair> const& args,
                                    cudf::error_policy error_policy,
                                    std::optional<int32_t> target_scale)
  {
    std::vector<std::reference_wrapper<cudf::ast::expression const>> regular_args;
    std::vector<std::reference_wrapper<cudf::ast::expression const>> jit_args;
    regular_args.reserve(args.size());
    jit_args.reserve(args.size());
    for (auto const& arg : args) {
      regular_args.emplace_back(arg.regular);
      jit_args.emplace_back(arg.jit);
    }
    return {cudf::ast::jit::operation(expressions, op, regular_args, error_policy, target_scale),
            cudf::ast::jit::operation(jit_expressions, op, jit_args, error_policy, target_scale)};
  }

  [[nodiscard]] bool has_literals() const { return !scalars.empty(); }

  /** Return the expression node at the top of the tree */
  cudf::ast::expression const& get_top_expression() const { return expressions.back(); }

  /** Return the expression node at the top of the JIT tree */
  cudf::ast::expression const& get_jit_top_expression() const { return jit_expressions.back(); }
};

}  // namespace ast
}  // namespace jni
}  // namespace cudf
