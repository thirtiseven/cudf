/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir.hpp"

#include "cudf_test/column_wrapper.hpp"

#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/transform.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <cctype>

namespace row_ir = cudf::detail::row_ir;

struct RowIRCudaCodeGenTest : public ::testing::Test {
  std::unique_ptr<cudf::column> f32 =
    cudf::test::fixed_width_column_wrapper<float>({1.0f, 2.0f, 3.0f}).release();
  std::unique_ptr<cudf::column> f64 =
    cudf::test::fixed_width_column_wrapper<double>({1.0, 2.0, 3.0}).release();
  std::unique_ptr<cudf::column> d32 =
    cudf::test::fixed_point_column_wrapper<int32_t>({1, 2, 3}, numeric::scale_type{2}).release();
  std::unique_ptr<cudf::column> i32 =
    cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}).release();
  std::unique_ptr<cudf::column> b8 =
    cudf::test::fixed_width_column_wrapper<bool>({true, false, true}).release();
  cudf::table_view table = cudf::table_view({*f32, *f64, *d32, *i32});
};

TEST_F(RowIRCudaCodeGenTest, GetInput)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    row_ir::code_sink sink;
    row_ir::node get_input_0{row_ir::input_reference{0}};
    get_input_0.instantiate(ctx);
    get_input_0.emit_code(ctx, target_info, sink);

    auto expected_code = "int32_t tmp_0 = in_0;\n";

    EXPECT_EQ(sink.get_code(), expected_code);
  }

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    row_ir::code_sink sink;
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*f32);
    row_ir::node get_input_1{row_ir::input_reference{1}};
    get_input_1.instantiate(ctx);
    get_input_1.emit_code(ctx, target_info, sink);

    auto expected_null_code = "float tmp_0 = in_1;\n";

    EXPECT_EQ(sink.get_code(), expected_null_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, SetOutput)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};

    [[maybe_unused]] auto in0  = ctx.add_input(*i32);
    [[maybe_unused]] auto in1  = ctx.add_input(*f32);
    [[maybe_unused]] auto out0 = ctx.add_output();
    [[maybe_unused]] auto out1 = ctx.add_output();
    row_ir::code_sink sink;
    row_ir::node set_output_0{row_ir::output_reference{0},
                              row_ir::node{row_ir::input_reference{0}}};
    set_output_0.instantiate(ctx);
    set_output_0.emit_code(ctx, target_info, sink);

    auto expected_code =
      R"***(int32_t tmp_0 = in_0;
int32_t tmp_1 = tmp_0;
*out_0 = tmp_1;
)***";

    EXPECT_EQ(sink.get_code(), expected_code);
  }

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    row_ir::code_sink sink;
    [[maybe_unused]] auto in0  = ctx.add_input(*i32);
    [[maybe_unused]] auto in1  = ctx.add_input(*f32);
    [[maybe_unused]] auto out0 = ctx.add_output();
    [[maybe_unused]] auto out1 = ctx.add_output();
    row_ir::node set_output_1{row_ir::output_reference{1},
                              row_ir::node{row_ir::input_reference{1}}};
    set_output_1.instantiate(ctx);
    set_output_1.emit_code(ctx, target_info, sink);

    auto expected_code =
      R"***(float tmp_0 = in_1;
float tmp_1 = tmp_0;
*out_1 = tmp_1;
)***";

    EXPECT_EQ(sink.get_code(), expected_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, UnaryOperation)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*f32);

    row_ir::code_sink sink;
    row_ir::node op{row_ir::opcode::IDENTITY,
                    std::nullopt,
                    cudf::error_policy::PROPAGATE,
                    row_ir::node{row_ir::input_reference{0}}};
    op.instantiate(ctx);
    op.emit_code(ctx, target_info, sink);

    auto expected_code =
      R"***(int32_t tmp_0 = in_0;
int32_t tmp_1 = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::IDENTITY, cudf::error_policy::PROPAGATE>(tmp_0);
)***";

    EXPECT_EQ(sink.get_code(), expected_code);
  }

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*d32);
    row_ir::code_sink sink;
    row_ir::node op{row_ir::opcode::IDENTITY,
                    std::nullopt,
                    cudf::error_policy::PROPAGATE,
                    row_ir::node{row_ir::input_reference{1}}};
    op.instantiate(ctx);
    op.emit_code(ctx, target_info, sink);

    auto expected_null_code =
      R"***(numeric::decimal32 tmp_0 = in_1;
numeric::decimal32 tmp_1 = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::IDENTITY, cudf::error_policy::PROPAGATE>(tmp_0);
)***";

    EXPECT_EQ(sink.get_code(), expected_null_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, BinaryOperation)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*d32);
    row_ir::code_sink sink;
    row_ir::node op{row_ir::opcode::ADD,
                    std::nullopt,
                    cudf::error_policy::PROPAGATE,
                    row_ir::node{row_ir::input_reference{0}},
                    row_ir::node{row_ir::input_reference{0}}};
    op.instantiate(ctx);
    op.emit_code(ctx, target_info, sink);

    auto expected_code =
      R"***(int32_t tmp_0 = in_0;
int32_t tmp_1 = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::ADD, cudf::error_policy::PROPAGATE>(tmp_0, tmp_0);
)***";

    EXPECT_EQ(sink.get_code(), expected_code);
  }

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*d32);
    row_ir::code_sink sink;
    row_ir::node op{row_ir::opcode::ADD,
                    std::nullopt,
                    cudf::error_policy::PROPAGATE,
                    row_ir::node{row_ir::input_reference{1}},
                    row_ir::node{row_ir::input_reference{1}}};
    op.instantiate(ctx);
    op.emit_code(ctx, target_info, sink);

    auto expected_null_code =
      R"***(numeric::decimal32 tmp_0 = in_1;
numeric::decimal32 tmp_1 = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::ADD, cudf::error_policy::PROPAGATE>(tmp_0, tmp_0);
)***";

    EXPECT_EQ(sink.get_code(), expected_null_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, BinaryOperationOverflow)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*d32);
    row_ir::code_sink sink;
    row_ir::node op{row_ir::opcode::ADD_OVERFLOW,
                    std::nullopt,
                    cudf::error_policy::PROPAGATE,
                    row_ir::node{row_ir::input_reference{0}},
                    row_ir::node{row_ir::input_reference{0}}};
    op.instantiate(ctx);
    op.emit_code(ctx, target_info, sink);

    auto expected_code =
      R"***(int32_t tmp_0 = in_0;
auto expected__tmp_1 = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::ADD_OVERFLOW, cudf::error_policy::PROPAGATE>(tmp_0, tmp_0);
if(!expected__tmp_1.has_value()) {
 return expected__tmp_1.error();
}
int32_t tmp_1 = expected__tmp_1.value();
)***";

    EXPECT_EQ(sink.get_code(), expected_code);
  }

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    ctx.set_has_nulls(true);  // needed for error_policy::NULLIFY
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*d32);
    row_ir::code_sink sink;
    row_ir::node op{row_ir::opcode::ADD_OVERFLOW,
                    std::nullopt,
                    cudf::error_policy::NULLIFY,
                    row_ir::node{row_ir::input_reference{0}},
                    row_ir::node{row_ir::input_reference{0}}};
    op.instantiate(ctx);
    op.emit_code(ctx, target_info, sink);

    auto expected_code =
      R"***(cuda::std::optional<int32_t> tmp_0 = in_0;
cuda::std::optional<int32_t> tmp_1 = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::ADD_OVERFLOW, cudf::error_policy::NULLIFY>(tmp_0, tmp_0);
)***";

    EXPECT_EQ(sink.get_code(), expected_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, VectorLengthOperation)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  auto length_operation = [&](int32_t input0, int32_t input1, int32_t output) {
    // This function generates the IR for the vector length operation:
    // length(v) = sqrt(x^2 + y^2)
    // where v = (x, y) and v is a 2D vector.
    auto x2 = row_ir::node(row_ir::opcode::MUL,
                           std::nullopt,
                           cudf::error_policy::PROPAGATE,
                           row_ir::node{row_ir::input_reference{input0}},
                           row_ir::node{row_ir::input_reference{input0}});

    auto y2 = row_ir::node(row_ir::opcode::MUL,
                           std::nullopt,
                           cudf::error_policy::PROPAGATE,
                           row_ir::node{row_ir::input_reference{input1}},
                           row_ir::node{row_ir::input_reference{input1}});

    auto sum = row_ir::node(row_ir::opcode::ADD,
                            std::nullopt,
                            cudf::error_policy::PROPAGATE,
                            std::move(x2),
                            std::move(y2));

    auto length = row_ir::node(
      row_ir::opcode::SQRT, std::nullopt, cudf::error_policy::PROPAGATE, std::move(sum));

    return row_ir::node(row_ir::output_reference{0}, std::move(length));
  };

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0  = ctx.add_input(*f64);
    [[maybe_unused]] auto in1  = ctx.add_input(*f64);
    [[maybe_unused]] auto out0 = ctx.add_output();
    row_ir::code_sink sink;

    auto expr_ir = length_operation(0, 1, 0);
    expr_ir.instantiate(ctx);
    expr_ir.emit_code(ctx, target_info, sink);

    auto expected_code =
      R"***(double tmp_0 = in_0;
double tmp_1 = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::MUL, cudf::error_policy::PROPAGATE>(tmp_0, tmp_0);
double tmp_2 = in_1;
double tmp_3 = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::MUL, cudf::error_policy::PROPAGATE>(tmp_2, tmp_2);
double tmp_4 = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::ADD, cudf::error_policy::PROPAGATE>(tmp_1, tmp_3);
double tmp_5 = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::SQRT, cudf::error_policy::PROPAGATE>(tmp_4);
double tmp_6 = tmp_5;
*out_0 = tmp_6;
)***";

    EXPECT_EQ(sink.get_code(), expected_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, AstConversionBasic)
{
  cudf::ast::tree ast_tree;
  auto forty_two = cudf::numeric_scalar(42);
  auto& column_ref =
    ast_tree.push(cudf::ast::column_reference{0, cudf::ast::table_reference::LEFT});
  auto& forty_two_literal = ast_tree.push(cudf::ast::literal{forty_two});
  auto& add_op            = ast_tree.push(
    cudf::ast::operation{cudf::ast::ast_operator::ADD, forty_two_literal, column_ref});

  auto column = cudf::test::fixed_width_column_wrapper<int32_t>({69, 69, 69, 69, 69, 69}).release();

  auto expected_iter = cuda::constant_iterator{69 + 42};
  auto expected =
    cudf::test::fixed_width_column_wrapper<int32_t>(expected_iter, expected_iter + column->size());

  auto transform_args =
    row_ir::ast_converter::compute_column(row_ir::target::CUDA,
                                          add_op,
                                          cudf::table_view{{*column}},
                                          cudf::table_view{},
                                          "expression",
                                          cudf::get_default_stream(),
                                          cudf::get_current_device_resource_ref());

  ASSERT_EQ(transform_args.scalar_columns.size(), 1);
  ASSERT_EQ(transform_args.scalar_columns[0]->view().size(), 1);
  EXPECT_EQ(transform_args.source_type, cudf::udf_source_type::CUDA);
  EXPECT_EQ(transform_args.is_null_aware, cudf::null_aware::NO);
  EXPECT_EQ(transform_args.outputs.size(), 1);
  EXPECT_EQ(transform_args.outputs[0].nullability, cudf::output_nullability::ALL_VALID);
  EXPECT_EQ(transform_args.outputs[0].type, cudf::data_type{cudf::type_id::INT32});
  ASSERT_EQ(transform_args.inputs.size(), 2);

  /// The first input should be a scalar value of 42
  ASSERT_TRUE(std::holds_alternative<cudf::scalar_column_view>(transform_args.inputs[0]));
  EXPECT_EQ(std::get<cudf::scalar_column_view>(transform_args.inputs[0]).type(),
            cudf::data_type{cudf::type_id::INT32});
  EXPECT_EQ(std::get<cudf::scalar_column_view>(transform_args.inputs[0]).null_count(), 0);

  /// The input column should be the second column in the transform args
  ASSERT_TRUE(std::holds_alternative<cudf::column_view>(transform_args.inputs[1]));
  ASSERT_EQ(std::get<cudf::column_view>(transform_args.inputs[1]).size(), column->size());
  EXPECT_EQ(std::get<cudf::column_view>(transform_args.inputs[1]).type(), column->type());
  EXPECT_EQ(std::get<cudf::column_view>(transform_args.inputs[1]).null_count(),
            column->null_count());

  auto expected_udf =
    R"***(__device__ cudf::errc expression(int32_t* out_0, int32_t in_0, int32_t in_1)
{
int32_t tmp_0 = in_0;
int32_t tmp_1 = in_1;
int32_t tmp_2 = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::ADD, cudf::error_policy::PROPAGATE>(tmp_0, tmp_1);
int32_t tmp_3 = tmp_2;
*out_0 = tmp_3;
return cudf::errc::SUCCESS;
})***";

  EXPECT_EQ(transform_args.udf, expected_udf);

  auto result = cudf::multi_transform(transform_args.udf,
                                      transform_args.source_type,
                                      transform_args.is_null_aware,
                                      transform_args.user_data,
                                      transform_args.inputs,
                                      transform_args.outputs,
                                      std::move(transform_args.string_offsets),
                                      transform_args.row_size);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->get_column(0).view());
}

TEST_F(RowIRCudaCodeGenTest, AstConversionMultipleOutputsEliminatesCommonSubexpressions)
{
  cudf::ast::tree ast_tree;
  auto& first_column_ref =
    ast_tree.push(cudf::ast::column_reference{0, cudf::ast::table_reference::LEFT});
  auto& second_column_ref =
    ast_tree.push(cudf::ast::column_reference{0, cudf::ast::table_reference::LEFT});
  auto& first_add = ast_tree.push(
    cudf::ast::operation{cudf::ast::ast_operator::ADD, first_column_ref, first_column_ref});
  auto& second_add = ast_tree.push(
    cudf::ast::operation{cudf::ast::ast_operator::ADD, second_column_ref, second_column_ref});
  auto& multiply = ast_tree.push(
    cudf::ast::operation{cudf::ast::ast_operator::MUL, second_add, second_column_ref});

  auto column = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}).release();
  std::reference_wrapper<cudf::ast::expression const> expressions[]{first_add, multiply};
  auto transform_args =
    row_ir::ast_converter::compute_columns(row_ir::target::CUDA,
                                           expressions,
                                           cudf::table_view{{*column}},
                                           cudf::table_view{},
                                           "expression",
                                           cudf::get_default_stream(),
                                           cudf::get_current_device_resource_ref());

  ASSERT_EQ(transform_args.inputs.size(), 1);
  ASSERT_EQ(transform_args.outputs.size(), 2);
  EXPECT_FALSE(transform_args.may_propagate_error);
  auto const add = transform_args.udf.find("opcode::ADD");
  ASSERT_NE(add, std::string::npos);
  EXPECT_EQ(transform_args.udf.find("opcode::ADD", add + 1), std::string::npos);

  auto result = cudf::multi_transform(transform_args.udf,
                                      transform_args.source_type,
                                      transform_args.is_null_aware,
                                      transform_args.user_data,
                                      transform_args.inputs,
                                      transform_args.outputs,
                                      std::move(transform_args.string_offsets),
                                      transform_args.row_size);

  auto expected_add      = cudf::test::fixed_width_column_wrapper<int32_t>({2, 4, 6});
  auto expected_multiply = cudf::test::fixed_width_column_wrapper<int32_t>({2, 8, 18});
  auto expected          = cudf::table_view{{expected_add, expected_multiply}};
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
}

TEST_F(RowIRCudaCodeGenTest, AstConversionMultipleOutputsKeepsLiteralsDistinct)
{
  cudf::ast::tree ast_tree;
  auto& first_column_ref =
    ast_tree.push(cudf::ast::column_reference{0, cudf::ast::table_reference::LEFT});
  auto& second_column_ref =
    ast_tree.push(cudf::ast::column_reference{0, cudf::ast::table_reference::LEFT});
  auto one = cudf::numeric_scalar<int32_t>(1);
  auto two = cudf::numeric_scalar<int32_t>(2);
  auto& one_literal = ast_tree.push(cudf::ast::literal{one});
  auto& two_literal = ast_tree.push(cudf::ast::literal{two});
  auto& add_one      = ast_tree.push(
    cudf::ast::operation{cudf::ast::ast_operator::ADD, first_column_ref, one_literal});
  auto& add_two = ast_tree.push(
    cudf::ast::operation{cudf::ast::ast_operator::ADD, second_column_ref, two_literal});

  auto column = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}).release();
  std::reference_wrapper<cudf::ast::expression const> expressions[]{add_one, add_two};
  auto transform_args =
    row_ir::ast_converter::compute_columns(row_ir::target::CUDA,
                                           expressions,
                                           cudf::table_view{{*column}},
                                           cudf::table_view{},
                                           "expression",
                                           cudf::get_default_stream(),
                                           cudf::get_current_device_resource_ref());

  ASSERT_EQ(transform_args.inputs.size(), 3);
  auto const first_add = transform_args.udf.find("opcode::ADD");
  ASSERT_NE(first_add, std::string::npos);
  EXPECT_NE(transform_args.udf.find("opcode::ADD", first_add + 1), std::string::npos);
}

TEST_F(RowIRCudaCodeGenTest, AstConversionMultipleOutputsSharesLiteralIdentity)
{
  cudf::ast::tree ast_tree;
  auto& first_column_ref =
    ast_tree.push(cudf::ast::column_reference{0, cudf::ast::table_reference::LEFT});
  auto& second_column_ref =
    ast_tree.push(cudf::ast::column_reference{0, cudf::ast::table_reference::LEFT});
  auto literal_value = cudf::numeric_scalar<int32_t>(7);
  auto& first_literal  = ast_tree.push(cudf::ast::literal{literal_value});
  auto& second_literal = ast_tree.push(cudf::ast::literal{literal_value});
  auto& first_add      = ast_tree.push(
    cudf::ast::operation{cudf::ast::ast_operator::ADD, first_column_ref, first_literal});
  auto& second_add = ast_tree.push(
    cudf::ast::operation{cudf::ast::ast_operator::ADD, second_column_ref, second_literal});

  auto column = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}).release();
  std::reference_wrapper<cudf::ast::expression const> expressions[]{first_add, second_add};
  auto transform_args =
    row_ir::ast_converter::compute_columns(row_ir::target::CUDA,
                                           expressions,
                                           cudf::table_view{{*column}},
                                           cudf::table_view{},
                                           "expression",
                                           cudf::get_default_stream(),
                                           cudf::get_current_device_resource_ref());

  ASSERT_EQ(transform_args.inputs.size(), 2);
  auto const add = transform_args.udf.find("opcode::ADD");
  ASSERT_NE(add, std::string::npos);
  EXPECT_EQ(transform_args.udf.find("opcode::ADD", add + 1), std::string::npos);
}

TEST_F(RowIRCudaCodeGenTest, AstConversionGeneratesMinimalLtoTopology)
{
  auto first  = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}).release();
  auto second = cudf::test::fixed_width_column_wrapper<int32_t>({4, 5, 6}).release();
  auto third  = cudf::test::fixed_width_column_wrapper<int32_t>({7, 8, 9}).release();
  auto table  = cudf::table_view{{*first, *second, *third}};

  auto ast_tree    = cudf::ast::tree{};
  auto& first_ref  = ast_tree.push(cudf::ast::column_reference{0});
  auto& second_ref = ast_tree.push(cudf::ast::column_reference{1});
  auto& third_ref  = ast_tree.push(cudf::ast::column_reference{2});
  auto& add        = ast_tree.push(
    cudf::ast::operation{cudf::ast::ast_operator::ADD, first_ref, second_ref});
  auto& multiply = ast_tree.push(
    cudf::ast::operation{cudf::ast::ast_operator::MUL, add, third_ref});
  std::reference_wrapper<cudf::ast::expression const> expressions[]{add, multiply};

  auto transform_args =
    row_ir::ast_converter::compute_columns(row_ir::target::CUDA,
                                           expressions,
                                           table,
                                           cudf::table_view{},
                                           "expression",
                                           cudf::get_default_stream(),
                                           cudf::get_current_device_resource_ref());

  ASSERT_TRUE(transform_args.lto_udf.has_value());
  auto const& code = *transform_args.lto_udf;
  EXPECT_NE(code.find("extern \"C\" __device__ int transform(uint32_t* out_0, uint32_t* out_1"),
            std::string::npos);
  auto const add_call = code.find(" = cudf_row_ir_add_u32(");
  ASSERT_NE(add_call, std::string::npos);
  EXPECT_EQ(code.find(" = cudf_row_ir_add_u32(", add_call + 1), std::string::npos);
  EXPECT_NE(code.find(" = cudf_row_ir_mul_u32("), std::string::npos);
}

TEST_F(RowIRCudaCodeGenTest, AstConversionMultipleOutputsTracksNullAndErrorMetadata)
{
  auto nullable =
    cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2, 3}, {1, 0, 1}}.release();
  auto valid = cudf::test::fixed_width_column_wrapper<int32_t>({4, 5, 6}).release();
  auto table = cudf::table_view{{*nullable, *valid}};

  auto nullable_ref = cudf::ast::column_reference{0};
  auto valid_ref    = cudf::ast::column_reference{1};
  std::reference_wrapper<cudf::ast::expression const> expressions[]{nullable_ref, valid_ref};
  auto transform_args =
    row_ir::ast_converter::compute_columns(row_ir::target::CUDA,
                                           expressions,
                                           table,
                                           cudf::table_view{},
                                           "expression",
                                           cudf::get_default_stream(),
                                           cudf::get_current_device_resource_ref());

  EXPECT_EQ(transform_args.is_null_aware, cudf::null_aware::YES);
  ASSERT_EQ(transform_args.outputs.size(), 2);
  EXPECT_EQ(transform_args.outputs[0].nullability, cudf::output_nullability::PRESERVE);
  EXPECT_EQ(transform_args.outputs[1].nullability, cudf::output_nullability::ALL_VALID);
  EXPECT_FALSE(transform_args.lto_udf.has_value());

  auto ast_tree = cudf::ast::tree{};
  auto& add_overflow = cudf::ast::jit::operation(
    ast_tree, cudf::ast::jit::op::ADD_OVERFLOW, {nullable_ref, valid_ref});
  std::reference_wrapper<cudf::ast::expression const> fallible_expressions[]{add_overflow};
  auto fallible_args =
    row_ir::ast_converter::compute_columns(row_ir::target::CUDA,
                                           fallible_expressions,
                                           table,
                                           cudf::table_view{},
                                           "expression",
                                           cudf::get_default_stream(),
                                           cudf::get_current_device_resource_ref());
  EXPECT_TRUE(fallible_args.may_propagate_error);
  EXPECT_FALSE(fallible_args.lto_udf.has_value());
}

TEST_F(RowIRCudaCodeGenTest, FilterPredicate)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*b8);
    row_ir::code_sink sink;
    row_ir::node filter_predicate(row_ir::opcode::PREDICATE,
                                  std::nullopt,
                                  cudf::error_policy::PROPAGATE,
                                  row_ir::node{row_ir::input_reference{0}});
    filter_predicate.instantiate(ctx);
    filter_predicate.emit_code(ctx, target_info, sink);

    auto expected_code = R"***(bool tmp_0 = in_0;
bool tmp_1 = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::PREDICATE, cudf::error_policy::PROPAGATE>(tmp_0);
)***";

    EXPECT_EQ(sink.get_code(), expected_code);
  }

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*b8);
    row_ir::code_sink sink;
    row_ir::node filter_predicate(row_ir::opcode::PREDICATE,
                                  std::nullopt,
                                  cudf::error_policy::PROPAGATE,
                                  row_ir::node{row_ir::input_reference{0}});
    ctx.set_has_nulls(true);
    filter_predicate.instantiate(ctx);
    filter_predicate.emit_code(ctx, target_info, sink);

    auto expected_code = R"***(cuda::std::optional<bool> tmp_0 = in_0;
cuda::std::optional<bool> tmp_1 = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::PREDICATE, cudf::error_policy::PROPAGATE>(tmp_0);
)***";

    EXPECT_EQ(sink.get_code(), expected_code);
  }
}

CUDF_TEST_PROGRAM_MAIN()
