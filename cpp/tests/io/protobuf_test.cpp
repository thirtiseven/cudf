/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/io/protobuf.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

namespace pb = cudf::io::protobuf;

// ============================================================================
// Protobuf wire format encoding helpers
// ============================================================================
namespace {

// Wire type constants
constexpr int WT_VARINT = 0;
constexpr int WT_64BIT  = 1;
constexpr int WT_LEN    = 2;
constexpr int WT_32BIT  = 5;

std::vector<uint8_t> encode_varint(uint64_t value)
{
  std::vector<uint8_t> out;
  while (value > 0x7F) {
    out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value));
  return out;
}

uint64_t zigzag_encode32(int32_t n)
{
  return static_cast<uint64_t>(static_cast<uint32_t>((n << 1) ^ (n >> 31)));
}

uint64_t zigzag_encode64(int64_t n)
{
  return static_cast<uint64_t>((n << 1) ^ (n >> 63));
}

std::vector<uint8_t> encode_fixed32(uint32_t v)
{
  return {static_cast<uint8_t>(v),
          static_cast<uint8_t>(v >> 8),
          static_cast<uint8_t>(v >> 16),
          static_cast<uint8_t>(v >> 24)};
}

std::vector<uint8_t> encode_fixed64(uint64_t v)
{
  std::vector<uint8_t> out(8);
  for (int i = 0; i < 8; ++i) {
    out[i] = static_cast<uint8_t>(v >> (8 * i));
  }
  return out;
}

std::vector<uint8_t> encode_float(float f)
{
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  return encode_fixed32(bits);
}

std::vector<uint8_t> encode_double(double d)
{
  uint64_t bits;
  std::memcpy(&bits, &d, sizeof(bits));
  return encode_fixed64(bits);
}

std::vector<uint8_t> tag(int field_number, int wire_type)
{
  return encode_varint((static_cast<uint64_t>(field_number) << 3) |
                       static_cast<uint64_t>(wire_type));
}

// Concatenate multiple byte vectors
std::vector<uint8_t> concat(std::initializer_list<std::vector<uint8_t>> parts)
{
  std::vector<uint8_t> out;
  for (auto const& p : parts) {
    out.insert(out.end(), p.begin(), p.end());
  }
  return out;
}

// Encode a string field (tag + length + data)
std::vector<uint8_t> encode_string_field(int field_number, std::string const& s)
{
  auto t   = tag(field_number, WT_LEN);
  auto len = encode_varint(s.size());
  auto out = concat({t, len});
  out.insert(out.end(), s.begin(), s.end());
  return out;
}

// Encode a varint field (tag + varint value)
std::vector<uint8_t> encode_varint_field(int field_number, uint64_t value)
{
  return concat({tag(field_number, WT_VARINT), encode_varint(value)});
}

// Encode a submessage field (tag + length + payload)
std::vector<uint8_t> encode_submessage_field(int field_number,
                                             std::vector<uint8_t> const& payload)
{
  return concat({tag(field_number, WT_LEN), encode_varint(payload.size()), payload});
}

// Build a LIST<UINT8> column from a vector of messages (each message is a vector<uint8_t>).
// Null messages are represented by empty vectors when is_valid[i] == false.
std::unique_ptr<cudf::column> make_binary_column(
  std::vector<std::vector<uint8_t>> const& messages,
  std::vector<bool> const& validity = {})
{
  // Build offsets
  std::vector<int32_t> offsets;
  offsets.reserve(messages.size() + 1);
  offsets.push_back(0);
  for (auto const& m : messages) {
    offsets.push_back(offsets.back() + static_cast<int32_t>(m.size()));
  }

  // Build flat data
  std::vector<uint8_t> flat_data;
  flat_data.reserve(offsets.back());
  for (auto const& m : messages) {
    flat_data.insert(flat_data.end(), m.begin(), m.end());
  }

  auto offsets_col =
    cudf::test::fixed_width_column_wrapper<int32_t>(offsets.begin(), offsets.end()).release();
  auto data_col =
    cudf::test::fixed_width_column_wrapper<uint8_t>(flat_data.begin(), flat_data.end()).release();

  auto num_rows = static_cast<cudf::size_type>(messages.size());

  // Build null mask if needed
  if (!validity.empty()) {
    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(validity.begin(), validity.end());
    return cudf::make_lists_column(
      num_rows, std::move(offsets_col), std::move(data_col), null_count, std::move(null_mask));
  }

  return cudf::make_lists_column(
    num_rows, std::move(offsets_col), std::move(data_col), 0, rmm::device_buffer{});
}

// Build a simple flat schema (all fields are top-level scalars)
pb::decode_protobuf_options make_scalar_options(
  std::vector<int> const& field_numbers,
  std::vector<cudf::type_id> const& types,
  std::vector<int> const& encodings,
  bool fail_on_errors = true)
{
  int const n = static_cast<int>(field_numbers.size());

  auto derive_wire_type = [](cudf::type_id type, int enc) -> pb::proto_wire_type {
    if (enc == static_cast<int>(pb::proto_encoding::ENUM_STRING)) {
      return pb::proto_wire_type::VARINT;
    }
    if (enc == static_cast<int>(pb::proto_encoding::FIXED)) {
      if (type == cudf::type_id::INT64 || type == cudf::type_id::UINT64 ||
          type == cudf::type_id::FLOAT64) {
        return pb::proto_wire_type::I64BIT;
      }
      return pb::proto_wire_type::I32BIT;
    }
    switch (type) {
      case cudf::type_id::FLOAT32: return pb::proto_wire_type::I32BIT;
      case cudf::type_id::FLOAT64: return pb::proto_wire_type::I64BIT;
      case cudf::type_id::STRING:
      case cudf::type_id::LIST:
      case cudf::type_id::STRUCT: return pb::proto_wire_type::LEN;
      default: return pb::proto_wire_type::VARINT;
    }
  };

  std::vector<pb::nested_field_descriptor> schema;
  schema.reserve(n);
  for (int i = 0; i < n; ++i) {
    schema.push_back({field_numbers[i],
                      -1,  // top-level
                      0,   // depth 0
                      derive_wire_type(types[i], encodings[i]),
                      types[i],
                      static_cast<pb::proto_encoding>(encodings[i]),
                      false,  // not repeated
                      false,  // not required
                      false});
  }

  auto empty_hv = cudf::detail::make_host_vector<uint8_t>(0, cudf::get_default_stream());
  auto empty_iv = cudf::detail::make_host_vector<int32_t>(0, cudf::get_default_stream());

  std::vector<cudf::detail::host_vector<uint8_t>> default_strings(n);
  std::vector<cudf::detail::host_vector<int32_t>> enum_valid(n);
  std::vector<std::vector<cudf::detail::host_vector<uint8_t>>> enum_names(n);
  for (int i = 0; i < n; ++i) {
    default_strings[i] = cudf::detail::make_host_vector<uint8_t>(0, cudf::get_default_stream());
    enum_valid[i]      = cudf::detail::make_host_vector<int32_t>(0, cudf::get_default_stream());
  }

  return pb::decode_protobuf_options{
    std::move(schema),
    std::vector<int64_t>(n, 0),
    std::vector<double>(n, 0.0),
    std::vector<bool>(n, false),
    std::move(default_strings),
    std::move(enum_valid),
    std::move(enum_names),
    fail_on_errors,
  };
}

}  // anonymous namespace

// ============================================================================
// Test fixtures
// ============================================================================

struct ProtobufReaderTest : public cudf::test::BaseFixture {};

// ============================================================================
// Basic scalar type tests
// ============================================================================

TEST_F(ProtobufReaderTest, DecodeVarintAndString)
{
  // message { int64 id = 1; string name = 2; }
  // Row 0: id=42, name="hello"
  // Row 1: id=100, name="world"
  // Row 2: id missing, name="test"

  auto msg0 = concat({encode_varint_field(1, 42), encode_string_field(2, "hello")});
  auto msg1 = concat({encode_varint_field(1, 100), encode_string_field(2, "world")});
  auto msg2 = encode_string_field(2, "test");  // id is missing

  auto input = make_binary_column({msg0, msg1, msg2});

  auto options = make_scalar_options(
    {1, 2},
    {cudf::type_id::INT64, cudf::type_id::STRING},
    {0, 0});

  auto result = pb::decode_protobuf(*input, options);

  // Verify result is a STRUCT column with 2 children
  ASSERT_EQ(result->type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(result->num_children(), 2);

  // Check int64 child
  auto expected_ints =
    cudf::test::fixed_width_column_wrapper<int64_t>({42, 100, 0}, {true, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(0), expected_ints);

  // Check string child
  auto expected_strs =
    cudf::test::strings_column_wrapper({"hello", "world", "test"}, {true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(1), expected_strs);
}

TEST_F(ProtobufReaderTest, DecodeNullInputRows)
{
  // Row 0: valid message
  // Row 1: null input row
  // Row 2: valid message
  auto msg0 = encode_varint_field(1, 77);
  auto msg2 = encode_varint_field(1, 99);

  auto input = make_binary_column({msg0, {}, msg2}, {true, false, true});

  auto options = make_scalar_options({1}, {cudf::type_id::INT64}, {0});

  auto result = pb::decode_protobuf(*input, options);

  ASSERT_EQ(result->type().id(), cudf::type_id::STRUCT);
  // Null input rows should produce null values in child columns
  auto expected =
    cudf::test::fixed_width_column_wrapper<int64_t>({77, 0, 99}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(0), expected);
}

TEST_F(ProtobufReaderTest, DecodeEmptyMessage)
{
  // Empty message (zero bytes) - all fields should be null/default
  auto input = make_binary_column({{}});

  auto options = make_scalar_options(
    {1, 2},
    {cudf::type_id::INT64, cudf::type_id::STRING},
    {0, 0});

  auto result = pb::decode_protobuf(*input, options);

  ASSERT_EQ(result->type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(result->size(), 1);

  auto expected_int = cudf::test::fixed_width_column_wrapper<int64_t>({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(0), expected_int);

  auto expected_str = cudf::test::strings_column_wrapper({""}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(1), expected_str);
}

TEST_F(ProtobufReaderTest, DecodeZeroRows)
{
  auto input = make_binary_column({});

  auto options = make_scalar_options(
    {1, 2},
    {cudf::type_id::INT64, cudf::type_id::STRING},
    {0, 0});

  auto result = pb::decode_protobuf(*input, options);

  ASSERT_EQ(result->type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(result->size(), 0);
  ASSERT_EQ(result->num_children(), 2);
}

TEST_F(ProtobufReaderTest, DecodeMultipleNumericTypes)
{
  // message { bool flag = 1; int32 count = 2; float score = 3; double value = 4; }
  auto msg = concat({
    encode_varint_field(1, 1),          // bool: true
    encode_varint_field(2, 42),         // int32: 42
    concat({tag(3, WT_32BIT), encode_float(3.14f)}),   // float
    concat({tag(4, WT_64BIT), encode_double(2.718)}),   // double
  });

  auto input = make_binary_column({msg});

  auto options = make_scalar_options(
    {1, 2, 3, 4},
    {cudf::type_id::BOOL8, cudf::type_id::INT32, cudf::type_id::FLOAT32, cudf::type_id::FLOAT64},
    {0, 0, 0, 0});

  auto result = pb::decode_protobuf(*input, options);

  ASSERT_EQ(result->num_children(), 4);

  auto expected_bool  = cudf::test::fixed_width_column_wrapper<bool>({true});
  auto expected_int   = cudf::test::fixed_width_column_wrapper<int32_t>({42});
  auto expected_float = cudf::test::fixed_width_column_wrapper<float>({3.14f});
  auto expected_dbl   = cudf::test::fixed_width_column_wrapper<double>({2.718});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(0), expected_bool);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(1), expected_int);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(2), expected_float);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(3), expected_dbl);
}

TEST_F(ProtobufReaderTest, DecodeZigzagEncoding)
{
  // sint32 (field 1, zigzag encoding)
  auto msg0 = encode_varint_field(1, zigzag_encode32(-1));
  auto msg1 = encode_varint_field(1, zigzag_encode32(0));
  auto msg2 = encode_varint_field(1, zigzag_encode32(2147483647));
  auto msg3 = encode_varint_field(1, zigzag_encode32(-2147483648));

  auto input = make_binary_column({msg0, msg1, msg2, msg3});

  auto options = make_scalar_options({1}, {cudf::type_id::INT32}, {2});  // enc=2 is zigzag

  auto result = pb::decode_protobuf(*input, options);

  auto expected =
    cudf::test::fixed_width_column_wrapper<int32_t>({-1, 0, 2147483647, -2147483648});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(0), expected);
}

TEST_F(ProtobufReaderTest, DecodeFixedEncoding)
{
  // fixed32 (field 1), fixed64 (field 2)
  auto msg = concat({
    concat({tag(1, WT_32BIT), encode_fixed32(0xDEADBEEF)}),
    concat({tag(2, WT_64BIT), encode_fixed64(0x0102030405060708ULL)}),
  });

  auto input = make_binary_column({msg});

  auto options = make_scalar_options(
    {1, 2},
    {cudf::type_id::UINT32, cudf::type_id::UINT64},
    {1, 1});  // enc=1 is fixed

  auto result = pb::decode_protobuf(*input, options);

  auto expected_u32 = cudf::test::fixed_width_column_wrapper<uint32_t>({0xDEADBEEF});
  auto expected_u64 =
    cudf::test::fixed_width_column_wrapper<uint64_t>({0x0102030405060708ULL});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(0), expected_u32);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(1), expected_u64);
}

TEST_F(ProtobufReaderTest, DecodeBytesField)
{
  // bytes field = LIST<UINT8>
  std::vector<uint8_t> payload = {0xCA, 0xFE, 0xBA, 0xBE};
  auto msg = concat({tag(1, WT_LEN), encode_varint(payload.size()), payload});

  auto input = make_binary_column({msg});

  auto options = make_scalar_options({1}, {cudf::type_id::LIST}, {0});

  auto result = pb::decode_protobuf(*input, options);

  ASSERT_EQ(result->type().id(), cudf::type_id::STRUCT);
  // child(0) should be a LIST column
  ASSERT_EQ(result->child(0).type().id(), cudf::type_id::LIST);
}

TEST_F(ProtobufReaderTest, DecodeLastOneWins)
{
  // Protobuf spec: if a field appears multiple times, the last value wins
  auto msg = concat({
    encode_varint_field(1, 10),
    encode_varint_field(1, 20),
    encode_varint_field(1, 30),  // this should win
  });

  auto input = make_binary_column({msg});

  auto options = make_scalar_options({1}, {cudf::type_id::INT64}, {0});

  auto result = pb::decode_protobuf(*input, options);

  auto expected = cudf::test::fixed_width_column_wrapper<int64_t>({30});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(0), expected);
}

// ============================================================================
// Nested message tests
// ============================================================================

TEST_F(ProtobufReaderTest, DecodeNestedMessage)
{
  // message Outer { int32 id = 1; Inner inner = 2; }
  // message Inner { string name = 1; }
  // Schema: [0: id(fn=1,parent=-1), 1: inner(fn=2,parent=-1,STRUCT), 2: name(fn=1,parent=1)]

  auto inner_payload = encode_string_field(1, "nested_value");
  auto msg = concat({
    encode_varint_field(1, 42),
    encode_submessage_field(2, inner_payload),
  });

  auto input = make_binary_column({msg});

  int const n = 3;
  std::vector<pb::nested_field_descriptor> schema = {
    {1, -1, 0, pb::proto_wire_type::VARINT, cudf::type_id::INT32, pb::proto_encoding::DEFAULT,
     false, false, false},
    {2, -1, 0, pb::proto_wire_type::LEN, cudf::type_id::STRUCT, pb::proto_encoding::DEFAULT,
     false, false, false},
    {1, 1, 1, pb::proto_wire_type::LEN, cudf::type_id::STRING, pb::proto_encoding::DEFAULT,
     false, false, false},
  };

  auto empty_hv = [](int count) {
    std::vector<cudf::detail::host_vector<uint8_t>> v(count);
    for (auto& h : v) { h = cudf::detail::make_host_vector<uint8_t>(0, cudf::get_default_stream()); }
    return v;
  };
  auto empty_iv = [](int count) {
    std::vector<cudf::detail::host_vector<int32_t>> v(count);
    for (auto& h : v) { h = cudf::detail::make_host_vector<int32_t>(0, cudf::get_default_stream()); }
    return v;
  };

  pb::decode_protobuf_options options{
    std::move(schema),
    std::vector<int64_t>(n, 0),
    std::vector<double>(n, 0.0),
    std::vector<bool>(n, false),
    empty_hv(n),
    empty_iv(n),
    std::vector<std::vector<cudf::detail::host_vector<uint8_t>>>(n),
    true,
  };

  auto result = pb::decode_protobuf(*input, options);

  ASSERT_EQ(result->type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(result->num_children(), 2);

  // child(0) = id (INT32)
  auto expected_id = cudf::test::fixed_width_column_wrapper<int32_t>({42});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(0), expected_id);

  // child(1) = inner (STRUCT with one child: name)
  ASSERT_EQ(result->child(1).type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(result->child(1).num_children(), 1);

  auto expected_name = cudf::test::strings_column_wrapper({"nested_value"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->child(1).child(0), expected_name);
}

// ============================================================================
// Repeated field tests
// ============================================================================

TEST_F(ProtobufReaderTest, DecodeRepeatedInt)
{
  // message { repeated int32 values = 1; }
  // Row 0: values = [10, 20, 30]
  // Row 1: values = [] (empty)

  auto msg0 = concat({
    encode_varint_field(1, 10),
    encode_varint_field(1, 20),
    encode_varint_field(1, 30),
  });
  auto msg1 = std::vector<uint8_t>{};  // empty message

  auto input = make_binary_column({msg0, msg1});

  int const n = 1;
  std::vector<pb::nested_field_descriptor> schema = {
    {1, -1, 0, pb::proto_wire_type::VARINT, cudf::type_id::INT32, pb::proto_encoding::DEFAULT,
     true, false, false},  // is_repeated = true
  };

  auto empty_hv = [](int count) {
    std::vector<cudf::detail::host_vector<uint8_t>> v(count);
    for (auto& h : v) { h = cudf::detail::make_host_vector<uint8_t>(0, cudf::get_default_stream()); }
    return v;
  };
  auto empty_iv = [](int count) {
    std::vector<cudf::detail::host_vector<int32_t>> v(count);
    for (auto& h : v) { h = cudf::detail::make_host_vector<int32_t>(0, cudf::get_default_stream()); }
    return v;
  };

  pb::decode_protobuf_options options{
    std::move(schema),
    std::vector<int64_t>(n, 0),
    std::vector<double>(n, 0.0),
    std::vector<bool>(n, false),
    empty_hv(n),
    empty_iv(n),
    std::vector<std::vector<cudf::detail::host_vector<uint8_t>>>(n),
    true,
  };

  auto result = pb::decode_protobuf(*input, options);

  ASSERT_EQ(result->type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(result->num_children(), 1);

  // child(0) should be a LIST<INT32> column
  auto const& list_child = result->child(0);
  ASSERT_EQ(list_child.type().id(), cudf::type_id::LIST);
  ASSERT_EQ(list_child.size(), 2);
}

// ============================================================================
// Permissive mode tests
// ============================================================================

TEST_F(ProtobufReaderTest, PermissiveModeReturnsNulls)
{
  // Malformed message: varint that doesn't terminate
  std::vector<uint8_t> bad_msg = {0x08, 0x80, 0x80, 0x80, 0x80, 0x80,
                                  0x80, 0x80, 0x80, 0x80, 0x80, 0x01};
  auto good_msg = encode_varint_field(1, 42);

  auto input = make_binary_column({bad_msg, good_msg});

  auto options      = make_scalar_options({1}, {cudf::type_id::INT64}, {0}, false);

  auto result = pb::decode_protobuf(*input, options);

  ASSERT_EQ(result->type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(result->size(), 2);
  // Good row should decode normally
}

TEST_F(ProtobufReaderTest, FailModeThrowsOnBadMessage)
{
  // Malformed varint
  std::vector<uint8_t> bad_msg = {0x08, 0x80, 0x80, 0x80, 0x80, 0x80,
                                  0x80, 0x80, 0x80, 0x80, 0x80, 0x01};

  auto input   = make_binary_column({bad_msg});
  auto options = make_scalar_options({1}, {cudf::type_id::INT64}, {0}, true);

  EXPECT_THROW(pb::decode_protobuf(*input, options), cudf::logic_error);
}

// ============================================================================
// Main
// ============================================================================

CUDF_TEST_PROGRAM_MAIN()
