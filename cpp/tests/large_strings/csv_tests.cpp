/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "large_strings_fixture.hpp"

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>

#include <fstream>
#include <string>
#include <vector>

struct CsvLargeReaderTest : public cudf::test::StringsLargeTest,
                            public testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(CsvLargeInput, CsvLargeReaderTest, testing::Bool());

TEST_P(CsvLargeReaderTest, InputExceedsCompactOffsets)
{
  cudf::test::TempDirTestEnvironment temp_dir;
  auto const path = temp_dir.get_temp_dir() + "wide_offsets.csv";
  {
    std::ofstream output(path, std::ios::binary);
    output << "id,ignored,value\n";
    // Keep output small while placing the last selected string beyond the 32-bit offset range.
    std::string const padding(1 << 20, 'x');
    for (int row = 0; row < 4096; ++row) {
      output << row % 10 << ',';
      output << padding;
      output << ",\"v\"\"" << row % 10 << "\"\n";
    }
    ASSERT_TRUE(output.good());
  }
  auto const strings = GetParam();
  auto options       = cudf::io::csv_reader_options::builder(cudf::io::source_info{path})
                   .header(0)
                   .dtypes(std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT32},
                                                        cudf::data_type{cudf::type_id::STRING},
                                                        cudf::data_type{cudf::type_id::STRING}})
                   .use_cols_names(strings ? std::vector<std::string>{"id", "value"}
                                           : std::vector<std::string>{"id"});
  auto const result = cudf::io::read_csv(options.build());
  std::vector<int32_t> expected_ids;
  std::vector<std::string> expected_values;
  for (int row = 0; row < 4096; ++row) {
    expected_ids.push_back(row % 10);
    expected_values.push_back("v\"" + std::to_string(row % 10));
  }
  cudf::test::fixed_width_column_wrapper<int32_t> const ids(expected_ids.begin(),
                                                            expected_ids.end());
  cudf::test::strings_column_wrapper const values(expected_values.begin(), expected_values.end());
  auto const expected = strings ? cudf::table_view{{ids, values}} : cudf::table_view{{ids}};
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, result.tbl->view());
}
