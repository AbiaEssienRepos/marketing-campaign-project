def test_schema(sample_input_data):

    # Given
    expected_no_of_columns = 27
    expected_no_of_rows = 448

    # When
    data = sample_input_data
    actual_columns = len(data.columns)
    actual_rows = len(data)

    # Then
    assert actual_columns == expected_no_of_columns
    assert actual_rows == expected_no_of_rows


def test_missing_values(sample_input_data):

    # Given
    expected_vars_with_na = 1
    variable_with_missing_data = "Income"

    # When
    actual_vars_with_na = [
        var
        for var in sample_input_data.columns
        if sample_input_data[var].isnull().sum() > 0
    ]

    # Then
    assert len(actual_vars_with_na) == expected_vars_with_na
    assert actual_vars_with_na[0] == variable_with_missing_data
