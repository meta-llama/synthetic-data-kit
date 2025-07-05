# Synthetic Data Kit Tests

This directory contains tests for the Synthetic Data Kit.

## Test Structure

The tests are organized into three categories:

- **Unit Tests** (`unit/`): Test individual components in isolation
- **Integration Tests** (`integration/`): Test interactions between components
- **Functional Tests** (`functional/`): Test end-to-end workflows

## Running Tests

Run all tests:

```bash
pytest
```

Run specific test categories:

```bash
# Unit tests only
pytest tests/unit

# Integration tests only
pytest tests/integration

# Functional tests only
pytest tests/functional
```

Run with coverage:

```bash
pytest --cov=synthetic_data_kit
```

## Important Notes for Test Maintenance

### Environment Variables

The tests are designed to work with mock API keys in CI environments. Make sure the following environment variables are set in your CI workflows:

```yaml
env:
  PROJECT_TEST_ENV: "1"
  OPENAI_API_KEY: "sk-mock-key-for-testing"
  API_ENDPOINT_KEY: "mock-llama-api-key-for-testing"
```

### Mocking Strategy

Our tests use extensive mocking to avoid dependencies on external services:

1. **Config Mocking**: We mock `load_config` to return predictable configurations
2. **API Client Mocking**: We mock `openai.OpenAI` to avoid actual API calls
3. **Environment Variables**: We use `patch.dict(os.environ, ...)` to set environment variables during tests

### Test Isolation

Each test should clean up after itself, removing any temporary files or directories it creates. Use the `try/finally` pattern to ensure cleanup code runs even if the test fails.

## Writing New Tests

When adding new features or fixing bugs, please add appropriate tests. Follow these guidelines:

1. Place tests in the appropriate category directory
2. Use descriptive test names (`test_feature_name.py`)
3. Follow the existing test patterns
4. Use fixtures from `conftest.py` when possible
5. Add test markers (`@pytest.mark.unit`, etc.)
6. Mock external dependencies and API calls

### Example Test Template

```python
@pytest.mark.unit  # or integration, functional
def test_my_feature():
    """Test description."""
    # Setup - create any test data or mock objects

    # Mock dependencies
    with patch("some.dependency", return_value=mock_value):
        # Exercise - call the function you're testing
        result = my_module.my_function()

        # Verify - check the result meets expectations
        assert result == expected_value
```
