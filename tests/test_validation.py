import pytest
from pydantic import ValidationError

from templateparity.api import CompareRequest
from templateparity.models import ToolDefinition


def test_tool_schema_requires_object_root():
    with pytest.raises(ValidationError):
        ToolDefinition(name="bad", parameter_schema={"type": "array"})


def test_compare_api_rejects_remote_llama_url():
    with pytest.raises(ValidationError):
        CompareRequest(
            fixture="simple-user", mlx_model="/tmp/model", llama_url="https://example.com"
        )


def test_compare_api_rejects_unknown_fixture():
    with pytest.raises(ValidationError):
        CompareRequest(fixture="../../private", mlx_model="/tmp/model")
