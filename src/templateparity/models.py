from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = ""
    parameter_schema: dict[str, Any]

    @field_validator("parameter_schema")
    @classmethod
    def valid_object_schema(cls, value):
        if value.get("type") != "object":
            raise ValueError("Tool parameter_schema must declare type: object")
        if "properties" in value and not isinstance(value["properties"], dict):
            raise ValueError("Tool parameter_schema properties must be an object")
        if "required" in value and (
            not isinstance(value["required"], list)
            or not all(isinstance(item, str) for item in value["required"])
        ):
            raise ValueError("Tool parameter_schema required must be a list of strings")
        return value


class Conversation(BaseModel):
    conversation_id: str
    description: str = ""
    messages: list[Message] = Field(min_length=1)
    tools: list[ToolDefinition] | None = None
    template_options: dict[str, Any] = Field(default_factory=dict)
    expected_behavior: str | None = None


class RuntimeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime: str
    model_path: str | None = None
    source_model_id: str | None = None
    source_revision: str | None = None
    conversion_metadata: dict[str, Any] = Field(default_factory=dict)
    tokenizer_metadata: dict[str, Any] = Field(default_factory=dict)
    template_source: str | None = None
    template_options: dict[str, Any] = Field(default_factory=dict)
    generation_options: dict[str, Any] = Field(default_factory=dict)
    runtime_version: str | None = None


class Prompt(BaseModel):
    source_runtime: str
    template_name: str
    template_source: str | None = None
    template_options: dict[str, Any] = Field(default_factory=dict)
    rendered_text: str
    rendered_utf8_hash: str
    rendered_bytes_length: int
    token_ids: list[int]
    token_count: int
    token_pieces: list[str] = Field(default_factory=list)
    special_token_positions: list[int] = Field(default_factory=list)
    message_roles: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    tokenizer_settings: dict[str, Any] = Field(default_factory=dict)
    provenance: str
    warnings: list[str] = Field(default_factory=list)


class Generation(BaseModel):
    runtime: str
    prompt_id: str
    supplied_prompt_token_ids: list[int]
    effective_generation_settings: dict[str, Any]
    output_text: str
    generated_token_ids: list[int] = Field(default_factory=list)
    generated_token_count: int = 0
    finish_reason: str | None = None
    finish_reason_source: str = "unavailable"
    stop_configuration: dict[str, Any] = Field(default_factory=dict)
    timing_metadata: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"]
    category: str
    evidence: str
    interpretation: str
    suggested_action: str
    confidence: str = "observed"


class Experiment(BaseModel):
    schema_version: str = "1.0"
    experiment_id: str
    mode: str
    synthetic: bool = False
    source_model: dict[str, Any] = Field(default_factory=dict)
    prompts: dict[str, Prompt] = Field(default_factory=dict)
    runtime_configurations: dict[str, RuntimeConfiguration] = Field(default_factory=dict)
    generations: dict[str, Generation] = Field(default_factory=dict)
    token_comparison: dict[str, Any] = Field(default_factory=dict)
    diagnostic_findings: list[Finding] = Field(default_factory=list)
    compatibility: str = "unverified"
    conclusion: str = ""
    uncertainty: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def schema_supported(self):
        if self.schema_version != "1.0":
            raise ValueError("Unsupported experiment schema version")
        return self
