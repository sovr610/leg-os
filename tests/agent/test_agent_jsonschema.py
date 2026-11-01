"""jsonschema_lite: the strict schema subset behind "JSON-schema-valid arguments" (§10)."""

from __future__ import annotations

import math

import pytest

from m63_omni.agent.jsonschema_lite import SchemaError, check_schema, is_valid, json_value_errors, validate


def test_integer_number_boolean_are_distinct() -> None:
    assert is_valid(3, {"type": "integer"})
    assert not is_valid(3.0, {"type": "integer"})
    assert not is_valid(True, {"type": "integer"})
    assert is_valid(3, {"type": "number"}) and is_valid(2.5, {"type": "number"})
    assert not is_valid(False, {"type": "number"})
    assert is_valid(True, {"type": "boolean"}) and not is_valid(1, {"type": "boolean"})
    assert is_valid(None, {"type": ["string", "null"]})
    # enum/const respect the bool/number distinction
    assert not is_valid(True, {"enum": [1]})
    assert not is_valid(1, {"const": True})


def test_object_keywords_and_paths() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["a", "b"],
        "properties": {
            "a": {"type": "integer", "minimum": 0, "maximum": 10},
            "b": {"type": "array", "minItems": 1, "maxItems": 2, "uniqueItems": True, "items": {"type": "string", "pattern": "^x"}},
        },
    }
    check_schema(schema)
    assert validate({"a": 3, "b": ["x1"]}, schema) == []
    errs = validate({"a": 11, "b": ["y", "y", "x"], "c": 1}, schema)
    joined = "\n".join(errs)
    assert "$.a: 11 is greater than maximum 10" in joined
    assert "$.b: array has more than 2 items" in joined
    assert "$.b[0]" in joined and "does not match pattern" in joined
    assert "$.b[1]: duplicate item" in joined
    assert "$.c: unknown field" in joined
    assert validate({"a": 1}, schema) == ["$.b: required property is missing"]


def test_one_of_discriminator_gives_precise_errors() -> None:
    schema = {
        "oneOf": [
            {"type": "object", "additionalProperties": False, "required": ["op", "n"],
             "properties": {"op": {"const": "inc"}, "n": {"type": "integer"}}},
            {"type": "object", "additionalProperties": False, "required": ["op"], "properties": {"op": {"const": "reset"}}},
        ]
    }
    check_schema(schema)
    assert is_valid({"op": "inc", "n": 2}, schema)
    assert validate({"op": "inc", "n": "2"}, schema) == ["$.n: expected integer, got string"]
    assert "is not one of ['inc', 'reset']" in validate({"op": "boom"}, schema)[0]
    assert "discriminator is missing" in validate({"n": 1}, schema)[0]
    # non-discriminated oneOf: exactly one branch must match
    ambiguous = {"oneOf": [{"type": "number"}, {"type": "integer"}]}
    assert is_valid(2.5, ambiguous)
    assert "matches 2 oneOf branches" in validate(2, ambiguous)[0]


def test_check_schema_rejects_unknown_keywords_and_bad_values() -> None:
    with pytest.raises(SchemaError):
        check_schema({"type": "object", "additionalProperites": False})  # typo must not silently pass
    with pytest.raises(SchemaError):
        check_schema({"type": "strnig"})
    with pytest.raises(SchemaError):
        check_schema({"pattern": "("})
    with pytest.raises(SchemaError):
        check_schema({"minItems": -1})


def test_json_value_errors() -> None:
    assert json_value_errors({"a": [1, "x", None, True, 2.5]}) == []
    errs = json_value_errors({"a": math.nan, 1: "x", "t": (1, 2), "o": object()})
    assert any("non-finite" in e for e in errs)
    assert any("not a string" in e for e in errs)
    assert any("tuple" in e for e in errs)
    assert any("object" in e for e in errs)
