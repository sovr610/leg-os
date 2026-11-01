"""A small, strict JSON-Schema subset validator (blueprint §10).

The control plane "rejects unknown fields" and requires "JSON-schema-valid
arguments" for every call (§10). Pulling in a full JSON-Schema implementation
is unnecessary for that contract, and a hand-sized validator is easier to
audit, so this module implements exactly the keywords the action envelope and
tool schemas use:

* ``type`` (single name or list) with a strict distinction between
  ``integer``, ``number`` and ``boolean``: ``True`` is never a number, and
  ``3.0`` is a ``number`` but **not** an ``integer``;
* objects: ``properties``, ``required``, ``additionalProperties`` (``false``
  or a schema), ``propertyNames``, ``minProperties``, ``maxProperties``;
* scalars: ``enum``, ``const``, ``minimum``, ``maximum``,
  ``exclusiveMinimum``, ``exclusiveMaximum``, ``minLength``, ``maxLength``,
  ``pattern`` (``re.search`` semantics, as in JSON Schema);
* arrays: ``items`` (single schema), ``minItems``, ``maxItems``,
  ``uniqueItems``;
* combinators: ``anyOf`` and ``oneOf``. ``oneOf`` detects a *discriminator*
  automatically when every branch is an object schema with a required
  property carrying a ``const`` (e.g. ``"op": {"const": "trim"}``); the
  matching branch is then validated alone, which yields precise errors.

Annotation keywords (``title``, ``description``, ``default``, ``examples``,
``$schema``, ``$id``, ``$comment``) are accepted and ignored. Any other keyword
makes :func:`check_schema` raise :class:`SchemaError`, so a typo in a tool
schema fails at registration time instead of silently accepting everything.

Errors are returned as strings prefixed with a JSON path such as
``$.calls[0].arguments.operations[1].end_ms``.
"""

from __future__ import annotations

import functools
import math
import re
from typing import Any, Mapping

from ..common import canonical_json

__all__ = [
    "SchemaError",
    "SUPPORTED_KEYWORDS",
    "check_schema",
    "validate",
    "is_valid",
    "json_value_errors",
]

_ANNOTATIONS = frozenset({"title", "description", "default", "examples", "$schema", "$id", "$comment"})
SUPPORTED_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "propertyNames",
        "minProperties",
        "maxProperties",
        "enum",
        "const",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "items",
        "minItems",
        "maxItems",
        "uniqueItems",
        "oneOf",
        "anyOf",
    }
) | _ANNOTATIONS

_TYPE_NAMES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})


class SchemaError(ValueError):
    """The *schema itself* is malformed or uses an unsupported keyword."""


# --------------------------------------------------------------------- types
def _type_of(value: Any) -> str:
    """JSON type name of a Python value (``integer`` is reported for ints)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _is_type(value: Any, name: str) -> bool:
    actual = _type_of(value)
    if name == "number":
        return actual in ("integer", "number")
    return actual == name


def _json_equal(a: Any, b: Any) -> bool:
    """Equality that respects the bool/number distinction (``True != 1``)."""
    if _type_of(a) != _type_of(b) and not (_is_type(a, "number") and _is_type(b, "number")):
        return False
    try:
        return canonical_json(a) == canonical_json(b)
    except (TypeError, ValueError):
        return a == b


@functools.lru_cache(maxsize=512)
def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


# ------------------------------------------------------------- schema check
def check_schema(schema: Any, path: str = "#") -> None:
    """Raise :class:`SchemaError` unless ``schema`` uses only supported keywords.

    The check is recursive and also validates keyword value types (e.g.
    ``required`` must be a list of strings, ``pattern`` must compile).
    """
    if isinstance(schema, bool):
        return
    if not isinstance(schema, Mapping):
        raise SchemaError(f"{path}: schema must be an object or boolean")
    unknown = set(schema) - SUPPORTED_KEYWORDS
    if unknown:
        raise SchemaError(f"{path}: unsupported schema keyword(s) {sorted(unknown)}")
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not types or any(t not in _TYPE_NAMES for t in types):
            raise SchemaError(f"{path}.type: invalid type {schema['type']!r}")
    for key in ("properties",):
        if key in schema:
            if not isinstance(schema[key], Mapping):
                raise SchemaError(f"{path}.{key}: must be an object")
            for name, sub in schema[key].items():
                check_schema(sub, f"{path}.{key}.{name}")
    for key in ("additionalProperties", "propertyNames", "items"):
        if key in schema:
            check_schema(schema[key], f"{path}.{key}")
    for key in ("oneOf", "anyOf"):
        if key in schema:
            if not isinstance(schema[key], list) or not schema[key]:
                raise SchemaError(f"{path}.{key}: must be a non-empty list")
            for i, sub in enumerate(schema[key]):
                check_schema(sub, f"{path}.{key}[{i}]")
    if "required" in schema:
        req = schema["required"]
        if not isinstance(req, list) or not all(isinstance(r, str) for r in req):
            raise SchemaError(f"{path}.required: must be a list of strings")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise SchemaError(f"{path}.enum: must be a non-empty list")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema and not _is_type(schema[key], "number"):
            raise SchemaError(f"{path}.{key}: must be a number")
    for key in ("minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"):
        if key in schema and (not _is_type(schema[key], "integer") or schema[key] < 0):
            raise SchemaError(f"{path}.{key}: must be a non-negative integer")
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        raise SchemaError(f"{path}.uniqueItems: must be a boolean")
    if "pattern" in schema:
        if not isinstance(schema["pattern"], str):
            raise SchemaError(f"{path}.pattern: must be a string")
        try:
            _compile(schema["pattern"])
        except re.error as exc:
            raise SchemaError(f"{path}.pattern: invalid regex ({exc})") from None


# -------------------------------------------------------------- JSON values
def json_value_errors(value: Any, path: str = "$") -> list[str]:
    """Errors for values that are not plain JSON (non-str keys, NaN, tuples, objects...).

    Envelopes may arrive as Python dicts rather than JSON text; this guards the
    "JSON-schema-valid arguments" contract against values JSON cannot carry.
    """
    errors: list[str] = []
    if value is None or isinstance(value, (bool, str)):
        return errors
    if isinstance(value, int):
        return errors
    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"{path}: non-finite number is not valid JSON")
        return errors
    if isinstance(value, list):
        for i, item in enumerate(value):
            errors.extend(json_value_errors(item, f"{path}[{i}]"))
        return errors
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                errors.append(f"{path}: object key {k!r} is not a string")
                continue
            errors.extend(json_value_errors(v, _child(path, k)))
        return errors
    errors.append(f"{path}: value of type {type(value).__name__} is not valid JSON")
    return errors


def _child(path: str, key: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


# --------------------------------------------------------------- validation
def validate(instance: Any, schema: Any, path: str = "$") -> list[str]:
    """Validate ``instance`` against ``schema``; return a list of error strings.

    An empty list means the instance is valid. The schema is assumed to have
    passed :func:`check_schema` (call it once when registering a schema).
    """
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: no value is allowed here"]
    errors: list[str] = []

    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(instance, t) for t in types):
            expected = " or ".join(types)
            return [f"{path}: expected {expected}, got {_type_of(instance)}"]

    if "const" in schema and not _json_equal(instance, schema["const"]):
        errors.append(f"{path}: expected constant {schema['const']!r}, got {instance!r}")
    if "enum" in schema and not any(_json_equal(instance, e) for e in schema["enum"]):
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']!r}")

    if _is_type(instance, "number"):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} is less than minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} is greater than maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {instance} must be greater than {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: {instance} must be less than {schema['exclusiveMaximum']}")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: string shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: string longer than maxLength {schema['maxLength']}")
        if "pattern" in schema and not _compile(schema["pattern"]).search(instance):
            errors.append(f"{path}: {instance!r} does not match pattern {schema['pattern']!r}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: array has fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: array has more than {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            seen: set[str] = set()
            for i, item in enumerate(instance):
                key = canonical_json(item) + "|" + _type_of(item)
                if key in seen:
                    errors.append(f"{path}[{i}]: duplicate item")
                seen.add(key)
        if "items" in schema:
            for i, item in enumerate(instance):
                errors.extend(validate(item, schema["items"], f"{path}[{i}]"))

    if isinstance(instance, dict):
        errors.extend(_validate_object(instance, schema, path))

    if "anyOf" in schema:
        branch_errors = [validate(instance, sub, path) for sub in schema["anyOf"]]
        if all(branch_errors):
            errors.append(f"{path}: does not match any allowed schema ({_summarise(branch_errors)})")
    if "oneOf" in schema:
        errors.extend(_validate_one_of(instance, schema["oneOf"], path))
    return errors


def is_valid(instance: Any, schema: Any) -> bool:
    return not validate(instance, schema)


def _validate_object(instance: dict, schema: Mapping[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    props: Mapping[str, Any] = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in instance:
            errors.append(f"{_child(path, name)}: required property is missing")
    if "minProperties" in schema and len(instance) < schema["minProperties"]:
        errors.append(f"{path}: object has fewer than {schema['minProperties']} properties")
    if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
        errors.append(f"{path}: object has more than {schema['maxProperties']} properties")
    additional = schema.get("additionalProperties", True)
    names_schema = schema.get("propertyNames")
    for key in sorted(instance, key=str):
        value = instance[key]
        child = _child(path, str(key))
        if not isinstance(key, str):
            errors.append(f"{path}: object key {key!r} is not a string")
            continue
        if names_schema is not None:
            for err in validate(key, names_schema, f"{child}<name>"):
                errors.append(err)
        if key in props:
            errors.extend(validate(value, props[key], child))
        elif additional is False:
            errors.append(f"{child}: unknown field (additionalProperties is false)")
        elif additional is not True:
            errors.extend(validate(value, additional, child))
    return errors


def _discriminator(branches: list[Any]) -> str | None:
    """Name of a property that every branch requires and pins with ``const``."""
    candidates: set[str] | None = None
    for b in branches:
        if not isinstance(b, Mapping):
            return None
        props = b.get("properties", {})
        names = {
            k for k in b.get("required", []) if isinstance(props.get(k), Mapping) and "const" in props[k]
        }
        candidates = names if candidates is None else candidates & names
        if not candidates:
            return None
    return sorted(candidates)[0] if candidates else None


def _validate_one_of(instance: Any, branches: list[Any], path: str) -> list[str]:
    disc = _discriminator(branches)
    if disc is not None and isinstance(instance, dict):
        if disc not in instance:
            return [f"{_child(path, disc)}: required discriminator is missing"]
        tags = [b["properties"][disc]["const"] for b in branches]
        for tag, branch in zip(tags, branches):
            if _json_equal(instance[disc], tag):
                return validate(instance, branch, path)
        return [f"{_child(path, disc)}: {instance[disc]!r} is not one of {tags!r}"]
    branch_errors = [validate(instance, sub, path) for sub in branches]
    matches = sum(1 for e in branch_errors if not e)
    if matches == 1:
        return []
    if matches == 0:
        return [f"{path}: does not match any oneOf branch ({_summarise(branch_errors)})"]
    return [f"{path}: matches {matches} oneOf branches; exactly one is required"]


def _summarise(branch_errors: list[list[str]]) -> str:
    return "; ".join(f"branch {i}: {errs[0]}" for i, errs in enumerate(branch_errors) if errs)
