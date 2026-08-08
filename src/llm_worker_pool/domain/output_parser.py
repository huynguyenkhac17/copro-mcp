"""Parse/repair output JSON returned from local model.

9B models cannot guarantee valid JSON returns even when 'format=json' is enforced.
The strategy here is graceful degrade: never raise a hard error when parsing fails -
only return a warning for the use case to decide (Claude at the top level
will ultimately be the one to review/reject this result).
"""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as jsonschema_validate
from pydantic import BaseModel

_JSON_OBJECT_OR_ARRAY_RE = re.compile(r"[{\[].*[}\]]", re.DOTALL)
_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class ParsedOutput(BaseModel):
    content: Any = None
    warnings: list[str] = []


class OutputParser:
    def parse(self, raw_text: str, schema: dict | None = None) -> ParsedOutput:
        warnings: list[str] = []
        data = self._try_parse(raw_text)

        if data is None:
            fenced = _MARKDOWN_FENCE_RE.search(raw_text)
            if fenced:
                data = self._try_parse(fenced.group(1))
                if data is not None:
                    warnings.append("json_extracted_from_markdown_fence")

        if data is None:
            match = _JSON_OBJECT_OR_ARRAY_RE.search(raw_text)
            if match:
                data = self._try_parse(match.group(0))
                if data is not None:
                    warnings.append("json_extracted_from_surrounding_text")

        if data is None:
            warnings.append("json_parse_failed")
            return ParsedOutput(content=None, warnings=warnings)

        if schema is not None:
            try:
                jsonschema_validate(instance=data, schema=schema)
            except JsonSchemaValidationError as exc:
                warnings.append(f"json_schema_validation_failed: {exc.message}")

        return ParsedOutput(content=data, warnings=warnings)

    @staticmethod
    def _try_parse(text: str) -> Any | None:
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, AttributeError):
            return None
