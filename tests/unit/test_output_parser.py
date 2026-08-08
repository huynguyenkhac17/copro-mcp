from local_ai_gateway.domain.output_parser import OutputParser


def test_parse_plain_json():
    parser = OutputParser()
    result = parser.parse('{"summary": "ok", "confidence": 0.9}')

    assert result.content == {"summary": "ok", "confidence": 0.9}
    assert result.warnings == []


def test_parse_json_inside_markdown_fence():
    parser = OutputParser()
    raw = 'This is the result:\n```json\n{"summary": "ok"}\n```\nThank you.'
    result = parser.parse(raw)

    assert result.content == {"summary": "ok"}
    assert "json_extracted_from_markdown_fence" in result.warnings


def test_parse_json_embedded_in_surrounding_text():
    parser = OutputParser()
    raw = 'It is definitely {"summary": "ok"} my friend.'
    result = parser.parse(raw)

    assert result.content == {"summary": "ok"}
    assert "json_extracted_from_surrounding_text" in result.warnings


def test_parse_failure_returns_none_with_warning():
    parser = OutputParser()
    result = parser.parse("this is not JSON at all")

    assert result.content is None
    assert "json_parse_failed" in result.warnings


def test_schema_validation_failure_adds_warning_but_keeps_content():
    parser = OutputParser()
    schema = {
        "type": "object",
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}},
    }
    result = parser.parse('{"other": 1}', schema=schema)

    assert result.content == {"other": 1}
    assert any(w.startswith("json_schema_validation_failed") for w in result.warnings)


def test_schema_validation_success_no_warning():
    parser = OutputParser()
    schema = {
        "type": "object",
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}},
    }
    result = parser.parse('{"summary": "ok"}', schema=schema)

    assert result.content == {"summary": "ok"}
    assert result.warnings == []
