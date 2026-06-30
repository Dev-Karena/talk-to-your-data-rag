import pytest
from pathlib import Path
from scripts.query_explainer import find_matched_case

def test_find_matched_case_nonexistent():
    assert find_matched_case("some query", Path("nonexistent.yaml")) is None

def test_find_matched_case_exists(tmp_path):
    # Create a dummy benchmark file
    yaml_content = """
corpus:
  description: "Test corpus"
  expected_doc_hashes: ["hash1"]
cases:
  - id: db-03
    query: "How does a B-tree index speed up queries?"
    type: single
    expected_doc_hashes: ["hash1"]
  - id: other-case
    query: "Compare virtual memory and normal databases"
    type: cross_document
    expected_doc_hashes: ["hash1", "hash2"]
"""
    benchmark_file = tmp_path / "test_cases.yaml"
    benchmark_file.write_text(yaml_content, encoding="utf-8")
    
    # 1. Exact match
    c1 = find_matched_case("How does a B-tree index speed up queries?", benchmark_file)
    assert c1 is not None
    assert c1.id == "db-03"
    
    # 2. Case-insensitive case ID match
    c2 = find_matched_case("db-03", benchmark_file)
    assert c2 is not None
    assert c2.id == "db-03"
    
    # 3. Case-insensitive query match
    c3 = find_matched_case("how does a b-tree index speed up queries?", benchmark_file)
    assert c3 is not None
    assert c3.id == "db-03"
    
    # 4. Miss
    c4 = find_matched_case("unknown query", benchmark_file)
    assert c4 is None
