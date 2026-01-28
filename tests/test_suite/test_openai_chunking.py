"""Tests for OpenAI chunking functionality."""

import json
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from soda_curation.pipeline.openai_utils import (
    chunk_file_list,
    count_messages_tokens,
    count_tokens,
    create_chunked_messages,
    get_token_limit,
    merge_pydantic_responses,
)


class MockAsignedFiles(BaseModel):
    """Mock model for assigned files."""

    panel_label: str
    panel_sd_files: List[str]


class MockAsignedFilesList(BaseModel):
    """Mock model for assigned files list."""

    assigned_files: List[MockAsignedFiles]
    not_assigned_files: List[str]


def test_count_tokens():
    """Test token counting functionality."""
    # Test with a simple string
    text = "Hello, this is a test string."
    tokens = count_tokens(text)
    assert isinstance(tokens, int)
    assert tokens > 0


def test_get_token_limit():
    """Test getting token limits for different models."""
    # Test known models
    assert get_token_limit("gpt-4o") == 120000
    assert get_token_limit("gpt-5") == 250000

    # Test unknown model (should return default)
    assert get_token_limit("unknown-model") == 120000


def test_count_messages_tokens():
    """Test counting tokens in message lists."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    token_count = count_messages_tokens(messages)
    assert isinstance(token_count, int)
    assert token_count > 0


def test_chunk_file_list():
    """Test chunking file lists."""
    # Create a file list with many files
    files = [f"file_{i}.txt" for i in range(1000)]
    file_list_str = "\n".join(files)

    # Chunk with a small limit
    chunks = chunk_file_list(file_list_str, chunk_size=1000, model="gpt-4o")

    # Should create multiple chunks
    assert len(chunks) > 1

    # Each chunk should be a string
    for chunk in chunks:
        assert isinstance(chunk, str)

    # All files should be present in some chunk
    all_files_in_chunks = "\n".join(chunks).split("\n")
    assert set(files) == set(all_files_in_chunks)


def test_chunk_file_list_single_large_file():
    """Test chunking when a single file exceeds chunk size."""
    # Create a very long filename
    large_file = "very_long_filename_" + ("x" * 10000) + ".txt"
    file_list_str = "\n".join([large_file, "normal_file.txt"])

    chunks = chunk_file_list(file_list_str, chunk_size=100, model="gpt-4o")

    # Should still create chunks, even with oversized file
    assert len(chunks) >= 1
    assert (
        large_file in chunks[0] or large_file in chunks[1] if len(chunks) > 1 else True
    )


def test_create_chunked_messages():
    """Test creating chunked messages."""
    # Create messages with a large file list
    file_list = "\n".join([f"source_data/file_{i}.txt" for i in range(1000)])

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"Please analyze these figures.\n\nFile list:\n{file_list}",
        },
    ]

    # Create chunks with a small token limit
    chunked_messages = create_chunked_messages(
        messages, model="gpt-4o", token_limit=5000
    )

    # Should create multiple message lists
    assert len(chunked_messages) > 1

    # Each should have the same structure
    for msg_list in chunked_messages:
        assert len(msg_list) == 2
        assert msg_list[0]["role"] == "system"
        assert msg_list[1]["role"] == "user"
        assert "File list:" in msg_list[1]["content"]


def test_create_chunked_messages_no_chunking_needed():
    """Test that small messages are not chunked."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "This is a small message."},
    ]

    # Use a large token limit
    chunked_messages = create_chunked_messages(
        messages, model="gpt-4o", token_limit=100000
    )

    # Should return single message list
    assert len(chunked_messages) == 1
    assert chunked_messages[0] == messages


def test_merge_pydantic_responses():
    """Test merging Pydantic responses."""
    # Create mock responses
    response1 = MagicMock()
    response1.choices = [MagicMock()]
    response1.choices[0].message = MagicMock()
    response1.choices[0].message.parsed = MockAsignedFilesList(
        assigned_files=[
            MockAsignedFiles(panel_label="A", panel_sd_files=["file1.txt", "file2.txt"])
        ],
        not_assigned_files=["file3.txt"],
    )
    response1.usage = MagicMock()
    response1.usage.prompt_tokens = 1000
    response1.usage.completion_tokens = 500
    response1.usage.total_tokens = 1500

    response2 = MagicMock()
    response2.choices = [MagicMock()]
    response2.choices[0].message = MagicMock()
    response2.choices[0].message.parsed = MockAsignedFilesList(
        assigned_files=[
            MockAsignedFiles(panel_label="B", panel_sd_files=["file4.txt"])
        ],
        not_assigned_files=["file5.txt"],
    )
    response2.usage = MagicMock()
    response2.usage.prompt_tokens = 800
    response2.usage.completion_tokens = 400
    response2.usage.total_tokens = 1200

    # Merge responses
    merged = merge_pydantic_responses([response1, response2], MockAsignedFilesList)

    # Check merged content
    assert hasattr(merged.choices[0].message, "parsed")
    parsed = merged.choices[0].message.parsed

    assert len(parsed.assigned_files) == 2
    assert len(parsed.not_assigned_files) == 2

    # Check token usage is summed
    assert merged.usage.prompt_tokens == 1800
    assert merged.usage.completion_tokens == 900
    assert merged.usage.total_tokens == 2700


def test_merge_pydantic_responses_single_response():
    """Test merging with a single response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.parsed = MockAsignedFilesList(
        assigned_files=[
            MockAsignedFiles(panel_label="A", panel_sd_files=["file1.txt"])
        ],
        not_assigned_files=["file2.txt"],
    )

    # Merge single response
    merged = merge_pydantic_responses([response], MockAsignedFilesList)

    # Should return the same response
    assert merged == response


@patch("soda_curation.pipeline.openai_utils.tiktoken")
def test_count_tokens_without_tiktoken(_mock_tiktoken):
    """Test token counting fallback when tiktoken is not available."""
    # Import again to trigger fallback
    from soda_curation.pipeline import openai_utils

    # Temporarily set tiktoken to None
    original_tiktoken = openai_utils.tiktoken
    openai_utils.tiktoken = None

    try:
        text = "Hello, this is a test."
        tokens = openai_utils.count_tokens(text)

        # Should use fallback estimation (1 token ≈ 4 chars)
        expected_tokens = len(text) // 4
        assert tokens == expected_tokens
    finally:
        # Restore original tiktoken
        openai_utils.tiktoken = original_tiktoken


def test_chunk_file_list_empty():
    """Test chunking an empty file list."""
    chunks = chunk_file_list("", chunk_size=1000, model="gpt-4o")
    assert len(chunks) == 1
    assert chunks[0] == ""


def test_create_chunked_messages_without_file_list_marker():
    """Test chunking when file list marker is not found."""
    # Create messages without explicit file list marker
    content = "\n".join(
        [
            "Please analyze these files:",
            "file1.txt",
            "file2.txt",
        ]
        + [f"file{i}.txt" for i in range(3, 1000)]
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]

    # Create chunks with a small token limit
    chunked_messages = create_chunked_messages(
        messages, model="gpt-4o", token_limit=3000
    )

    # Should still create chunks (may use generic line splitting)
    assert len(chunked_messages) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
