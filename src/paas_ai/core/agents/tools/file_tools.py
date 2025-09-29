"""
File operation tools for agents to read and write files.
"""

import os
from pathlib import Path
from typing import Optional

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from paas_ai.utils.logging import get_logger

logger = get_logger("paas_ai.agents.tools.file_tools")


class WriteFileInput(BaseModel):
    """Input schema for file writing tool."""

    filename: str = Field(description="The name/path of the file to write")
    content: str = Field(description="The content to write to the file")
    directory: str = Field(
        default=".", description="Directory to write the file in (default: current directory)"
    )


class ReadFileInput(BaseModel):
    """Input schema for file reading tool."""

    filename: str = Field(description="The name/path of the file to read")
    directory: str = Field(
        default=".", description="Directory to read the file from (default: current directory)"
    )


class WriteFileTool(BaseTool):
    """Tool for writing content to a file."""

    name: str = "write_file"
    description: str = """
    Write content to a file. Use this to create YAML manifests, configuration files, or any other text files.
    The tool will create the directory structure if it doesn't exist.
    Provide the filename, content, and optionally a directory path.
    """
    args_schema: type = WriteFileInput

    def _run(
        self,
        filename: str,
        content: str,
        directory: str = ".",
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs,
    ) -> str:
        """Write content to a file."""
        try:
            # Create the full path
            dir_path = Path(directory)
            file_path = dir_path / filename

            # Create directory if it doesn't exist
            dir_path.mkdir(parents=True, exist_ok=True)

            # Write the content to the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Successfully wrote file: {file_path}")
            return f"Successfully wrote file: {file_path} ({len(content)} characters)"

        except Exception as e:
            error_msg = f"Error writing file {filename}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _arun(self, filename: str, content: str, directory: str = ".", **kwargs) -> str:
        """Async version of file writing."""
        return self._run(filename, content, directory, **kwargs)


class ReadFileTool(BaseTool):
    """Tool for reading content from a file."""

    name: str = "read_file"
    description: str = """
    Read content from a file. Use this to read existing configuration files, templates, or any text files.
    Provide the filename and optionally a directory path.
    """
    args_schema: type = ReadFileInput

    def _run(
        self,
        filename: str,
        directory: str = ".",
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs,
    ) -> str:
        """Read content from a file."""
        try:
            # Create the full path
            dir_path = Path(directory)
            file_path = dir_path / filename

            # Check if file exists
            if not file_path.exists():
                return f"File not found: {file_path}"

            # Read the content from the file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            logger.info(f"Successfully read file: {file_path} ({len(content)} characters)")
            return content

        except Exception as e:
            error_msg = f"Error reading file {filename}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _arun(self, filename: str, directory: str = ".", **kwargs) -> str:
        """Async version of file reading."""
        return self._run(filename, directory, **kwargs)
