"""Utility functions for processing and saving large text files."""

import os
from typing import Callable, Iterator, Optional

from tqdm import tqdm


def process_large_file(
    file_path: str,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
    processor: Optional[Callable[[str], str]] = None,
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> Iterator[str]:
    """
    Process a large text file in chunks to avoid memory issues.

    Args:
        file_path (str): Path to the text file.
        chunk_size (int): Size of chunks to read (in bytes).
        processor (Callable[[str], str], optional): Function to process chunks.
        encoding (str): File encoding (default: 'utf-8').
        show_progress (bool): Whether to show a progress bar.

    Yields:
        str: Processed text chunks.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    file_size = os.path.getsize(file_path)

    with open(file_path, "r", encoding=encoding) as file:
        with tqdm(
            total=file_size, disable=not show_progress, unit="B", unit_scale=True
        ) as pbar:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break

                if processor:
                    chunk = processor(chunk)

                pbar.update(len(chunk.encode(encoding)))
                yield chunk


def save_processed_text(
    input_path: str,
    output_path: str,
    processor: Callable[[str], str],
    chunk_size: int = 1024 * 1024,
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> None:
    """
    Process a large text file and save results to another file.

    Args:
        input_path (str): Path to the input file.
        output_path (str): Path to save the processed text.
        processor (Callable[[str], str]): Function to process each chunk.
        chunk_size (int): Size of chunks to read.
        encoding (str): File encoding.
        show_progress (bool): Whether to show a progress bar.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding=encoding) as out_file:
        for chunk in process_large_file(
            input_path,
            chunk_size=chunk_size,
            processor=processor,
            encoding=encoding,
            show_progress=show_progress,
        ):
            out_file.write(chunk)


def read_balochi_file(file_path: str, encoding: str = "utf-8") -> list:
    """
    Read a Balochi text file and return lines.

    Args:
        file_path: Path to the text file
        encoding: File encoding (default: 'utf-8')

    Returns:
        List of text lines
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except UnicodeDecodeError:
        raise UnicodeDecodeError(
            "utf-8", b"", 0, 1, f"Unable to decode {file_path} with {encoding} encoding"
        )
    except Exception:
        raise Exception(f"Unable to decode {file_path} with {encoding} encoding")


def write_balochi_file(file_path: str, data: list, encoding: str = "utf-8") -> None:
    """
    Write data to a Balochi text file.

    Args:
        file_path: Path to output file
        data: List of text lines to write
        encoding: File encoding (default: 'utf-8')
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding=encoding) as f:
        for line in data:
            f.write(line + "\n")
