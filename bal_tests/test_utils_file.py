
from balnlp.utils.utils_file import process_large_file, save_processed_text

def test_process_large_file(tmp_path):
    # Create a sample text file
    file_path = tmp_path / "sample.txt"
    content = "Hello Balochi\nThis is a test.\n" * 10
    file_path.write_text(content, encoding="utf-8")

    # Test processing without a processor
    chunks = list(process_large_file(str(file_path), chunk_size=10, show_progress=False))
    assert "".join(chunks) == content

    # Test with a processor
    chunks = list(
        process_large_file(str(file_path), chunk_size=10, processor=lambda x: x.upper(), show_progress=False)
    )
    assert "".join(chunks) == content.upper()


def test_save_processed_text(tmp_path):
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    content = "Hello Balochi\nTest data."
    input_file.write_text(content, encoding="utf-8")

    save_processed_text(
        str(input_file),
        str(output_file),
        processor=lambda x: x.replace("Balochi", "BLC"),
        chunk_size=5,
        show_progress=False,
    )

    saved_content = output_file.read_text(encoding="utf-8")
    assert "BLC" in saved_content
    assert "Balochi" not in saved_content
