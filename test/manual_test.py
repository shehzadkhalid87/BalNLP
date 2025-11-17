from balnlp.preprocessing.text_clearner import BalochiTextCleaner

# Initialize the cleaner
cleaner = BalochiTextCleaner()

# Example text

text = """
منی نام احمد اِنت۔ 😊
 منی ویب سایٹ: https://example.com
ای میل: user@email.com
12345 کئی لمبر اِنت۔
دشتءِ کتابءَ گسءُ
=-۹۷٦٦ /؛
"""

# Clean the text
cleaned_text = cleaner.clean_text(text)

# Print the cleaned text
print("Original Text:\n", text)
print("\nCleaned Text:\n", cleaned_text)
