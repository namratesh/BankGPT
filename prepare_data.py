import os
import json
import re
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf


class PDFTextProcessor:
    """
    A class to extract, clean, and save structured content from PDF files into JSON format.

    Attributes:
        dataset_dir (str): Directory containing PDF files.
        year (int): Year to associate with each document.
        output_dir (str): Directory to store output JSON files.
        strategy (str): Extraction strategy for `partition_pdf`.
        infer_table_structure (bool): Whether to infer table structure in PDFs.
    """

    def __init__(self, dataset_dir: str, year: int = 2024):
        """
        Initializes the processor by loading environment variables and setting up paths.
        """
        load_dotenv()
        self.dataset_dir = dataset_dir
        self.year = year
        self.output_dir = os.getenv("output_dir", "dataset/json")
        self.strategy = os.getenv("strategy", "fast")
        self.infer_table_structure = os.getenv("infer_table_structure", "false").lower() == "true"
        os.makedirs(self.output_dir, exist_ok=True)

    def get_pdf_paths(self) -> List[str]:
        """
        Retrieves all PDF file paths from the dataset directory.
        """
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Directory not found: {self.dataset_dir}")

        pdfs = [
            os.path.join(self.dataset_dir, f)
            for f in os.listdir(self.dataset_dir)
            if f.lower().endswith(".pdf")
        ]

        if not pdfs:
            print("⚠️ No PDF files found in the directory.")

        return sorted(pdfs)

    def extract_pdf_text_by_page(self, pdf_path: str) -> Dict[int, str]:
        """
        Extracts text from a PDF file, grouped by page.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        try:
            elements = partition_pdf(
                filename=pdf_path,
                strategy=self.strategy,
                infer_table_structure=self.infer_table_structure
            )

            if not elements:
                raise ValueError(f"No extractable content found in {pdf_path}")

            pagewise_text = {}
            for el in elements:
                page_num = el.metadata.page_number or 0
                if page_num not in pagewise_text:
                    pagewise_text[page_num] = []

                if el.text:
                    clean_text = el.text.strip()
                    if clean_text:
                        pagewise_text[page_num].append(clean_text)

            return {
                page: "\n".join(lines)
                for page, lines in sorted(pagewise_text.items())
            }

        except Exception as e:
            raise Exception(f"Failed to extract text from {pdf_path}: {str(e)}")

    def clean_pdf_json_content(self, data: list) -> list:
        """
        Cleans the 'content' field in a list of dictionaries extracted from PDFs
        and adds a new key 'clean_content' with the cleaned version.

        Cleaning operations:
        - Remove hyphenated line breaks
        - Collapse multiple spaces
        - Remove table borders
        - Remove page numbers
        - Normalize whitespace

        Args:
            data (list): List of dicts with a 'content' field.

        Returns:
            list: List with 'clean_content' added per item.
        """
        def clean_text(text: str) -> str:
            text = re.sub(r'-\n(\w+)', r'\1', text)
            text = re.sub(r'[ ]{2,}', ' ', text)
            text = re.sub(r'[─═╚╩╝╔╦╗╠╣╬]+', '', text)
            text = re.sub(r'^\s*(Page|PAGE)?\s*\d+\s*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'\n{2,}', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        for idx, item in enumerate(data):
            try:
                if 'content' in item and isinstance(item['content'], str):
                    item['clean_content'] = clean_text(item['content'])
                else:
                    print(f"[WARN] Skipping index {idx}: Missing or non-string 'content'")
            except Exception as e:
                print(f"[ERROR] Failed to process index {idx}: {e}")

        return data

    def create_json(self, pdf_path: str, company: str, data: Dict[int, str]) -> str:
        """
        Creates and saves a JSON file with structured (and cleaned) content per page.
        """
        if not isinstance(data, dict):
            raise ValueError("Expected `data` to be a dictionary of page_num -> text")

        try:
            final_text = [
                {
                    "page_num": page_num,
                    "content": text,
                    "year": self.year,
                    "company": company
                }
                for page_num, text in data.items()
            ]

            final_text = self.clean_pdf_json_content(final_text)

            output_path = os.path.join(
                self.output_dir,
                f"{os.path.basename(pdf_path).split('.')[0]}_{self.year}.json"
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_text, f, indent=4, ensure_ascii=False)

            print(f"✅ JSON saved to: {output_path}")
            return output_path

        except Exception as e:
            raise Exception(f"Failed to write JSON for {pdf_path}: {str(e)}")

    def process_all(self):
        """
        Executes the entire flow: PDF reading → text extraction → cleaning → JSON output.
        """
        pdf_paths = self.get_pdf_paths()

        for path in tqdm(pdf_paths, desc="Processing PDFs"):
            company_name = os.path.basename(path).split(".")[0]
            data = self.extract_pdf_text_by_page(path)
            self.create_json(path, company_name, data)


if __name__ == "__main__":
    processor = PDFTextProcessor(dataset_dir="dataset/pdfs", year=2024)
    processor.process_all()
