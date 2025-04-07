import os
import json
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

class PDFTextProcessor:
    """
    A class to extract text from PDF files and save structured content as JSON.

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

        Args:
            dataset_dir (str): Directory containing the input PDF files.
            year (int): The year associated with the documents.
            output_dir (str): Directory to save the resulting JSON files.
        """
        load_dotenv()
        self.dataset_dir = dataset_dir
        self.year = year
        self.output_dir = os.getenv("output_dir", "dataset/json")
        self.strategy = os.getenv("strategy", "fast")  # Fallback to 'fast' if not set
        self.infer_table_structure = os.getenv("infer_table_structure", "false").lower() == "true"
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists

    def get_pdf_paths(self) -> List[str]:
        """
        Retrieves all PDF file paths from the dataset directory.

        Returns:
            List[str]: List of full paths to PDF files.

        Raises:
            FileNotFoundError: If the dataset directory does not exist.
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

        Args:
            pdf_path (str): Full path to the PDF file.

        Returns:
            Dict[int, str]: Dictionary mapping page numbers to text content.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If no extractable content is found.
            Exception: For general extraction failures.
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

            # Organize text content by page number
            for el in elements:
                page_num = el.metadata.page_number or 0
                if page_num not in pagewise_text:
                    pagewise_text[page_num] = []

                if el.text:
                    clean_text = el.text.strip()
                    if clean_text:  # Avoid empty strings
                        pagewise_text[page_num].append(clean_text)

            # Join lines into one string per page
            return {
                page: "\n".join(lines)
                for page, lines in sorted(pagewise_text.items())
            }

        except Exception as e:
            raise Exception(f"Failed to extract text from {pdf_path}: {str(e)}")

    def create_json(self, pdf_path: str, company: str, data: Dict[int, str]) -> str:
        """
        Creates a JSON file containing structured text per page for a given PDF.

        Args:
            pdf_path (str): Path to the original PDF file.
            company (str): Name of the company/document.
            data (Dict[int, str]): Extracted page-wise text.

        Returns:
            str: Path to the saved JSON file.

        Raises:
            ValueError: If data is not in the expected format.
            Exception: If JSON writing fails.
        """
        if not isinstance(data, dict):
            raise ValueError("Expected `data` to be a dictionary of page_num -> text")

        try:
            # Structure the JSON data
            final_text = [
                {
                    "page_num": page_num,
                    "content": text,
                    "year": self.year,
                    "company": company
                }
                for page_num, text in data.items()
            ]

            output_path = os.path.join(
                self.output_dir,
                f"{os.path.basename(pdf_path).split('.')[0]}_{self.year}.json"
            )

            # Write the structured content to a JSON file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_text, f, indent=4, ensure_ascii=False)

            print(f"✅ JSON saved to: {output_path}")
            return output_path

        except Exception as e:
            raise Exception(f"Failed to write JSON for {pdf_path}: {str(e)}")

    def process_all(self):
        """
        Orchestrates the end-to-end process of loading PDFs, extracting content,
        and saving structured output to JSON files.
        """
        pdf_paths = self.get_pdf_paths()

        for path in tqdm(pdf_paths, desc="Processing PDFs"):
            company_name = os.path.basename(path).split(".")[0]
            data = self.extract_pdf_text_by_page(path)
            self.create_json(path, company_name, data)

if __name__ == "__main__":
    processor = PDFTextProcessor(dataset_dir="dataset/pdfs", year=2024)
    processor.process_all()
