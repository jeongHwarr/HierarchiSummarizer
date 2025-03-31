import base64
import json
import os
from pathlib import Path

from mistralai import DocumentURLChunk, Mistral


class MistralOCRProcessor:
    def __init__(self, api_key: str, output_dir: str):
        self.client = Mistral(api_key=api_key)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def replace_images_in_markdown(self, markdown_str: str, images_dict: dict) -> str:
        """
        Converts base64 encoded images in markdown into external links for better readability.
        """
        for img_name, base64_str in images_dict.items():
            markdown_str = markdown_str.replace(
                f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
            )
        return markdown_str

    def get_combined_markdown(self, ocr_response) -> str:
        """
        Combines markdown content from all pages of the OCR response into a single markdown string.
        """
        markdowns = []
        for page in ocr_response.pages:
            image_data = {img.id: img.image_base64 for img in page.images}
            markdowns.append(self.replace_images_in_markdown(page.markdown, image_data))
        return "\n\n".join(markdowns)

    def process_pdf(self, pdf_path: str, md_file_name: str) -> str:
        pdf_path = Path(pdf_path)
        pdf_base = pdf_path.stem

        image_dir = os.path.join(self.output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        # Step 1: Upload PDF
        uploaded_file = self.upload_pdf(pdf_path)

        # Step 2: Process OCR
        ocr_response = self.process_ocr(uploaded_file)

        # Step 3: Save OCR response
        self.save_ocr_response(ocr_response)

        # Step 4: Extract images
        image_map = self.extract_images(ocr_response, image_dir, pdf_base)

        # Step 5: Generate markdown
        final_markdown = self.generate_markdown(ocr_response, image_map)
        output_markdown_path = os.path.join(self.output_dir, md_file_name)

        with open(output_markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(final_markdown)

    def upload_pdf(self, pdf_path: Path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        uploaded_file = self.client.files.upload(
            file={"file_name": pdf_path.name, "content": pdf_bytes}, purpose="ocr"
        )

        return uploaded_file

    def process_ocr(self, uploaded_file):
        signed_url = self.client.files.get_signed_url(
            file_id=uploaded_file.id, expiry=1
        )
        ocr_response = self.client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True,
        )
        return ocr_response

    def save_ocr_response(self, ocr_response):
        ocr_json_path = os.path.join(self.output_dir, "ocr_response.json")
        with open(ocr_json_path, "w", encoding="utf-8") as json_file:
            json.dump(ocr_response.dict(), json_file, indent=4, ensure_ascii=False)

    def extract_images(self, ocr_response, image_dir: str, pdf_base: str):
        global_counter = 1
        image_map = {}

        for page in ocr_response.pages:
            for image_obj in page.images:
                base64_str = (
                    image_obj.image_base64.split(",", 1)[1]
                    if image_obj.image_base64.startswith("data:")
                    else image_obj.image_base64
                )
                image_bytes = base64.b64decode(base64_str)

                ext = Path(image_obj.id).suffix if Path(image_obj.id).suffix else ".png"
                new_image_name = f"img_{global_counter}{ext}"
                global_counter += 1

                image_output_path = os.path.join(image_dir, new_image_name)
                with open(image_output_path, "wb") as f:
                    f.write(image_bytes)

                image_map[image_obj.id] = image_output_path

        return image_map

    def generate_markdown(self, ocr_response, image_map):
        updated_markdown_pages = []

        for page in ocr_response.pages:
            updated_markdown = page.markdown
            for img_id, image_path in image_map.items():
                image_name = os.path.basename(image_path)
                image_path = Path(image_path)
                relative_path = image_path.relative_to(image_path.parent.parent)
                updated_markdown = updated_markdown.replace(
                    f"![{img_id}]({img_id})", f"![{image_name}]({relative_path})"
                )
            updated_markdown_pages.append(updated_markdown)

        return "\n\n".join(updated_markdown_pages)
