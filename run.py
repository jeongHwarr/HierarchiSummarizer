import os
import shutil
from pathlib import Path

import yaml

from modules.pdf_to_md import MistralOCRProcessor
from modules.section_parser import SectionParser
from modules.section_summarizer import SectionSummarizer


def process_all_documents(input_folder, output_folder, done_folder, config):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(done_folder, exist_ok=True)

    files_to_process = [
        f for f in os.listdir(input_folder) if f.endswith(".pdf") or f.endswith(".md")
    ]

    for filename in files_to_process:
        file_path = os.path.join(input_folder, filename)
        process_document(file_path, output_folder, done_folder, config)


def process_document(file_path, output_folder, done_folder, config):
    filename = Path(file_path).stem

    if file_path.endswith(".pdf"):
        output_path = os.path.join(output_folder, filename)
        md_file_name = "pdf_to_md.md"
        md_path = os.path.join(output_path, md_file_name)
        pdf_converter = MistralOCRProcessor(
            api_key=config["mistral"]["api_key"], output_dir=output_path
        )
        pdf_converter.process_pdf(file_path, md_file_name=md_file_name)

    else:
        output_path = output_folder
        md_path = file_path

    section_parser = SectionParser(md_path)
    parsed_sections = section_parser.parse()

    summary_md_name = filename + "_summary.md"
    md_output_path = os.path.join(output_path, summary_md_name)
    summarizer = SectionSummarizer(config=config)
    summarizer.summarize_all_sections_and_save_result(parsed_sections, md_output_path)

    shutil.move(file_path, os.path.join(done_folder, os.path.basename(file_path)))


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    input_folder = config["dir_path"]["input_dir_path"]
    output_folder = config["dir_path"]["output_dir_path"]
    done_folder = config["dir_path"]["done_dir_path"]

    process_all_documents(input_folder, output_folder, done_folder, config)


if __name__ == "__main__":
    main()
