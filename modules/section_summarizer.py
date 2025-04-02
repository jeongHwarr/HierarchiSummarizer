import os

from utils.llm import Groq, Mistral, Ollama, OpenAI


class SectionSummarizer:
    def __init__(self, config):
        self.config = config

        self.processing_system_prompt()
        self.llm = self.get_llm()

        self.additional_requirements = self.config["prompt"]["additional_requirements"]
        self.prompt = self.config["prompt"]["template"]

    def processing_system_prompt(self):
        system_prompt = """
        You are an AI specialized in generating high-quality summaries.
        Your goal is to successfully reconstruct the key information while preserving the original meaning as much as possible, following the requirements.
        The summary must be written in **{language}**.
        Do NOT include any introductory or filler phrases. 
        
        **Strict Requirements:**
        - The summary must be written in **{language}**.
        - **Do NOT include any introductory or filler phrases** such as "Understood," "Starting the summary," or similar expressions. The summary should begin directly with the relevant content.
        - **Preserve image tags** only if they match the exact pattern `!\[\[.*?\]\]`. Do NOT create or modify any image tags that do not follow this pattern, and keep them exactly as they appear, maintaining their position and content.
        - The summary format must follow **{summary_style}**
          - `hierarchical list`: Nested bullet points for structured summaries.
          - `hierarchical numbered list`: Nested numbered points for structured summaries with numbering.
          - `bulleted list`: Simple bullet points without hierarchy.
          - `paragraph`: A concise paragraph-style summary.
        - The summary level must match **{summary_level}**, where:
          - `detailed`: Covers all key points comprehensively.
          - `medium`: Balances between brevity and completeness.
          - `concise`: Provides only the most essential points.
        """

        self.system_prompt = system_prompt.format(
            language=self.config["output"]["language"],
            summary_style=self.config["output"]["summary_style"],
            summary_level=self.config["output"]["summary_level"],
        )

    def get_llm(self):
        summary_provider = self.config["llm_settings"]["summary_provider"]
        temperature = self.config["llm_settings"]["temperature"]
        system_prompt = self.system_prompt

        if summary_provider == "ollama":
            return Ollama(
                model=self.config["ollama"]["model"],
                base_url=self.config["ollama"]["localhost"],
                temperature=temperature,
                system_prompt=system_prompt,
            )

        elif summary_provider == "openai":
            return OpenAI(
                model=self.config["open_ai"]["model"],
                api_key=self.config["open_ai"]["api_key"],
                temperature=self.config["llm_settings"]["temperature"],
                system_prompt=self.config["prompt"]["base_prompt"],
            )

        elif summary_provider == "groq":
            return Groq(
                model=self.config["groq"]["model"],
                api_key=self.config["groq"]["api_key"],
                temperature=temperature,
                system_prompt=system_prompt,
            )

        elif summary_provider == "mistral":
            return Mistral(
                model=self.config["mistral"]["model"],
                api_key=self.config["mistral"]["api_key"],
                temperature=temperature,
                system_prompt=system_prompt,
            )

    def summarize_all_sections_and_save_result(self, parsed_sections, save_path):
        summarized_sections = self.summarize_all_sections(parsed_sections)
        markdown_summary = self.generate_markdown_summary(summarized_sections)
        self.save_markdown_summary(markdown_summary, save_path)

    def summarize_section(self, section, additional_requirements):
        full_prompt = self.prompt.format(
            section=section["content"], additional_requirements=additional_requirements
        )
        response = self.llm.get_response(full_prompt)

        image_tags = "\n".join(
            [f"![{img['id']}]({img['path']})" for img in section.get("images", [])]
        )

        if image_tags and image_tags not in response:
            response += "\n\n" + image_tags
        return response

    def summarize_all_sections(self, sections):
        summarized_sections = []
        exclude_titles = [
            title.lower() for title in self.config["output"]["exclude_section"]
        ]

        for section in sections:
            if section["level"] in self.config["output"]["exclude_level"]:
                continue
            if section["title"].lower() in exclude_titles:
                continue
            if len(section["content"]) == 0:
                summarized_content = ""
            else:
                summarized_content = self.summarize_section(
                    section, self.additional_requirements
                )

            summarized_sections.append(
                {
                    "level": section["level"],
                    "title": section["title"],
                    "summary": summarized_content,
                }
            )
        return summarized_sections

    def generate_markdown_summary(self, summarized_sections):
        markdown_content = []
        for section in summarized_sections:
            header = "#" * section["level"]
            markdown_content.append(f"{header} {section['title']}\n")
            markdown_content.append(f"{section['summary']}\n")

        return "\n".join(markdown_content)

    def save_markdown_summary(self, markdown_content, output_path="paper_summary.md"):
        if len(output_path) > 150:
            file_name = "summary.md"
            output_path = os.path.join(os.path.dirname(output_path), file_name)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
