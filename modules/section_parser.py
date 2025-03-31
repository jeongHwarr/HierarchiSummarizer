import re
from collections import defaultdict


class SectionParser:
    def __init__(self, md_file_path):
        self.md_file_path = md_file_path
        self.md_lines = self.read_markdown()
        self.hierarchy = defaultdict(list)

    def read_markdown(self):
        with open(self.md_file_path, "r", encoding="utf-8") as f:
            return f.readlines()

    def parse(self):
        sections = []
        current_section = {"title": "", "level": 0, "content": ""}

        for line in self.md_lines:
            if line == "\n":
                continue
            match = re.match(r"^(#{1,6})\s+(.*)", line)
            if match:
                if current_section["title"]:
                    sections.append(current_section)

                level = len(match.group(1))
                title = match.group(2)
                current_section = {"title": title, "level": level, "content": ""}
            else:
                current_section["content"] += line + "\n"

        if current_section["title"]:
            sections.append(current_section)

        return sections

    def display_hierarchy(self):
        for parent, children in self.hierarchy.items():
            print(f"{parent}:")
            for child in children:
                print(f"  - {child}")
