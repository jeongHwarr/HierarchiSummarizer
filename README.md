# HierarchiSummarizer
![logo](asset/logo.png)

## Overview
HierarchiSummarizer is a document processing tool that analyzes document hierarchy, extracts structured content, summarizes, and translates paragraphs from Markdown and PDF files. It supports multiple LLM providers and provides customizable output formats.

## Features
- Converts PDF and Markdown files into structured summaries.
- Uses Mistral API for PDF to Markdown conversion (**Mistral API key required**).
- Supports multiple LLM providers for summarization (Mistral, OpenAI, Groq, Ollama).
- Excludes specified sections and heading levels.
- Provides hierarchical or paragraph-style summaries.
- Outputs results in multiple languages.
- Allows customized summary output through prompt tuning (`additional_requirements` in `config.yaml`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jeongHwarr/HierarchiSummarizer.git
   cd HierarchiSummarizer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## API Key Requirement
Since HierarchiSummarizer uses the Mistral API for PDF to Markdown conversion, obtaining a Mistral API key is **mandatory**.
Example configuration (in `config.yaml`):
```yaml
mistral:
  model: open-mistral-nemo # https://docs.mistral.ai/getting-started/models/models_overview/
  api_key: your_api_key # Required! This API key is needed for converting PDFs to Markdown.
```

### How to Get a Mistral API Key
1. Visit the [Mistral AI website](https://docs.mistral.ai/getting-started/)
2. Sign up or log in
3. Navigate to the API section to generate your key
4. Add your API key to the `config.yaml` file under `mistral.api_key`

## Usage
1. Modify the Configuration (`config.yaml`).
   - Open the `config.yaml` file and ensure you add your mistral api_key. This key is required for the script to function properly.

2. Place PDF or Markdown Files in the `workspace/to_process/` Folder.
   - Place any PDF or Markdown documents you wish to process in the `workspace/to_process/` directory.
   - If the document is a PDF, it will automatically be converted to Markdown before summarization.

3. Run the Script.
   - To begin processing, execute the following command:
     ```bash
     python run.py
     ```
  
### Processing Workflow
1. Place PDF or Markdown files in `workspace/to_process/`.
    - If the file is a PDF, it will first be converted to Markdown format before the summarization process begins.
    - The conversion result will be saved as `workspace/output/{title}/pdf_to_md.md`.
2. Summarized files will be saved in `workspace/output/`.
    - For each processed document, the final summary will be saved as `title_summary.md` in the corresponding folder.
    - If the document was originally a PDF, you will find the Markdown conversion in `workspace/output/{title}/pdf_to_md.md`.
3. Processed files will be moved to `workspace/done/`.

## Configuration
Edit the `config.yaml` file to customize the summarization settings.

### 1. **`llm_settings`**
This section configures the AI model provider for summarization. Choose the model provider you want to use, and adjust the settings accordingly.

- **`summary_provider`**: The AI model provider for summarization. Options:
    - `mistral`: Use the Mistral AI API.
    - `open_ai`: Use the OpenAI API.
    - `groq`: Use the Groq AI API.
    - `ollama`: Use the Ollama model (for local inference).
- **`temperature`**: Controls the creativity of the generated text. Higher values (e.g., 0.7) produce more creative text, while lower values (e.g., 0.3) generate more consistent responses.

### 2. **API Key Configuration**
- Each AI model provider (such as Mistral, OpenAI, Groq) requires an API key to function properly.
- Ensure that you add the required API keys for the selected provider.
- **mistral api_key is REQUIRED!**

#### For Mistral:
The Mistral API is used for converting PDF files into Markdown format. You **must** provide an API key in the `mistral` section to enable this conversion process.

Example configuration:
```yaml
mistral:
  model: open-mistral-nemo # https://docs.mistral.ai/getting-started/models/models_overview/
  api_key: your_api_key # Required! This API key is needed for converting PDFs to Markdown.
```

### 3. **Directory Settings**
This section defines the directory paths for input, output, and processed files.

- **`input_dir_path`**: The directory where documents to be processed are located. Default: `workspace/to_process`.
- **`output_dir_path`**: The directory where summarized files will be saved. Default: `workspace/output`.
- **`done_dir_path`**: The directory where processed files will be moved after summarization. Default: `workspace/done`.

### 4. **Output Settings**
This section customizes the output format and language of the summary.

- **`language`**: The language of the summary. Default is `english`.
- **`exclude_level`**: A list of heading levels to exclude from the summary. For example, `- 1` excludes level 1 headings (e.g., `# Title`).
- **`exclude_section`**: A list of sections to exclude from summarization, such as `- reference`, `- references`.
- **`summary_style`**: The style of the summary. Options:
    - `hierarchical bullet list`
    - `hierarchical numbered list`
    - `hierarchical bullet list`
    - `paragraph`
    - Others depending on your needs.
- **`summary_level`**: The detail level of the summary. Options:
    - `detailed`
    - `medium`
    - `concise`
    - Others depending on your needs.

### 5. **Prompt Customization**
Customize the prompt that is sent to the AI model for summarization.

- **`additional_requirements`**: Add custom instructions for how the AI should generate the summary. For example:
  ```yaml
  additional_requirements: |
    **Sentence Structure**: Do NOT use periods at the end of sentences.


## Supported Providers
HierarchiSummarizer supports multiple LLM providers:
- **Mistral** 
- **OpenAI** 
- **Groq** 
- **Ollama**

## License
This project is licensed under the MIT License.

## Contributing
Pull requests and feature suggestions are welcome.

## Contact
For issues or inquiries, open an issue on GitHub or contact the maintainer.

