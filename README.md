# File Renaming Assistant
An OpenAI Assistant (API) for renaming files based on their contents. Currently, [assistants support](https://platform.openai.com/docs/assistants/tools/supported-files) retrieval of these [file types](supported-file-types.csv).

## Initial setup
1. Save new `credentials.json` file to working directory, replacing with your API key, using format:
    ```
    {
        "openai_api_key": "sk-####",
    }
    ```
2. Create new assistant:
    `python3 file_renamer_asst.py --asst_create`

## Usage
- Create new thread and query:
    `python3 file_renamer_asst.py --query_new`
- Query last thread:
    `python3 file_renamer_asst.py --query_last_thread`
