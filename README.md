![File Renamer Assistant logo](file_renamer_assistant_logo.png)

# File Renaming Assistant
An OpenAI Assistant (API) for renaming files base on their contents, using Python, Bash, and Linux CLI.

[File Renamer Helper app](https://chat.openai.com/g/g-O1sujw5iD-file-renamer) available on OpenAI's GPT store, to help install, use and understand this repository.

## Initial setup
1. clone [File Renamer Assistant](https://github.com/toadlyBroodle/asst-file-renamer) repository:
    ```git clone https://github.com/toadlyBroodle/asst-file-renamer.git```
2. install dependencies
    ```pip3 install python-docx openpyxl PyPDF2 pillow pytesseract```

3. Save new *credentials.json* file to working directory, replacing with your API key, using format:
    ```
    {
        "openai_api_key": "sk-####",
    }
    ```
4. Create new assistant:
    `python3 file_renamer_asst.py --asst_create`

## Usage, overview
File types currently supported: .txt, .csv, .pdf, .docx, .xlsx, .jpg, .jpeg, .png
Please submit requests for additionally desired file types. 

Renamed files are saved to new *renamed/* subdirectory in target directory, with originals left as is.

EXTRACTION_PERCENT variable may need to be adjusted to achieving accurate new file names, while still preserving file privacy.

Disclaimer: This assistant does **not** upload files directly to OpenAI, but rather parses files locally to extract small percentage of beginning text contexts. This text summary is then necessarily sent to OpenAI API for analysis to generate new file names. Use with discretion and at your own risk. 

All the included functions are not necessarily used for renaming files, but are nonetheless included for user customization purposes, as well as to provide a demonstrative, documented, example of how to create and use OpenAI Assistants API.

```
usage: file_renaming_asst.py [-h] [--asst_create] [--asst_update] [--asst_file_upload ASST_FILE_UPLOAD] [--files_list] [--file_delete FILE_DELETE]
                             [--files_rename FILES_RENAME] [--query_new QUERY_NEW] [--query_last_thread QUERY_LAST_THREAD]
                             [--get_steps GET_STEPS GET_STEPS] [--get_thread GET_THREAD] [--delete_thread DELETE_THREAD] [--verbose]

OpenAI Assistant to rename files in a given directory.

optional arguments:
  -h, --help            show this help message and exit
  --asst_create, -ac    Create the Assistant
  --asst_update, -au    Update the Assistant
  --asst_file_upload ASST_FILE_UPLOAD, -afu ASST_FILE_UPLOAD
                        Upload file for Assistant to retrieve; input: file_path
  --files_list, -fl     List organization's files
  --file_delete FILE_DELETE, -fd FILE_DELETE
                        Delete file; input: file_id
  --files_rename FILES_RENAME, -fr FILES_RENAME
                        Rename all files in directory; input: dir_path
  --query_new QUERY_NEW, -qn QUERY_NEW
                        Create new thread and run query
  --query_last_thread QUERY_LAST_THREAD, -qlt QUERY_LAST_THREAD
                        Append query to last thread
  --get_steps GET_STEPS GET_STEPS, -gs GET_STEPS GET_STEPS
                        Get the run steps; input: thread_id, run_id
  --get_thread GET_THREAD, -gt GET_THREAD
                        Get the thread; input: thread_id, "new"
  --delete_thread DELETE_THREAD, -dt DELETE_THREAD
                        Delete the thread; input: thread_id
  --verbose, -v         Enable verbose output
```