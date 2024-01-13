# File Renaming Assistant
An OpenAI Assistant (API) for renaming files based on their contents. 

## Notes
Currently, assistants support uploading at most 20 files at a time. If this limitation is not lifted soon, we will devise a workaround.
[Assistants support](https://platform.openai.com/docs/assistants/tools/supported-files) retrieval of these [file types](supported-file-types.csv), hopefully all common types are added soon.

Renamed files are saved to new *renamed/* subdirectory in target directory, with originals left as is.

Disclaimer: This assistant currently requires uploading files to OpenAI's servers, to analyze content. Use with discretion and at your own risk.

## Initial setup
1. install dependencies
    ```
    pip3 install python-docx openpyxl PyPDF2 pillow pytesseract
    ```

2. Save new `credentials.json` file to working directory, replacing with your API key, using format:
    ```
    {
        "openai_api_key": "sk-####",
    }
    ```
3. Create new assistant:
    `python3 file_renamer_asst.py --asst_create`

## Usage
```
usage: file_renaming_asst.py [-h] [--asst_create] [--asst_update] [--asst_file_upload ASST_FILE_UPLOAD] [--files_list] [--file_delete FILE_DELETE]
                             [--files_rename FILES_RENAME] [--query_new QUERY_NEW] [--query_last_thread QUERY_LAST_THREAD]
                             [--get_steps GET_STEPS GET_STEPS] [--get_thread GET_THREAD] [--delete_thread DELETE_THREAD] [--verbose]

OpenAI Assistant to rename directory files

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