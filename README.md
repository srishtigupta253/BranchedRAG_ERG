In order to convert an epub file into desired input:
 ``` 
  python epub_cases.py parse book.epub --work work/
```
A noteworthy point here is that cases for each book need to be saved in different files. Ensure you change the name in [Line 3]().
```
  python epub_cases.py window --work work/ [--target-words 1800 --overlap-words 350]
  python run_llama.py --work work/ --model model_name --prompt extraction_prompt.txt
  python epub_cases.py merge --work work/ --responses work/responses.jsonl
```

