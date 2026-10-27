In order to convert an epub file into desired input:
 ``` 
  python epub_cases.py parse book.epub --work work/
```
A noteworthy point here is that cases for each book need to be saved in different files. Ensure you change the name in [Line 508](epub_cases.py).
```
  python epub_cases.py window --work work/ [--target-words 1800 --overlap-words 350]
  python run_llama.py --work work/ --model model_name --prompt extraction_prompt.txt
  python epub_cases.py merge --work work/ --responses work/responses.jsonl
```

To combine individual cases of all books in to one:
```
  cat cases_book1.jsonl cases_book2.jsonl ... cases_bookx.jsonl > kb_all.jsonl
```

In order to run individual queries:
```
 python empathy_rag.py --cases kb_all.jsonl --model model-name --query "My boss criticised me in front of everyone and I can't stop shaking."
```

In order to prepare and run all test queries:
```
 python run_benchmark.py prepare --cases kb_all.jsonl --queries queries.jsonl --bench bench/
 python run_benchmark.py run --cases kb_all.jsonl --queries queries.jsonl --bench bench/ --model-config models.json --models model_name
```

Outputs can be found in bench folder.

_Disclaimer: We do not publish any actual results and only give the pipeline due to copyright issues on the chosen books._
