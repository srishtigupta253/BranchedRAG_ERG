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

### Table 1. Details of major books considered in our study

| Name | Author | ISBN | Windows | Cases | Avg. words/window |
|---|---|---|---:|---:|---:|
| Dare to Lead | Brené Brown | 0399592547, 9780399592546 | 30 | 58 | 1721 |
| The Stress Solution | Arthur Ciaramicoli | 0452282284, 9780452282285 | 44 | 43 | 1441 |
| The Power of Empathy | Arthur Ciaramicoli | 1608684091, 9781608684090 | 710 | 710 | 361 |
| Empathy and Moral Development | Martin L. Hoffman | 052101297X, 9780521012973 | 74 | 306 | 359 |
| Practical Empathy | Indi Young | 1933820640, 9781933820644 | 17 | 17 | 1663 |
| Don't Be Nice, Be Real | Kelly Mft Bryson | 0972002855, 9780972002851 | 70 | 62 | 1682 |
| Empathy: Why It Matters, and How to Get It | Roman Krznaric | 1846043859, 9781846043857 | 49 | 40 | 1779 |
| A Rumor of Empathy | Lou Agosta | 1317575334, 9781317575337 | 34 | 7 | 1701 |
| The Art of Empathy | Karla McLaren | 1622030672, 9781622030675 | 83 | 26 | 1700 |
| A Way of Being | Carl Ransom Rogers | 0395755301, 9780395755303 | 120 | 34 | 928 |
| Nonviolent Communication | Marshall B. Rosenberg | 1892005549, 9781892005540 | 112 | 83 | 581 |
| Empathy in the Context of Philosophy | Lou Agosta | 1349317543, 9781349317547 | 179 | 8 | 445 |
| Empathy Reconsidered | Arthur C. Bohart | 1557984107, 9781557984104 | 3 | 0 | 458 |
| Listening Well | William R. Miller | 1532634846, 9781532634840 | 25 | 22 | 1149 |
| From Detached Concern to Empathy | Jodi Halpern | 0195111192, 9780195111194 | 168 | 45 | 412 |
| The Age of Empathy | Frans de Waal | 0285638904, 9780285638907 | 56 | 46 | 1761 |
| The Empathic Brain | Christian Keysers | 1105018075, 9781105018077 | 60 | 4 | 1666 |
| The Empathy Effect | Helen Riess | 1683640292, 9781683640295 | 49 | 39 | 1679 |
| Emotional Intelligence | Daniel Goleman | 1526633620, 9781526633620 | 92 | 49 | 1607 |
