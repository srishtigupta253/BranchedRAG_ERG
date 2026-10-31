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

## Additional Experiments

### Effect of Retrieval Depth

We investigate how the number of retrieved cases ```m``` affects retrieval quality, alignment with user needs, and response-generation latency. We evaluate four retrieval depths ```m=[5, 15, 30, 45]``` using Mistral-7B, Llama-3.1-8B, and Phi-4-mini. Each setting is evaluated on 100 paired queries.

#### Results

| Model | Metric | m=5 | m=15 | m=30 | m=45 | Friedman p-value | Δ (45−5) [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|
| Mistral-7B | Retrieval match | 0.547 | 0.599 | **0.613** | 0.601 | 3e-9 | +0.054 [+0.035, +0.072] |
| | Need match | 0.755 | 0.763 | 0.750 | **0.789** | 0.009 | +0.034 [+0.007, +0.062] |
| | Latency (s) | 29.4 | 33.6 | 38.8 | 44.3 | 7e-24 | +14.9 [+11.2, +18.6] |
| Llama-3.1-8B | Retrieval match | 0.594 | **0.655** | 0.650 | 0.639 | 3e-4 | +0.045 [+0.018, +0.070] |
| | Need match | 0.655 | 0.771 | 0.721 | **0.784** | 8e-9 | +0.129 [+0.087, +0.170] |
| | Latency (s) | 19.0 | 18.7 | 23.9 | 23.6 | 9e-5 | +4.6 [+2.0, +7.3] |
| Phi-4-mini | Retrieval match | **0.429** | 0.385 | 0.394 | 0.381 | 0.006 | −0.049 [−0.082, −0.013] |
| | Need match | **0.604** | 0.548 | 0.555 | 0.534 | 0.002 | −0.070 [−0.108, −0.032] |

*Note: Paired sample size is 100 queries per model. Friedman tests compare the four retrieval-depth conditions. Confidence intervals refer to the paired difference between m=45 and m=5. Bold indicates the best observed value within each model and metric.*

#### Discussion

The results indicate that the effect of retrieval depth varies across language models. Mistral-7B achieves its highest retrieval-match score at \(m=30\) (0.613), whereas Llama-3.1-8B performs best at \(m=15\) (0.655). Phi-4-mini achieves its highest retrieval-match and need-match scores at \(m=5\), with lower scores at larger retrieval depths.

For need match, Mistral-7B and Llama-3.1-8B achieve their highest scores at \(m=45\) (0.789 and 0.784, respectively). However, increasing retrieval depth also introduces latency costs. For example, Mistral-7B latency increases from 29.4 seconds at \(m=5\) to 44.3 seconds at \(m=45\).

Overall, these findings suggest that retrieval depth should be selected according to the generation model and the desired trade-off between response quality and computational cost. A larger retrieval pool does not consistently lead to better performance.

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
