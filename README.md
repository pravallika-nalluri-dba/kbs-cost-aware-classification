# Cost-aware classification for credit default

This repository reproduces a short communication on cost-aware credit decisions. Two validation rules turn predicted default probabilities into a thresholded action. Strategy A (prediction-first) fixes the model by validation F1 at threshold 0.50, then chooses a cost-minimizing threshold. Strategy B (decision-first) chooses the model and the threshold together by validation cost. The test split is scored once.

The primary split and the bootstrap use seed 42. The same split is repeated for seeds 42 through 61. Model fitting keeps random seed 42, so only the partition changes. The min, median, and max across those splits are a stability range, not a confidence interval. A separate percentile bootstrap, 1,000 resamples and seed 42, intervals the mean of the 20 split-level cost differences. That interval is not the seed-42 test-set bootstrap. The 20 splits reuse the same 30,000 accounts, so the interval measures partition stability on this file, not an independent population replication.

For ratio 20, after the primary grid is locked, the script also searches thresholds 0.001 to 0.009. On seed 42 that search moves the prediction-first threshold from 0.01 to 0.009 and does not lower test cost. Validation Brier scores are recorded and do not enter selection.

The manuscript is [`manuscript.tex`](manuscript.tex).

## Reproduce

Python 3.12 is the version used for the reported run. From this directory:

```bash
pip install -r requirements.txt
python run.py
```

`python run.py` downloads the UCI Default of Credit Card Clients workbook (30,000 accounts) into `data_cache/` on the first run, fits the models, and writes `results.json`, `Figure_1.pdf`, and `Figure_2.pdf`. A network connection is required for that first download. Later runs reuse the cache. The script then checks that the test costs and thresholds printed in `manuscript.tex` match `results.json`, and it exits with a non-zero status if they differ.

Optional manuscript PDF:

```bash
pdflatex manuscript.tex
```

## Figures

- `Figure_1.pdf` is the seed-42 validation cost of the prediction-first model against the threshold, one panel for each cost ratio. The circle is the chosen threshold. The dotted line is threshold 0.50.
- `Figure_2.pdf` is a plot from this run. Markers are the seed-42 rules. Panel (a) shows the chosen threshold against the false-negative cost ratio for both strategies. Panel (b) shows the test cost per case, including strategy A scored at threshold 0.50. Whiskers are the minimum and maximum over 20 stratified splits (seeds 42 through 61). They are not a confidence interval. The bootstrap interval in the manuscript is only for seed 42.

## License

MIT License

Copyright (c) 2026 Pravallika Nalluri

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
