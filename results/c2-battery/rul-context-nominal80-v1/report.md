# Direct RUL context sweep

MIT: nominal 1.1Ah x 80%, audited official cycle_life. Other groups retain initial-reference80.
201 heads, frozen C2 backbone, 2 source cells x 16 labels, 3 seeds. Fixed labels/origins across context lengths.
NASA only L8/16/32 due to EOL61 cell. All other groups start at128, so late-survivor evaluation.
81 MIT exact labels restored via official labels, not last-record+1; 5 genuinely censored cells stay censored.
Within-dataset and transfer are separate. Existing historical results unchanged. This is exploratory reused data.

| Dataset | L | Head cell-macro MAE | Seed SD |
|---|---:|---:|---:|
| calce_cs2_1c | 8 | 110.937 | 0.998 |
| calce_cs2_1c | 16 | 107.845 | 1.998 |
| calce_cs2_1c | 32 | 104.863 | 0.152 |
| calce_cs2_1c | 64 | 94.747 | 2.102 |
| calce_cs2_1c | 128 | 76.139 | 2.968 |
| mit_batch1 | 8 | 790.137 | 51.817 |
| mit_batch1 | 16 | 362.844 | 42.358 |
| mit_batch1 | 32 | 452.311 | 42.704 |
| mit_batch1 | 64 | 481.699 | 35.341 |
| mit_batch1 | 128 | 305.647 | 10.188 |
| mit_batch2 | 8 | 97.953 | 8.256 |
| mit_batch2 | 16 | 98.285 | 1.968 |
| mit_batch2 | 32 | 93.635 | 1.135 |
| mit_batch2 | 64 | 117.517 | 6.158 |
| mit_batch2 | 128 | 111.399 | 4.608 |
| mit_batch3 | 8 | 174.344 | 1.897 |
| mit_batch3 | 16 | 177.012 | 5.226 |
| mit_batch3 | 32 | 165.094 | 4.903 |
| mit_batch3 | 64 | 180.921 | 0.733 |
| mit_batch3 | 128 | 202.818 | 3.502 |
| nasa_aging | 8 | 24.873 | 1.666 |
| nasa_aging | 16 | 25.049 | 0.595 |
| nasa_aging | 32 | 25.121 | 1.597 |
| xjtu_2c | 8 | 65.097 | 3.314 |
| xjtu_2c | 16 | 51.665 | 1.115 |
| xjtu_2c | 32 | 35.347 | 1.631 |
| xjtu_2c | 64 | 29.286 | 3.045 |
| xjtu_2c | 128 | 17.956 | 1.110 |

All-cell figures: 237. Three actual seeds and their mean/range; range is not a confidence interval.