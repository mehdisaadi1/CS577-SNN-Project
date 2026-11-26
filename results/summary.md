# SELU vs ELU: matched ablation

Config: optimizer=sgd, lr=0.01, batch=128, epochs=20 (1580 updates), seeds=[0, 1, 2]. Values are mean ± std over seeds; statistics measured on the 2,000 validation images.

| Metric | SELU | ELU |
|---|---|---|
| D at init | 0.0898 ± 0.0246 | 0.8070 ± 0.0112 |
| D after training | 0.6735 ± 0.0079 | 0.4776 ± 0.0497 |
|   mean part (1/L) sum |mu_l| | 0.0898 ± 0.0057 | 0.3442 ± 0.0491 |
|   variance part (1/L) sum |var_l - 1| | 0.5836 ± 0.0134 | 0.1334 ± 0.0154 |
| Final train cross-entropy | 0.0511 ± 0.0030 | 0.1220 ± 0.0038 |
| Final val cross-entropy | 0.2004 ± 0.0168 | 0.2077 ± 0.0141 |
| Final val accuracy (%) | 93.70 ± 0.18 | 93.47 ± 0.53 |
| Training time per run (s) | 3.9750 ± 0.0371 | 3.9869 ± 0.0184 |
