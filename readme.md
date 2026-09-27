## Quick reproduction
```
python run_full.py
python benchmark_baselines.py
```

## Full reproduction
```
python run_full.py
python audit_dataset.py dataset_full
python dataset_statistics.py

python generate_level2_small.py
python run_level2_small.py
python ablation_level2.py

python generate_level2_scaling.py
python run_level2_scaling.py
python analyze_level2_scaling.py
```