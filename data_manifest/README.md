# Data Manifest Index

Case manifests use CSV format.

Required columns:

```text
case_id,ct_path,annotation_folder
```

Optional columns include `report_path`, `clinical_path`, and
`pathology_path`.

| File | Purpose |
| --- | --- |
| `case_list_50_tumor.csv` | primary 50-case manifest |
| `case_list_50_tumor_template.csv` | manifest template |
| `case_list_20_tumor_formal_round1.csv` | formal Round 1 subset |
| `case_list_2_test.csv` | two-case test manifest |
| `case_list_50_planned_minimal_block.csv` | minimal-block planning manifest |
