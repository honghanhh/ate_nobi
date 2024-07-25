### en
python nobi_classifier.py --train1 ./acter/nobi/data/en_corp_ann.csv --train2 ./acter/nobi/data/en_wind_ann.csv --val ./acter/nobi/data/en_equi_ann.csv --test ./acter/nobi/data/en_htfl_ann.csv --gold_val ./acter/nobi/groundtruth/equi_en_terms.tsv --gold_test ./acter/nobi/groundtruth/htfl_en_terms.tsv --output_dir ./output_dir --log_dir ./logs --metric_path en_ann_equi_htfl_nobi.txt --prediction_path en_ann_equi_htfl_pred_nobi.txt --hf_repo_id tthhanh/xlm-ate-nobi-en


python nobi_classifier.py --train1 ./acter/nobi/data/en_corp_nes.csv --train2 ./acter/nobi/data/en_wind_nes.csv --val ./acter/nobi/data/en_equi_nes.csv --test ./acter/nobi/data/en_htfl_nes.csv --gold_val ./acter/nobi/groundtruth/equi_en_terms_nes.tsv --gold_test ./acter/nobi/groundtruth/htfl_en_terms_nes.tsv --output_dir ./output_dir --log_dir ./logs --metric_path en_nes_equi_htfl_nobi.txt --prediction_path en_nes_equi_htfl_pred_nobi.txt --hf_repo_id tthhanh/xlm-ate-nobi-en-nes


### fr
python nobi_classifier.py --train1 ./acter/nobi/data/fr_corp_ann.csv --train2 ./acter/nobi/data/fr_wind_ann.csv --val ./acter/nobi/data/fr_equi_ann.csv --test ./acter/nobi/data/fr_htfl_ann.csv --gold_val ./acter/nobi/groundtruth/equi_fr_terms.tsv --gold_test ./acter/nobi/groundtruth/htfl_fr_terms.tsv --output_dir ./output_dir --log_dir ./logs --metric_path fr_ann_equi_htfl_nobi.txt --prediction_path fr_ann_equi_htfl_pred_nobi.txt --hf_repo_id tthhanh/xlm-ate-nobi-fr


python nobi_classifier.py --train1 ./acter/nobi/data/fr_corp_nes.csv --train2 ./acter/nobi/data/fr_wind_nes.csv --val ./acter/nobi/data/fr_equi_nes.csv --test ./acter/nobi/data/fr_htfl_nes.csv --gold_val ./acter/nobi/groundtruth/equi_fr_terms_nes.tsv --gold_test ./acter/nobi/groundtruth/htfl_fr_terms_nes.tsv --output_dir ./output_dir --log_dir ./logs --metric_path fr_nes_equi_htfl_nobi.txt --prediction_path fr_nes_equi_htfl_pred_nobi.txt --hf_repo_id tthhanh/xlm-ate-nobi-fr-nes


### nl
python nobi_classifier.py --train1 ./acter/nobi/data/nl_corp_ann.csv --train2 ./acter/nobi/data/nl_wind_ann.csv --val ./acter/nobi/data/nl_equi_ann.csv --test ./acter/nobi/data/nl_htfl_ann.csv --gold_val ./acter/nobi/groundtruth/equi_nl_terms.tsv --gold_test ./acter/nobi/groundtruth/htfl_nl_terms.tsv --output_dir ./output_dir --log_dir ./logs --metric_path nl_ann_equi_htfl_nobi.txt --prediction_path nl_ann_equi_htfl_pred_nobi.txt --hf_repo_id tthhanh/xlm-ate-nobi-nl


python nobi_classifier.py --train1 ./acter/nobi/data/nl_corp_nes.csv --train2 ./acter/nobi/data/nl_wind_nes.csv --val ./acter/nobi/data/nl_equi_nes.csv --test ./acter/nobi/data/nl_htfl_nes.csv --gold_val ./acter/nobi/groundtruth/equi_nl_terms_nes.tsv --gold_test ./acter/nobi/groundtruth/htfl_nl_terms_nes.tsv --output_dir ./output_dir --log_dir ./logs --metric_path nl_nes_equi_htfl_nobi.txt --prediction_path nl_nes_equi_htfl_pred_nobi.txt --hf_repo_id tthhanh/xlm-ate-nobi-nl-nes

### mul
python nobi_classifier.py --train1 ./acter/mul/train_ann.csv --val ./acter/mul/val_ann.csv --test ./acter/nobi/data/nl_htfl_ann.csv --gold_val ./acter/mul/gold_mul_val_ann.csv --gold_test ./acter/nobi/groundtruth/htfl_nl_terms.tsv --output_dir ./output_dir --log_dir ./logs --metric_path mul_ann_htfl_nobi.txt --prediction_path mul_ann_htfl_pred_nobi.txt --hf_repo_id tthhanh/xlm-ate-nobi-mul


python nobi_classifier.py --train1 ./acter/mul/train_nes.csv --val ./acter/mul/val_nes.csv --test ./acter/nobi/data/nl_htfl_nes.csv --gold_val ./acter/mul/gold_mul_val_nes.csv --gold_test ./acter/nobi/groundtruth/htfl_nl_terms_nes.tsv --output_dir ./output_dir --log_dir ./logs --metric_path mul_nes_htfl_nobi.txt --prediction_path mul_nes_htfl_pred_nobi.txt --hf_repo_id tthhanh/xlm-ate-nobi-mul-nes