### en
python nobi_classifier.py --train1 ./acter/nobi/data/en_corp_ann.csv --train2 ./acter/nobi/data/en_wind_ann.csv --val ./acter/nobi/data/en_equi_ann.csv --test ./acter/nobi/data/en_htfl_ann.csv --gold_val ./acter/nobi/groundtruth/equi_en_terms.tsv --gold_test ./acter/nobi/groundtruth/htfl_en_terms.tsv --output_dir ./output_dir --log_dir ./logs --metric_path en_ann_equi_htfl_nobi.txt --prediction_path en_ann_equi_htfl_pred_nobi.txt
