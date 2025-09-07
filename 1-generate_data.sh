#!/bin/bash
# export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python src/data/create_synth_data.py --generate --data_dir all_data/synthetic/range/eval --synthetic_func synthetic_dataset_with_out_of_range_anomalies --seed 42 --window 30
python src/data/create_synth_data.py --generate --data_dir all_data/synthetic/range/train --synthetic_func synthetic_dataset_with_out_of_range_anomalies --seed 3407 --window 30

python src/data/create_synth_data.py --generate --data_dir all_data/synthetic/point/eval --synthetic_func synthetic_dataset_with_point_anomalies --seed 42 --window 30
python src/data/create_synth_data.py --generate --data_dir all_data/synthetic/point/train --synthetic_func synthetic_dataset_with_point_anomalies --seed 3407 --window 30

python src/data/create_synth_data.py --generate --data_dir all_data/synthetic/freq/eval --synthetic_func synthetic_dataset_with_frequency_anomalies --seed 42 --window 30
python src/data/create_synth_data.py --generate --data_dir all_data/synthetic/freq/train --synthetic_func synthetic_dataset_with_frequency_anomalies --seed 3407 --window 30

python src/data/create_synth_data.py --generate --data_dir all_data/synthetic/trend/eval --synthetic_func synthetic_dataset_with_trend_anomalies --seed 42 --window 30
python src/data/create_synth_data.py --generate --data_dir all_data/synthetic/trend/train --synthetic_func synthetic_dataset_with_trend_anomalies --seed 3407 --window 30


python src/data/create_synth_data.py --generate --add_noise --data_dir all_data/synthetic/noisy-point/eval --synthetic_func synthetic_dataset_with_point_anomalies --seed 42 --window 30
python src/data/create_synth_data.py --generate --add_noise --data_dir all_data/synthetic/noisy-point/train --synthetic_func synthetic_dataset_with_point_anomalies --seed 3407 --window 30

python src/data/create_synth_data.py --generate --add_noise --data_dir all_data/synthetic/noisy-freq/eval --synthetic_func synthetic_dataset_with_frequency_anomalies --seed 42 --window 30
python src/data/create_synth_data.py --generate --add_noise --data_dir all_data/synthetic/noisy-freq/train --synthetic_func synthetic_dataset_with_frequency_anomalies --seed 3407 --window 30

python src/data/create_synth_data.py --generate --add_noise --data_dir all_data/synthetic/noisy-trend/eval --synthetic_func synthetic_dataset_with_trend_anomalies --seed 42 --window 30
python src/data/create_synth_data.py --generate --add_noise --data_dir all_data/synthetic/noisy-trend/train --synthetic_func synthetic_dataset_with_trend_anomalies --seed 3407 --window 30

python src/data/generate_csv.py all_data/synthetic
