To run the Training model following command helps 

    python3 titanic-ml-pipeline.py train --output-dir artifacts_end_to_end --n-iter 18

Create sample raw input:
    python3 titanic-ml-pipeline.py sample-input --output-csv artifacts_end_to_end/sample_passengers.csv --rows 10

Predict:

    python3 titanic-ml-pipeline.py predict --artifact-dir artifacts_end_to_end --input-csv artifacts_end_to_end/sample_passengers.csv --output-csv artifacts_end_to_end/predictions.csv

Monitor:

    python3 titanic-ml-pipeline.py monitor --artifact-dir artifacts_end_to_end --input-csv artifacts_end_to_end/sample_passengers.csv --output-json artifacts_end_to_end/monitoring_report.json



Californina housing- data set:


python california_pipeline.py train --output-dir artifacts_ca --n-iter 20
python california_pipeline.py train --output-dir artifacts_ca --n-iter 20 --drop-capped
python california_pipeline.py sample-input --output-csv sample.csv --rows 20
python california_pipeline.py predict --artifact-dir artifacts_ca --input-csv sample.csv
python california_pipeline.py monitor --artifact-dir artifacts_ca --input-csv new_data.csv