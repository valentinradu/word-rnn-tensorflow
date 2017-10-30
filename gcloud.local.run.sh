export BUCKET_NAME=rnn-c-data
export JOB_NAME="train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=europe-west1

gcloud ml-engine local train \
    --module-name word_rnn.train \
    --package-path ./word_rnn \
    -- \
    --data_dir gs://$BUCKET_NAME \
    --log_dir gs://$BUCKET_NAME/logs \
    --save_dir gs://$BUCKET_NAME/save