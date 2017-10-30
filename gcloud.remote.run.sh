export BUCKET_NAME=rnn-c-data
export JOB_NAME="train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=europe-west1

gcloud ml-engine jobs submit training $JOB_NAME\
    --job-dir gs://$BUCKET_NAME/$JOB_NAME \
    --runtime-version 1.0 \
    --module-name word_rnn.train \
    --package-path ./word_rnn \
    --region $REGION \
    --config cloudml-gpu.yaml \
    -- \
    --data_dir gs://$BUCKET_NAME/ \
    --log_dir gs://$BUCKET_NAME/logs/ \
    --save_dir gs://$BUCKET_NAME/save/ \
    --init_from gs://$BUCKET_NAME/save/ \
    --num_epochs 15
    

# gcloud ml-engine jobs submit training $JOB_NAME \
#     --job-dir gs://$BUCKET_NAME/$JOB_NAME \
#     --runtime-version 1.0 \
#     --module-name trainer.example5 \
#     --package-path ./trainer \
#     --region $REGION \
#     --config=trainer/cloudml-gpu.yaml \
#     -- \
#     --train-file gs://tf-learn-simple-sentiment/sentiment_set.pickle