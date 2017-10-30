gcloud ml-engine local train \
    --module-name word-rnn.train \
    --package-path ./word-rnn \
    -- \
    --data_dir ./data/rnn-c-data