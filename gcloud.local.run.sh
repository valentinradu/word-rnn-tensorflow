gcloud ml-engine local train \
    --module-name word-rnn.train \
    --package-path ./word-rnn \
    -- \
    --data_dir ./word-rnn/data/rnn-c-data