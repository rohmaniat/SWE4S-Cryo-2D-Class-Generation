# Let's call this file func_testing.sh until we have another functional test file. Then let's call them something more specific.

test -e ssshtest || curl -q -O https://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest

# ----- PREDICTION TESTING -----
# First, just run a basic predict execution
run standard_predict python src/predict.py \
--mrc_file "test/func/EXAMPLE_Falcon_2012_06_12-14_33_35_0.mrc" \
--output_csv "test/func/coord_predictions.csv" \
--model_path "src/models/my_model.pt" \
--threshold 1.5
assert_in_stdout "Model loaded successfully!"
assert_exit_code 0