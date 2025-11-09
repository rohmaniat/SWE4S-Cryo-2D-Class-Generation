# Let's call this file func_testing.sh until we have another functional test file. Then let's call them something more specific.

test -e ssshtest || curl -q -O https://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest

# Define a cleanup to make sure we aren't just using last run's visualization
cleanup() {
    rm -f "test/func/EXAMPLE_visualization.png"
    rm -f "test/func/coord_predictions.csv"
    }
# Trap cleanup() to run upon exit
trap cleanup EXIT

# ----- PREDICTION TESTING -----
# First, just run a basic predict execution
run standard_predict python src/predict.py \
--mrc_file "test/func/EXAMPLE_Falcon_2012_06_12-14_33_35_0.mrc" \
--output_csv "test/func/coord_predictions.csv" \
--model_path "src/models/my_model.pt" \
--threshold 1.5
assert_in_stdout "Prediction finished in "
assert_exit_code 0

# Check to make sure it can produce an image
run produce_image python src/predict.py \
--mrc_file "test/func/EXAMPLE_Falcon_2012_06_12-14_33_35_0.mrc" \
--output_csv "test/func/coord_predictions.csv" \
--model_path "src/models/my_model.pt" \
--threshold 2 \
--ground_truth_csv "EXAMPLE_Falcon_2012_06_12-14_33_35_0.csv" \
--output_image "test/func/EXAMPLE_visualization.png"
assert_in_stdout "Visualization saved!"
assert_exit_code 0
# Make sure the visualization was produced
run check_for_visualization test -f test/func/EXAMPLE_visualization.png
assert_exit_code 0

# Check mis-inputs
run misinputs python src/predict.py \
--output_csv "test/func/coord_predictions.csv" \
--threshold 2
assert_in_stdout "Error: The following required arguments are missing:"
assert_in_stdout "mrc_file"
assert_in_stdout "model_path"
assert_exit_code 1

# Check bad model upload handling
run bad_model_handling python src/predict.py \
--mrc_file "test/func/EXAMPLE_Falcon_2012_06_12-14_33_35_0.mrc" \
--output_csv "test/func/coord_predictions.csv" \
--model_path "src/models/FAKE_model.pt" \
--threshold 1.5
assert_in_stdout "Could not load model: "
assert_exit_code 2

# Check configuration file control
run configuration_check python src/predict.py --config test/func/standard_config.ini
assert_in_stdout "Model loaded successfully!"
assert_exit_code 0