
# Function to record and display elapsed time
record_time() {
    END_TIME=$(date +"%s")
    ELAPSED_TIME=$((END_TIME - START_TIME))
    echo "Time elapsed for $1: $ELAPSED_TIME seconds"
}

# Start time for the entire script
START_TIME=$(date +"%s")

cd template
RUNS=2
RUN_START_TIME=$(date +"%s")

for RUN in $( seq 1 $RUNS )
do
python ../utils/configurator.py --input_file ./profile/test"$RUN".csv --template_file ./template.jinja2 --output_name test"$RUN".yaml --run "$RUN"
done
sleep 10
RUN_END_TIME=$(date +"%s")

record_time "RUN $RUNS"