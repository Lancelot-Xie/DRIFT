#!/bin/bash

# Configuration parameters 
BASE_INPUT_DIR="" # The path of source data
BASE_OUTPUT_DIR=""
SAMPLE_RATIO=1.0  # Use 100% of the dataset (adjust as needed)
NUM_WORKERS=256  # Adjust based on your system's capabilities

# Define the datasets to process
DATASETS=("train" "val" "test")

# Display script header
echo "===================================="
echo "QA Generation Tool Runner (Batch Mode)"
echo "===================================="
echo "Started at: $(date)"
echo "Base input directory: $BASE_INPUT_DIR"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Sample ratio: ${SAMPLE_RATIO} ($(python3 -c "print($SAMPLE_RATIO * 100)")%)"
echo "Using $NUM_WORKERS parallel workers"
echo "Processing datasets: ${DATASETS[@]}"
echo "===================================="

# Create output directory if it doesn't exist
mkdir -p "$BASE_OUTPUT_DIR"

# Initialize counters
success_count=0
skipped_count=0
failed_count=0
total_count=${#DATASETS[@]}

# Process each dataset
for dataset in "${DATASETS[@]}"; do
    INPUT_FILE="${BASE_INPUT_DIR}/${dataset}.parquet"
    OUTPUT_FILE="${BASE_OUTPUT_DIR}/${dataset}_qa.jsonl"
    
    echo ""
    echo "------------------------------------"
    echo "Processing: $dataset"
    echo "------------------------------------"
    echo "Input file: $INPUT_FILE"
    echo "Output file: $OUTPUT_FILE"
    
    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: Input file does not exist: $INPUT_FILE"
        echo "Skipping $dataset..."
        ((skipped_count++))
        continue
    fi
    
    # Check if output file already exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "✓ Output file already exists: $OUTPUT_FILE"
        echo "Skipping $dataset (file already processed)..."
        ((success_count++))
        continue
    fi
    
    echo "Started processing at: $(date)"
    
    # Run the QA generation script
    python ./generate_qa.py \
        "$INPUT_FILE" \
        "$OUTPUT_FILE" \
        --sample_size "$SAMPLE_RATIO" \
        --num_workers "$NUM_WORKERS"
    
    # Check if the script completed successfully
    if [ $? -eq 0 ]; then
        echo "✓ $dataset processing completed successfully!"
        echo "Output saved to: $OUTPUT_FILE"
        if [ -f "$(dirname "$OUTPUT_FILE")/qa_example.json" ]; then
            echo "Example QA pairs saved to: $(dirname "$OUTPUT_FILE")/qa_example.json"
        fi
        ((success_count++))
    else
        echo "✗ Error: $dataset processing failed!"
        ((failed_count++))
    fi
    
    echo "Finished processing $dataset at: $(date)"
done

# Display final summary
echo ""
echo "===================================="
echo "BATCH PROCESSING SUMMARY"
echo "===================================="
echo "Total datasets: $total_count"
echo "Successfully processed/existing: $success_count"
echo "Failed: $failed_count"
echo "Skipped (missing input): $skipped_count"
echo "===================================="

if [ $failed_count -eq 0 ]; then
    if [ $success_count -eq $total_count ]; then
        echo "🎉 All datasets processed or already exist!"
    else
        echo "✓ All available datasets processed successfully!"
        echo "Note: Some input files were missing and skipped."
    fi
    echo "All outputs available in: $BASE_OUTPUT_DIR"
else
    echo "⚠️  Some datasets failed to process. Check the logs above."
    echo "Successfully processed/existing: $success_count"
    echo "Failed: $failed_count"
fi

echo "Batch processing finished at: $(date)"
echo "===================================="

# Exit with error code only if there were actual processing failures
if [ $failed_count -gt 0 ]; then
    exit 1
fi