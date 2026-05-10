# Function to generate all valid MPS percentage combinations, including 100 for all combinations
generate_mps_combinations() {
    local n=$1
    local percentages=(10 20 30 40 50 60 70 80 90 100)
    local total=100

     # Initialize combinations based on n
    if [ $n -eq 1 ]; then
        combinations=( "${percentages[@]}" )
    else
        combinations=( "$(printf "100 %.0s" $(seq 1 $n))" )
        #generate_combinations_recursive "" $n $total
    fi
}

# Recursive helper function to generate combinations
generate_combinations_recursive() {
    local prefix=$1
    local n=$2
    local total=$3

    if [ $n -eq 1 ]; then
        if (( total % 10 == 0 )) && (( total >= 10 )) && (( total <= 100 )); then
            combinations+=( "${prefix}${total}" )
        fi
    else
        for pct in 10 30 50 70 90; do
            if (( pct <= total )); then
                generate_combinations_recursive "${prefix}${pct} " $((n-1)) $((total - pct))
            fi
        done
    fi
}

# Example usage
generate_mps_combinations 2
for combo in "${combinations[@]}"; do
    echo "$combo"
done