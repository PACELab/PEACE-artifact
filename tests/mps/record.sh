url='http://localhost:8000/metrics'

while true; do
    result=$(curl "$url")
    echo $result
    sleep 1
done