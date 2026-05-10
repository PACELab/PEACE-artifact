from datetime import datetime
import sys

def calculate_average_duration(start, end, requests):
    total_duration = 0
    count = 0
    for request in requests:
        if start <= request['timestamp'] <= end:
            total_duration += request['duration']
            count += 1
    return total_duration / count if count > 0 else 0

def parse_input_file(filename):
    requests = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(': ')
            print(parts)
            if len(parts) >= 3 and parts[2].startswith('request'):
                timestamp = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S')
                request_num = int(parts[2].split()[1])
                duration = float(parts[-1])
                print({'timestamp': timestamp, 'request_num': request_num, 'duration': duration})
                requests.append({'timestamp': timestamp, 'request_num': request_num, 'duration': duration})
    return requests

def main(filename, start_request, end_request):
    requests = parse_input_file(filename)
    average_duration = calculate_average_duration(start_request, end_request, requests)
    print(f"Average duration for requests {start_request}-{end_request}: {average_duration:.2f} seconds")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <filename> <start_request> <end_request>")
        sys.exit(1)
    filename = sys.argv[1]
    start_request = int(sys.argv[2])
    end_request = int(sys.argv[3])
    main(filename, start_request, end_request)
