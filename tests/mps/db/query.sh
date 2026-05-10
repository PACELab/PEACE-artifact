
#!/bin/bash
SERVER_PATH="/opt/heavyai"
# Define the SQL query
sql_query='\memory_summary SELECT origin_city AS "Origin", dest_city AS "Destination", AVG(airtime) AS "Average Airtime" FROM flights_2008_7M WHERE distance < 15000 GROUP BY origin_city, dest_city;'
sql_query1='SELECT origin_city AS "Origin", dest_city AS "Destination", AVG(airtime) AS "Average Airtime" FROM flights_2008_7M WHERE distance < 15000 GROUP BY origin_city, dest_city;'

#time CPU 188s vs GPU 46s
#echo "$sql_query" | "$HEAVYAI_PATH"/bin/heavysql -p HyperInteractive -t
# Execute the SQL query and pipe the output to the heavysql command
while true; do
    echo "$sql_query" | "$SERVER_PATH/bin/heavysql" -p HyperInteractive -t
    echo "\clear_gpu" | "$SERVER_PATH/bin/heavysql" -p HyperInteractive -t
    
    sleep 0.5
done

#