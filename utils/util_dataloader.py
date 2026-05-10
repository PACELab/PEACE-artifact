def get_thread_columns(workload_data, filter = False):
    all_threads  = [col for col in workload_data.columns if 'w1_' in col]
    if not filter:
        #default - get all thread columns
        return all_threads
    else:
        #custom filter
        unique_combinations = {}
        for col in all_threads:
            #strip  col.replace("\"", "").replace("(", "").replace(")", "").replace(" ","").split(",") in all_threads
            col_s = col.replace("\"", "").replace("(", "").replace(")", "").replace(" ","").split(",")
            col_s = [c.split("_")[1] for c in col_s]

            #for 
            thread_tuple = tuple(col_s)
            #print(thread_tuple)
            # Use the sorted tuple as a key to ensure uniqueness
            if thread_tuple not in unique_combinations:
                unique_combinations[thread_tuple] = col
            

        #keep keys in unique_combinations if it is in sum_100_combs or over_100_combs
        custom_unique_comb = {}
        #add 100,100
        custom_unique_comb[(str(100), str(100))] = unique_combinations[(str(100), str(100))]

        for k in range(0, 101, 10):
            
            if (str(k), str(100-k)) in unique_combinations and (str(k), str(100)) in unique_combinations:
                custom_unique_comb[(str(k), str(100-int(k)))] = unique_combinations[(str(k), str(100-int(k)))]
                custom_unique_comb[(str(k), str(100))] = unique_combinations[(str(k), str(100))]

        #unique_combinations = {k: v for k, v in unique_combinations.items() if k in sum_100_combs or k in over_100_combs}
        #print(unique_combinations.keys())
        #print(custom_unique_comb.keys())
        #exit(1)

        return list(custom_unique_comb.values())