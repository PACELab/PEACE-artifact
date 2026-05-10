import pickle
from utils.loaddata import filter_data, train_test_custom_split, Dataloader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import json
import h2o
from pathlib import Path
import logging
import os

class Predictor():
    def __init__(self, args, modelpath, stage1_data, stage2_data, actual_share_throughput_data, excluded_cols):
        
        self.args = args
        if modelpath:
            self.model = self.loadModel(Path(modelpath))
        self.df = pd.read_csv(stage1_data)
        #print(stage2_data)
        #if stage2_data:
        #    self.stage2df = pd.read_csv(stage2_data)
        
        self.actual_throughput_df = pd.DataFrame() #store the actual throughput data

        if actual_share_throughput_data[0] is not None :
           
            for file in actual_share_throughput_data:
                self.actual_throughput_df = pd.concat([self.actual_throughput_df, pd.read_csv(file)])
        self.results = defaultdict(dict)
        self.excluded_cols = excluded_cols
        
         # Initialize a logger for this class/module
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Predictor initialized with debug logging enabled.")

    def loadModel(self, modelpath):
        #load pickle model
        #load   the model from a file
        if self.args.modeltype != "AutoML" and "AutoML" not in str(modelpath).split("/")[-1]:
            with open(modelpath, 'rb') as file:
                model = pickle.load(file)
            return model
        else:
            h2o.init()
            #print(modelpath)
            import  os
            modelpath = os.path.join(os.getcwd(), str(modelpath))
            #print(modelpath)
            model = h2o.load_model(str(modelpath))
            return model


    def loadData(self, HPworkload, target, randomseed, correlation, n_combination):
        print("processing test data...")
        #load data
        #filter out resnet and mobile which are low util
        self.df = filter_data(data=self.df, workload="", isbatchThroughput=False, custom_col_exclude=self.args.custom_col_exclude, n_combination=self.args.n_combination)
        if self.args.test_file:
            print("input data already splited, no need to split...")
            test_workloads = pd.DataFrame()
            #already split - just need to remove non corrleated columns
            #self.df = self.df.drop(columns=self.excluded_cols)
            #set target
            if target !=  "threadclass" :
                self.df[target] = 0
            for i in range(1, n_combination+1):
                if target == "L2norm":
                    self.df[target] += self.df[f'w{i}throughput'] / self.df[f'w{i}exclusive_throughput']
                elif target == "sum_throughput":
                    self.df[target] += self.df[f'w{i}throughput']
                elif target == "threadclass" :
                    self.df[target] = self.df["Label"]
                    break
                else:
                    print("target not found")
                    exit(1)
    
            
            #get a set of feature colunmn names after removing w1 prefix
            feat_names = [col[2:] for col in self.df.columns if 'w1' in col] 
            print(feat_names)
            #remove throughput  columen since they were used to calculate L2norm
            columns_excluded = []
            for i in range(1, n_combination+1):
                columns_excluded += [f"workload{i}", f"idx{i}", f"w{i}throughput"]
            #columns_excluded = ['workload1', 'workload2', 'idx1', 'idx2'] + ['w1throughput', 'w2throughput']
            columns_included = set(self.df.columns) - set(self.excluded_cols)
            excluded_numerical_cols = []
            for feat in self.excluded_cols:
                workload_idx = [f"w{i}" for i in range(1, n_combination+1)]
                should_be_added =  True
                for w_idx in workload_idx:
                    if w_idx in feat:
                        should_be_added = False
                        break
                if should_be_added: 
                    excluded_numerical_cols.append(feat)
                        
            #excluded_numerical_cols = [feat for feat in self.excluded_cols if 'w1' not in feat and  'w2' not in feat]
           
            excluded_numerical_cols += [f'w{i}throughput' for i in range(1, n_combination+1)]
            print(f"excluded_numerical_cols={excluded_numerical_cols}")
            print(f"columns_included={columns_included}")
            for feat in feat_names:
                #add a new column with the name feat
                #for i in range(1, n_combination+1):
                if f'w1{feat}' in excluded_numerical_cols:
                    print(f'w{i}{feat} excluded')
                    continue
                self.df[feat] = 0
                for i in range(1, n_combination+1):
                    self.df[feat] += self.df[f'w{i}{feat}']
                    columns_excluded.append(f'w{i}{feat}')
                if '%' in feat:
                    self.df[feat] = self.df[feat] / n_combination
                #if f'w1{feat}' in  excluded_numerical_cols or f'w2{feat}' in  excluded_numerical_cols:
                #    print("excluded", f'{feat}')
                #    continue
                
                #self.df[feat] = (self.df[f'w1{feat}'] + self.df[f'w2{feat}'])/2  if '%' in feat else self.df[f'w1{feat}'] + self.df[f'w2{feat}']
                
                #columns_excluded.append(f'w1{feat}')
                #columns_excluded.append(f'w2{feat}')
            print(f"after exclusion = {self.df.columns}")
            #filter out
            if HPworkload !="":

                condition = pd.Series([False] * len(self.df))
                for i in range(1, n_combination+1):
                    condition = condition | self.df[f'workload{i}'].str.startswith(HPworkload)
                test_set = self.df[condition]
                #test_set = self.df[(self.df['workload1'].str.startswith(HPworkload) ) | (self.df['workload2'].str.startswith(HPworkload) )]
            else:
                test_set = self.df
            print(f"filter out testset of {HPworkload} = {len(test_set)}")
            for i in range(1, n_combination+1):
                test_workloads[f"workload{i}"] = test_set[f"workload{i}"]
            #test_workloads["workload1"] = test_set['workload1'] 
            #test_workloads["workload2"] = test_set['workload2']
            print(f"self.exclude_cols={self.excluded_cols}")
            print(f"testworkloads=\n{test_workloads}")
            test_set = test_set.drop(columns=self.excluded_cols)
            #create output dir
            Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
            test_set.to_csv(f"{self.args.output_dir}/test_check_set.csv", index=False)
            test_mean, test_std = test_set[target].mean(), test_set[target].std()
            #normalize
            if target == "threadclass":
                y_test = test_set[target]
                X_test = test_set.drop(target, axis=1)
                X_test = (X_test- X_test.mean()) / X_test.std()
            else:
                test_set = (test_set - test_set.mean()) / test_set.std()
                X_test = test_set.drop(target, axis=1)
                y_test = test_set[target]
            
            
            
            
       
        else:
            #entire set need spliting
            X_train, X_test, y_train, y_test, test_workloads = train_test_custom_split(data=self.df,
                                                                                    test_workload=HPworkload, 
                                                                                    target=target,  
                                                                                    RANDOMSEED=randomseed, 
                                                                                    CORRELATION=correlation)
            
            #get test mean test st
            test_mean, test_std = self.df[target].mean(), self.df[target].std()
        
        return X_test,  y_test, test_workloads, (test_mean, test_std)
        

    def predictPair(self, HPworkload, target, randomseed, correlation, base100=True, rulebase=False):
        self.target = target
    
        X_test, y_test, test_workloads, (test_mean, test_std) = self.loadData( HPworkload=HPworkload, target=target, randomseed=randomseed, correlation=correlation)
        
        
        if self.args.modeltype == "AutoML":
            h2o_test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
            y_pred = self.model.predict(h2o_test)
            y_pred = h2o.as_list(y_pred)
        else:
            y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("num of test samples", len(y_test))
        print("MSE: ", mse)
        print("R2: ", r2)
        self.mse = mse
        self.r2 = r2

        """
        HELPER FUNCTIONS
        """
        
        
        def swapworkloadcol(df):
            print("swapping workload names...")
            # Swap the positions of columns 'A' and 'C'
            df["workload1"], df["workload2"] = df["workload2"], df["workload1"]

        def swap_w1_w2(column_name):
            if column_name.startswith('w1'):
                return column_name.replace('w1', 'w2')
            elif column_name.startswith('w2'):
                return column_name.replace('w2', 'w1')
            return column_name
        
        def check_HProw(workloadPair):
            if workloadPair["workload1"] != HPworkload:
                swapworkloadcol(workloadPair)
            row = self.df[(self.df["workload1"] == workloadPair[0]) & (self.df["workload2"] == workloadPair[1])]
            print(f"row found:\n {row}")
            if row.empty:
                print("reverse curr col...")
                #reverse
                row = self.df[(self.df["workload1"] == workloadPair[1]) & (self.df["workload2"] == workloadPair[0])]
                #rename w1,w2 in exclusive_row
                # Function to swap 'w1_' with 'w2_' and 'w2_' with 'w1_'
                # Apply the function to all column names
                row.columns = [swap_w1_w2(col) for col in row.columns]
            return row
        """
        HELPER FUNCTION ENDS
        """

        """
        PREDICTING INDEX...
        """

        predicted_index = np.argmax(y_pred)
        max_index = np.argmax(y_test)
        #RANDOM BASELINE:
        np.random.seed(randomseed)
        #rand_index = np.random.randint(0, len(y_test))
        

        print(f"test workload pairs: \n {test_workloads}")
        print(f"predicted index: {predicted_index}")
        print(f"max index: {max_index}")
        #print(f"rand index: {rand_index}")

        #get testing data from matching test_workloads
        
        #find best pair by index in test_workloads
        predicted_pair = test_workloads.iloc[predicted_index, :]
        max_pair = test_workloads.iloc[max_index, :]
        #rand_pair = test_workloads.iloc[rand_index, :]

        """
        GET RANDOM by getting mean of all testing samples
        """
        #get all testing data from self.df by matching test_workloads.index
        all_testing_data = self.df.iloc[test_workloads.index, :]
        #get mean of throughput of HPworkload from all_testing_data
        #get the row of HPworkload
        HPrand_throughput, BErand_throughput = 0, 0
        for i, row in all_testing_data.iterrows():
            
            if row["workload1"] == HPworkload :
                #get 
                HPrand_throughput += row["w1throughput"]
                BErand_throughput += row["w2throughput"]
                print("HPadd", row["w1throughput"], row["workload1"],"BEadd", row['w2throughput'], row["workload2"] )
            else:
                HPrand_throughput += row["w2throughput"]
                BErand_throughput += row["w1throughput"]
                print("HPadd", row["w2throughput"], row["workload2"], "BEadd", row['w1throughput'], row["workload1"] )
        HPrand_throughput, BErand_throughput = HPrand_throughput / len(all_testing_data), BErand_throughput / len(all_testing_data)
        
        
        #get sumthroughput
        


        #swap position of maxpair if max_pair[workload1] != HPworkload
        if max_pair["workload1"] != HPworkload:
            swapworkloadcol(max_pair)
  
        if predicted_pair["workload1"] != HPworkload:
            swapworkloadcol(predicted_pair)
            
        
        print("Predicted workload Pair: ", predicted_pair.to_list())
        print("Max workload Pair: ", max_pair.to_list())
        #print("Random workload Pair: ", rand_pair.to_list())
        #get std, mean of sum_throughput from self.df
        #denormalize the y_test
        y_test = (y_test * test_std) + test_mean
        predict_throughput = (y_test.iloc[predicted_index]) 
        rand_sum_throughput = np.mean(y_test)
        print(f"predicted {target}: {predict_throughput}")
        print(f"Exclusive: max of {target}: {np.max(y_test) }")
        print(f"random select =mean of all samples throughput: {rand_sum_throughput}")
        
        

            
        
       
        
        #filter out test_workloads from self.df and store in test_df for later use
        
        #record results in self.results
        #get data row from self.df
        max_row = self.df[(self.df["workload1"] == max_pair[0]) & (self.df["workload2"] == max_pair[1])]
        print(f"max_row found:\n {max_row}")
        self.maxPair = (max_pair[0], max_pair[1])
       
        #reverse all w1, w2 columns
        if max_row.empty:
            print("reverse max...")
            #reverse
            max_row = self.df[(self.df["workload1"] == max_pair[1]) & (self.df["workload2"] == max_pair[0])]
            #rename w1,w2 in exclusive_row
            # Function to swap 'w1_' with 'w2_' and 'w2_' with 'w1_'
            # Apply the function to all column names
            max_row.columns = [swap_w1_w2(col) for col in max_row.columns]
            
        

        #predicted_row = self.df[(self.df["workload1"] == predicted_pair[0]) & (self.df["workload2"] == predicted_pair[1])]
        
        #get 50-50% row with bestShared_row
        """
        bestShared_row = self.actual_throughput_df[(self.actual_throughput_df['workload1'] == max_pair[0]) & (self.actual_throughput_df['workload2'] == max_pair[1])]
        self.BestsharedPair = (max_pair[0], max_pair[1])
        if bestShared_row.empty:
            #reverse
            bestShared_row = self.actual_throughput_df[(self.actual_throughput_df['workload1'] == max_pair[1]) & (self.actual_throughput_df['workload2'] == max_pair[0])]
            bestShared_row .columns = [swap_w1_w2(col) for col in bestShared_row.columns]
         
        print(bestShared_row)
        bestShared_threads = [eval(y) for y in bestShared_row.iloc[:, 2:-1].values[0]]
        """
        #print(f"shared y_values", y_values)
     
        
        predicted_row = self.df[(self.df['workload1'] == predicted_pair[0]) & (self.df['workload2'] == predicted_pair[1])]
        self.predictedPair = (predicted_pair[0], predicted_pair[1])
        #predicted_row = self.actual_throughput_df[(self.actual_throughput_df['workload1'] == predicted_pair[0]) & (self.actual_throughput_df['workload2'] == predicted_pair[1])]
        if predicted_row.empty:
            #reverse
            predicted_row = self.df[(self.df['workload1'] == predicted_pair[1]) & (self.df['workload2'] == predicted_pair[0])]
            #predicted_pair[0], predicted_pair[1] = predicted_pair[1], predicted_pair[0]
            #predicted_row = self.actual_throughput_df[(self.actual_throughput_df['workload1'] == predicted_pair[1]) & (self.actual_throughput_df['workload2'] == predicted_pair[0])]
            predicted_row.columns = [swap_w1_w2(col) for col in predicted_row.columns]


        #get rulbase row from X_test

        if rulebase:
            rulebase_throughput = defaultdict()
            self.rulebase_pair = defaultdict()
            self.Xtest_cols = X_test.columns
            #get the pair with the highest value
            #for each column in X_test, get the pair with the highest value
            #get the index of the highest value in each column
            X_test.reset_index(drop=True, inplace=True)
            for X_col in X_test.columns:
                
                #print("curr column", X_col)
                #print(f"current col \n {X_test[X_col]}")
                if X_col == "exclusive_throughput" or X_col == "_Compute (SM) Throughput" or X_col == "sm%":
                    col_idx = X_test[X_col].idxmax()
                else:
                    col_idx = X_test[X_col].idxmin()
                print(f"max index of {X_col} = {col_idx}")
                
                col_pair = test_workloads.iloc[col_idx, :]
                #
                #reverse worklaod Row if needed
                col_row = check_HProw(col_pair)
                rulebase_throughput[X_col] = (col_row['w1throughput'], col_row['w2throughput'], col_row['w1throughput'] + col_row['w2throughput']) 
                self.rulebase_pair[X_col] = (col_pair[0], col_pair[1])
                #set attribute
                setattr(self, f"{X_col}Pair", col_pair)
                #print("selectedpairs", rulebase_pair[X_col])
                #print(f"throughput of {X_col}", rulebase_throughput[X_col])

                
                
                
                
        
                
            

                         

            
        
        #BUG - 50-50 is None. use stage1df 's 100% first
        if not base100:
            predicted_threads = [eval(y) for y in predicted_row.iloc[:, 2:-1].values[0]]
            #BUG - 50-50 is None 
            #get the next index that is not None
            best_idx, predict_idx = 4, 4
            while best_idx < len(bestShared_threads):
                if bestShared_threads[best_idx][0] is not None and bestShared_threads[best_idx][0] is not None:
                    print(f"choosing bestShared_idx= {best_idx}")
                    break
                best_idx += 1
            while predict_idx < len(predicted_threads):
                if predicted_threads[predict_idx][0] is not None and predicted_threads[predict_idx][0] is not None:
                    print(f"choosing predicted_idx= {predict_idx}")
                    break
                predict_idx += 1
        

            print(f"bestShared_threads- {bestShared_threads}")
        #get w1exclusive_throughput, w2exclusive_throughput from self.df
        #get the run of actual 
        #self.results["workload1"]["Exclusive"] = exclusive_row['w1exclusive_throughput']
        #BUG - too many None values
        #self.results["workload1"]["Best shared with 50-50% MPS thread%"] = bestShared_threads[best_idx][0]
        self.results["workload1"]["Best shared with 100-100% MPS thread%"] = max_row["w1throughput"]
        self.results["workload1"]["Random select collocation"] = HPrand_throughput
        #self.results["workload1"]["stage1-shared 50-50% throughput of predicted pairs"] = predicted_threads[predict_idx][0]
        self.results["workload1"]["stage1-shared 100-100% throughput of predicted pairs"] = predicted_row["w1throughput"]

        #self.results["workload2"]["Exclusive"] = exclusive_row['w2exclusive_throughput']
        #self.results["workload2"]["Best shared with 50-50% MPS thread%"] = bestShared_threads[best_idx][1]  
        self.results["workload2"]["Best shared with 100-100% MPS thread%"] = max_row["w2throughput"]
        self.results["workload2"]["Random select collocation"] = BErand_throughput
        #self.results["workload2"]["stage1-shared 50-50% throughput of predicted pairs"] = predicted_threads[predict_idx][1]
        self.results["workload2"]["stage1-shared 100-100% throughput of predicted pairs"] = predicted_row["w2throughput"]

        #self.results["workload1+workload2"]["Exclusive"] = exclusive_row['w1exclusive_throughput']  + exclusive_row['w2exclusive_throughput']
        #self.results["workload1+workload2"]["Best shared with 50-50% MPS thread%"] = sum(bestShared_threads[best_idx])
        self.results["workload1+workload2"]["Best shared with 100-100% MPS thread%"] = max_row["w1throughput"] + max_row["w2throughput"]
        self.results["workload1+workload2"]["Random select collocation"] = rand_sum_throughput
        #self.results["workload1+workload2"]["stage1-shared 50-50% throughput of predicted pairs"] = sum(predicted_threads[predict_idx])
        self.results["workload1+workload2"]["stage1-shared 100-100% throughput of predicted pairs"] = predicted_row["w1throughput"] + predicted_row["w2throughput"]

        if rulebase:
            #add to self.result for rulebase
            for key in rulebase_throughput:
                self.results["workload1"][f"{key}"] = rulebase_throughput[key][0]
                self.results["workload2"][f"{key}"] = rulebase_throughput[key][1]
                self.results["workload1+workload2"][f"{key}"] = rulebase_throughput[key][2]

        return predicted_pair.to_list()


    def predictAllWorkloads(self, modelpath, HPworkload, X_test, y_test, target, randomseed, correlation, base100=True, rulebase=False, workload_subdir=""):
        #predict all pairs in the testing set. no specifying collocations
        self.target = target
        model = self.loadModel(Path(modelpath))
        self.logger.info(f"predicting with policy {self.args.label_policy}")
        
        self.logger.info(f"X_test = \n{X_test}")
        #create outputdir
        if workload_subdir:
            os.makedirs(f"{self.args.output_dir}/{workload_subdir}", exist_ok=True)
        else:
            os.makedirs(self.args.output_dir, exist_ok=True)
        #logging.info(f"X_test columns = {X_test.columns}")
        #print model attrs
        #logging.info(f"model features = {self.model.best_estimator_.feature_names_in_}")
        #reorder X_test columns to match model features
        #if model has attr of best_estimator_ and feature_names_in_
        if hasattr(model, 'best_estimator_'):
            X_test = X_test[model.best_estimator_.feature_names_in_]

        if hasattr(model, 'feature_names_in_'):
            X_test = X_test[model.feature_names_in_]


        #assert len X_test == y_test
        assert len(X_test) == len(y_test)
        #reset index of X_test
        y_test = y_test.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        if self.args.modeltype == "AutoML" or "AutoML" in str(modelpath).split("/")[-1]:
            h2o_test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
            y_pred = model.predict(h2o_test)
            y_pred = h2o.as_list(y_pred)
        else:
            y_pred = model.predict(X_test)

       
        #save accuracy score to file
        # Open a file to save the result
        if target == "maxthroughput-powercap":
            #print stats for classification
            self.logger.info("predicted classes", y_pred)
            self.logger.info("Accuracy:", accuracy_score(y_test, y_pred))
            self.logger.info("Classification Report:")
            self.logger.info(classification_report(y_test, y_pred))
            feature_importances = pd.Series(model.best_estimator_.feature_importances_, index=X_test.columns)
            self.logger.info("Feature Importances:")
            self.logger.info(feature_importances.sort_values(ascending=False))
            
            if workload_subdir:
                with open(f"{self.args.output_dir}/{workload_subdir}/classification_report_{target}.txt", "w") as f:
                    f.write(classification_report(y_test, y_pred))
                feature_importances.sort_values(ascending=False).to_csv(f"{self.args.output_dir}/{workload_subdir}/feature_importances.csv")
            else:
                with open(f"{self.args.output_dir}/classification_report_{target}.txt", "w") as f:
                    f.write(classification_report(y_test, y_pred))
                feature_importances.sort_values(ascending=False).to_csv(f"{self.args.output_dir}/feature_importances.csv")


        #write y_test, y_pred to file
        # Create a DataFrame with the two columns

        df = pd.DataFrame({
            'y_test': np.array(y_test).flatten(),
            'y_pred': np.array(y_pred).flatten()
        })
        self.logger.info(f"predicted values = \n{df}")
        #add test workload instance to df using args.test_file
        #only add workload1, workload2 column

        if self.args.test_file and not self.args.end_to_end:
            #add workload1, workload2 to df
            test_workloads = pd.read_csv(self.args.test_file)
            #should read from self.args.n_combination to get workload{i} and w{i}_Threadpercent
            workload_cols = [f"workload{i}" for i in range(1, self.args.n_combination +1)]
            threadpercent_cols = [f"w{i}_Threadpercent" for i in range(1, self.args.n_combination +1)]
            test_workloads = test_workloads[workload_cols + threadpercent_cols]
            #test_workloads = test_workloads[["workload1", "workload2", "w1_Threadpercent", "w2_Threadpercent"]]
            df = pd.concat([df, test_workloads], axis=1)
            
        elif self.args.end_to_end and workload_subdir:
            #workload_subdir is a tuple, expand to (workload1, workload2...). add 
            #create list of workload_subdir of len(df)
            #self.logger.debug(f"workload_idx= {[list(workload_subdir)]*len(df)}")
            test_workloads = pd.DataFrame([list(workload_subdir)]*len(df), columns=[f"workload{i}" for i  in range(1, self.args.n_combination +1)])
            df = pd.concat([df, test_workloads], axis=1)
        # Save the DataFrame to a CSV file
        if workload_subdir:
            df.to_csv(f"{self.args.output_dir}/{workload_subdir}/pred_{target}.csv", index=False)
        else:
            df.to_csv(f"{self.args.output_dir}/pred_{target}.csv", index=False)
        # Feature Importance (optional)
        if self.args.modeltype == "AutoML":
            y_test = y_test.squeeze()
            y_pred = y_pred.squeeze()
        self.logger.debug(y_test)
        self.logger.debug(y_pred)
        #save feature importance to file
        mse = mean_squared_error(y_test, y_pred)

        absolute_percentage_errors = np.abs((y_test - y_pred) / y_test) * 100
       
        # Compute the MAPE
        mape = np.mean(absolute_percentage_errors)
        print(f"MAPE: {mape:.2f}%")

        # Compute the 25th, 50th (median), and 75th percentiles
        percentiles = np.percentile(absolute_percentage_errors, [25, 50, 75])
        r2 = r2_score(y_test, y_pred)
        print("num of test samples", len(y_test))
        
        print(f"25th Percentile: {percentiles[0]:.2f}%")
        print(f"50th Percentile (Median): {percentiles[1]:.2f}%")
        print(f"75th Percentile: {percentiles[2]:.2f}%")
        print("MSE: ", mse)
        print("R2: ", r2)
        self.mse = mse
        self.r2 = r2
        #save all metrics to file 
        if workload_subdir: 
            with open(f"{self.args.output_dir}/{workload_subdir}/pred_metrics_{target}.txt", "w") as f:
                f.write(f"MAPE: {mape:.2f}%\n")
                f.write(f"25th Percentile: {percentiles[0]:.2f}%\n")
                f.write(f"50th Percentile (Median): {percentiles[1]:.2f}%\n")
                f.write(f"75th Percentile: {percentiles[2]:.2f}%\n")
                f.write(f"MSE: {mse}\n")
                f.write(f"R2: {r2}\n")
        else:
            with open(f"{self.args.output_dir}/pred_metrics_{target}.txt", "w") as f:
                f.write(f"MAPE: {mape:.2f}%\n")
                f.write(f"25th Percentile: {percentiles[0]:.2f}%\n")
                f.write(f"50th Percentile (Median): {percentiles[1]:.2f}%\n")
                f.write(f"75th Percentile: {percentiles[2]:.2f}%\n")
                f.write(f"MSE: {mse}\n")
                f.write(f"R2: {r2}\n")


        #save mse r2 to file
        return y_pred

        #get throuighput results
        predicted_index = np.argmax(y_pred)
        max_index = np.argmax(y_test)

        self.predictedPair = test_workloads.iloc[predicted_index, :]
        self.maxPair = test_workloads.iloc[max_index, :]
        max_row = self.df[(self.df["workload1"] == self.maxPair[0]) & (self.df["workload2"] == self.maxPair[1])]
        predicted_row = self.df[(self.df["workload1"] == self.predictedPair[0]) & (self.df["workload2"] == self.predictedPair[1])]
        y_test = (y_test * test_std) + test_mean
        predict_throughput = (y_test.iloc[predicted_index]) 
        max_throughput = np.max(y_test)



        if rulebase:
            rulebase_throughput = defaultdict()
            self.rulebase_pair = defaultdict()
            self.Xtest_cols = X_test.columns
            #get the pair with the highest value
            #for each column in X_test, get the pair with the highest value
            #get the index of the highest value in each column
            X_test.reset_index(drop=True, inplace=True)
            for X_col in X_test.columns:
                
                #print("curr column", X_col)
                #print(f"current col \n {X_test[X_col]}")
                if X_col == "exclusive_throughput" or X_col == "_Compute (SM) Throughput" or X_col == "sm%":
                    col_idx = X_test[X_col].idxmax()
                else:
                    col_idx = X_test[X_col].idxmin()
                print(f"max index of {X_col} = {col_idx}")
                
                col_pair = test_workloads.iloc[col_idx, :]
                #
                #reverse worklaod Row if needed
                #col_row = check_HProw(col_pair)
                col_row = self.df[(self.df["workload1"] == col_pair[0]) & (self.df["workload2"] == col_pair[1])]
                rulebase_throughput[X_col] = (col_row['w1throughput'], col_row['w2throughput'], col_row['w1throughput'] + col_row['w2throughput']) 
                self.rulebase_pair[X_col] = (col_pair[0], col_pair[1])
                #set attribute
                setattr(self, f"{X_col}Pair", col_pair)

        self.results["workload1"]["Best shared with 100-100% MPS thread%"] = max_row["w1throughput"]
        #self.results["workload1"]["Random select collocation"] = HPrand_throughput
        #self.results["workload1"]["stage1-shared 50-50% throughput of predicted pairs"] = predicted_threads[predict_idx][0]
        self.results["workload1"]["stage1-shared 100-100% throughput of predicted pairs"] = predicted_row["w1throughput"]

        #self.results["workload2"]["Exclusive"] = exclusive_row['w2exclusive_throughput']
        #self.results["workload2"]["Best shared with 50-50% MPS thread%"] = bestShared_threads[best_idx][1]  
        self.results["workload2"]["Best shared with 100-100% MPS thread%"] = max_row["w2throughput"]
        #self.results["workload2"]["Random select collocation"] = BErand_throughput
        #self.results["workload2"]["stage1-shared 50-50% throughput of predicted pairs"] = predicted_threads[predict_idx][1]
        self.results["workload2"]["stage1-shared 100-100% throughput of predicted pairs"] = predicted_row["w2throughput"]

        #self.results["workload1+workload2"]["Exclusive"] = exclusive_row['w1exclusive_throughput']  + exclusive_row['w2exclusive_throughput']
        #self.results["workload1+workload2"]["Best shared with 50-50% MPS thread%"] = sum(bestShared_threads[best_idx])
        self.results["workload1+workload2"]["Best shared with 100-100% MPS thread%"] = max_throughput
        #self.results["workload1+workload2"]["Random select collocation"] = rand_sum_throughput
        #self.results["workload1+workload2"]["stage1-shared 50-50% throughput of predicted pairs"] = sum(predicted_threads[predict_idx])
        self.results["workload1+workload2"]["stage1-shared 100-100% throughput of predicted pairs"] = predict_throughput

        #dump results to json
       

        return self.predictedPair.to_list()

    def _get_workload_instances(self, test_file):
        test_workloads = pd.read_csv(test_file)
        workload_columns = [f"workload{i}" for i in range(1, self.args.n_combination+1)]
        test_workloads = test_workloads[workload_columns]
        test_workloads = test_workloads.drop_duplicates()
        #transform to list of tuples
        test_workloads = [tuple(x) for x in test_workloads.values]
        return test_workloads
    
    def _get_final_MPS_config(self, y_test_throughput, y_pred_throughput, y_test_power, y_pred_power):
        #get the correct best config by getting max of y_test_throughput within average of y_test_power
        avg_y_test_power = np.mean(y_test_power)
        self.logger.debug(f"avg_y_test_power = {avg_y_test_power}")
        #get idx of y_test_power with value less than avg_y_test_power
        idx_mask = (y_test_power < avg_y_test_power)
        #filter out y_test_throughput with idx
        self.logger.debug(f"idx of < avg_y_test_power = {idx_mask}")
        self.logger.debug(f"y_test_throughput = {y_test_throughput}")
        masked_y_test_power = y_test_power[idx_mask]
        masked_y_test_throughput = y_test_throughput[idx_mask]
        self.logger.debug(f"y_test_throughput after filter mask = {masked_y_test_throughput}")
        #self.logger.debug("idx_mask", idx_mask)
        
        #get idx of np.max(y_test_throughput)
        best_config_idx = np.argmax(masked_y_test_throughput)
        #get the power of best_config
        self.logger.debug(f"best_config_idx = {best_config_idx}")
        best_config = (masked_y_test_throughput.iloc[best_config_idx], masked_y_test_power.iloc[best_config_idx])

        
        #get idx of y_pred_power with value less than avg_y_test_power
        idx = np.where(y_pred_power < avg_y_test_power)[0]
        #filter out y_pred_throughput with idx
        #if idx cannot be found, y_pred_throhgput = nan
        #if idx is a 0 
        if len(idx) == 0:
            self.logger.info("no predicted config lower than avg power")
            y_pred_throughput = np.nan
            pred_config = (np.nan, np.nan)
            pred_actual_config = (np.nan, np.nan)
        else:
            self.logger.debug(f"idx of < avg_y_test_power = {idx}")
            self.logger.debug(f"y_pred_throughput = {y_pred_throughput}")
            y_pred_throughput = y_pred_throughput[idx]
            self.logger.debug(f"y_pred_throughput after filter mask = {y_pred_throughput}")
            pred_config_idx = np.argmax(y_pred_throughput)
            self.logger.debug(f"pred_config_idx = {pred_config_idx}")
            pred_config = (y_pred_throughput.iloc[pred_config_idx], y_pred_power.iloc[pred_config_idx])
            pred_actual_config = (y_test_throughput.iloc[pred_config_idx], y_test_power.iloc[pred_config_idx])
        self.logger.debug(f"y_pred lower than avg power = {y_pred_throughput}")
        #get idx of np.max(y_pred_throughput)
        
        #get the power of pred_config
        
        return best_config, pred_config, pred_actual_config,  avg_y_test_power
        

        #get the predicted config by getting max of y_pred_throughput within average of y_pred_power
       


        

    def predictE2E(self, test_file, 
                            policy, throughput_model, power_model, randomseed=30):
        if policy == "separate_throughputpower_regression" or policy == "power_regression":
            
            #split test workloads by workload instances.
            workload_instances = self._get_workload_instances(test_file)
            self.logger.info(f"workload instances = {workload_instances}")
            #create dummy data loader
            test_dataloader = Dataloader(args=self.args, 
                                    baselineData="",
                                    kernelData="",
                                    shareThroughputData="",
                                    sharePowerData="")
            #load model
            #throughput_model = self.loadModel(Path(throughput_model))
            #power_model = self.loadModel(Path(power_model))
            #iterate throughput workload instances, find corresponding workload instance as testset
            answer_dict = defaultdict(dict)
            for workload_instance in workload_instances:
                self.logger.info(f"testing instance {workload_instance}")
                #use aggregate_instances
                X_test_throughput, y_test_throughput, test_workloads, columns_excluded_throughput, (test_throughput_mean, test_throughput_std) = test_dataloader.aggregate_and_split_train_test(
                        data=test_file, test_workload=workload_instance, mode="test",
                        target= "separate_throughputpower_regression", correlation=self.args.correlation,
                        n_combination=self.args.n_combination)
                self.logger.debug(f"get testset {X_test_throughput}")
                #aggregate power
                X_test_power, y_test_power, test_workloads, columns_excluded_power, (test_power_mean, test_power_std) = test_dataloader.aggregate_and_split_train_test(
                        data=test_file, test_workload=workload_instance, mode="test",
                        target= "power_regression", correlation=self.args.correlation,
                        n_combination=self.args.n_combination)
                #predict with separated model
                y_pred_throughput = self.predictAllWorkloads(modelpath = throughput_model, 
                                          HPworkload="",
                                         X_test=X_test_throughput, 
                                        y_test=y_test_throughput, 
                                        target="separate_throughputpower_regression", 
                                        randomseed=randomseed, 
                                        correlation=self.args.correlation, 
                                        rulebase=self.args.rulebase,
                                        workload_subdir=workload_instance)
                y_pred_power = self.predictAllWorkloads(modelpath = power_model,
                                            HPworkload="",
                                             X_test=X_test_power, 
                                            y_test=y_test_power, 
                                            target="power_regression", 
                                            randomseed=randomseed, 
                                            correlation=self.args.correlation, 
                                            rulebase=self.args.rulebase,
                                            workload_subdir=workload_instance)
                
                #label by hearuistic
                
                (best_throughput, best_power), (y_pred_throughput, y_pred_power), (y_pred_actual_throughput, y_pred_actual_power),  powercap = self._get_final_MPS_config(y_test_throughput, y_pred_throughput, y_test_power, y_pred_power)
                answer_dict[workload_instance] = {"best_throughput": best_throughput, "best_power": best_power, 
                                                  "pred_throughput": y_pred_throughput, "pred_power": y_pred_power, "pred_actual_throughput": y_pred_actual_throughput,
                                                   "pred_actual_power": y_pred_actual_power,"powercap": powercap}
                self.logger.info(f"workload_instance: {workload_instance}")
                
            #save answer_dict to csv
            import csv
            self.logger.info(f"answer_dict = {answer_dict}")
            
            with open(f"{self.args.output_dir}/e2e_results.csv", "w") as f:
                writer = csv.writer(f)
                #create header with workload[i]...
                header = [f"workload{i}" for i in range(1, self.args.n_combination+1)]
                
                header += answer_dict[workload_instances[0]].keys()
                self.logger.debug(f"header = {header}")
                writer.writerow(header)
                self.logger.info(f"answerdict keys = {answer_dict.keys()}")
                
                for key, value in answer_dict.items():
                    if key == 0:
                        continue
                    #expand key from tuple to list
                    #print(f"key = {key}")
                    row = list(key)
                    row += value.values()
                    writer.writerow(row)
            self.logger.info(f"e2e results saved to {self.args.output_dir}/e2e_results.csv")
            return
                        
            #get throughput, power model 
        else:
            raise ValueError(f"policy {policy} not supported")
        #predict
        return 
    def getBestThread(self ,policy, pair):
        self.policy = policy
        
        

        #actual throughput
        is_reversed = False
        actual_shared_row = self.actual_throughput_df[(self.actual_throughput_df['workload1'] == pair[0]) & (self.actual_throughput_df['workload2'] == pair[1])]
        if actual_shared_row.empty:
            #reverse
            actual_shared_row = self.actual_throughput_df[(self.actual_throughput_df['workload1'] == pair[1]) & (self.actual_throughput_df['workload2'] == pair[0])]
            #swap pair 
            is_reversed = True
        print(actual_shared_row)
        y_values = actual_shared_row.iloc[:, 2:-1]
        columns = y_values.columns
        #change into tuples
        y_values = [eval(y) for y in y_values.values[0]]
        if is_reversed:
            y_values = y_values[::-1]
            #swap pair in y_values
            y_values = [(y[1], y[0]) for y in y_values]
        print(f"shared y_values", y_values)
        #df = pd.read_csv('share_steps_stage2.csv')
        #read baseline_steps_stage2.csv to get the 100 throughput of each model

        #get the throughput of the pair
        if is_reversed:
            exclusive_w1 = self.stage2df.loc[(self.stage2df['workload'] == pair[1])]
            exclusive_w2 = self.stage2df.loc[(self.stage2df['workload'] == pair[0])]
        else:
            exclusive_w1 = self.stage2df.loc[(self.stage2df['workload'] == pair[0])]
            exclusive_w2 = self.stage2df.loc[(self.stage2df['workload'] == pair[1])]
        #thread combu
        min_throughput = []

        for x in range(10, 91,10):
            if exclusive_w1[str(x)] is None or exclusive_w2[str(100-x)] is None:
                continue
            mini = min( np.array(exclusive_w1[str(x)]/exclusive_w1["100"]),np.array(exclusive_w2[str(100-x)]/exclusive_w2["100"]))
            min_throughput.append(mini)
        
        #for each column, get the min of the column
        #get max out of min_throughput
        print(f"min_throughput={min_throughput}")
        print(f"actual_throughput={y_values}")
        max_idx, max_throughput = np.argmax(min_throughput) ,np.max(min_throughput)
        

        
        actual_shared_thorughput = y_values[max_idx]
        #BUG - selected 
        print("maxidx", max_idx)
        
        
        print(f"BUG: - {columns[max_idx]} has None value, fallback to closest value...")
        for i in range(max_idx, len(y_values)):
            if y_values[i][0] is not None and y_values[i][1] is not None:
                max_idx = i
                actual_shared_thorughput = y_values[max_idx]
                break
        if y_values[max_idx][0] is None or y_values[max_idx][1] is None:
            #go back from max_idx to 0
            for i in range(max_idx, -1, -1):
                if y_values[i][0] is not None and y_values[i][1] is not None:
                    max_idx = i
                    actual_shared_thorughput = y_values[max_idx]
                    break
            
        print(f"selected {pair[0]} throughput: {actual_shared_thorughput[0]}")
        print(f"selected {pair[1]} throughput: {actual_shared_thorughput[1]}")
        print(f"selected sum throughput: {sum(actual_shared_thorughput)}")
        
        #add predicted results to self.results
        self.results["workload1"][f"stage2 actual throughput with relative {policy}"] = y_values[max_idx][0]
        self.results["workload2"][f"stage2 actual throughput with relative {policy}"] = y_values[max_idx][1]
        self.results["workload1+workload2"][f"stage2 actual throughput with relative {policy}"] = sum(y_values[max_idx])
        

        
        return 
    
    def predictAll_multiinstance_Workloads(self, HPworkload, target, randomseed, correlation, base100=True, rulebase=False):
        #predict all pairs in the testing set. no specifying collocations
        self.target = target
    
        X_test, y_test, test_workloads, (test_mean, test_std) = self.loadData( HPworkload=HPworkload, target=target, randomseed=randomseed, correlation=correlation, n_combination=self.args.n_combination)
        
        
        if self.args.modeltype == "AutoML":
            h2o_test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
            y_pred = self.model.predict(h2o_test)
            y_pred = h2o.as_list(y_pred)
        else:
            y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("num of test samples", len(y_test))
        print("MSE: ", mse)
        print("R2: ", r2)
        self.mse = mse
        self.r2 = r2
        #save mse r2 to file


        #dump results to json
        with open(f"{self.args.output_dir}/mse_r2.txt", 'w') as file:
            file.write(f"MSE: {self.mse}\n")
            file.write(f"R2: {self.r2}\n")
        
        
        

        #return self.predictedPair.to_list()
        predicted_index = np.argmax(y_pred)
        max_index = np.argmax(y_test)

        self.predictedPair = test_workloads.iloc[predicted_index, :]
        self.maxPair = test_workloads.iloc[max_index, :]
        #writepredicedPair, 
        y_test = (y_test * test_std) + test_mean
        predict_throughput = (y_test.iloc[predicted_index]) 
        max_throughput = np.max(y_test)
        #write predict throughput, max throughput, predictedPair, maxPair to file
        with open(f"{self.args.output_dir}/predicted.txt", 'w') as file:
            file.write(f"predictedpair: {self.predictedPair.to_list()}\n")
            file.write(f"maxPair: {self.maxPair.to_list()}\n")
            file.write(f"predicted throughput: {predict_throughput}\n")
            file.write(f"max throughput: {max_throughput}\n")


        if rulebase:
            rulebase_throughput = defaultdict()
            self.Xtest_cols = X_test.columns
            #get the pair with the highest value
            #for each column in X_test, get the pair with the highest value
            #get the index of the highest value in each column
            X_test.reset_index(drop=True, inplace=True)
            #create a file to store rulebase pairs
            with open(f"{self.args.output_dir}/rulebase.txt", 'w') as file:
                #add mean
                rand_sum_throughput = np.mean(y_test)
                file.write(f"Oracle: {max_throughput}\n")
                file.write(f"Random: {rand_sum_throughput}\n")
                #write max throughput
                
            for X_col in X_test.columns:
                #print("curr column", X_col)
                #print(f"current col \n {X_test[X_col]}")
                if X_col == "exclusive_throughput" or X_col == "_Compute (SM) Throughput" or X_col == "sm%":
                    col_idx = X_test[X_col].idxmax()
                else:
                    col_idx = X_test[X_col].idxmin()
                print(f"max index of {X_col} = {col_idx}")
                
                col_pair = test_workloads.iloc[col_idx, :]
                #
                #reverse worklaod Row if needed
                #col_row = check_HProw(col_pair)
                rulebase_throughput[X_col] = y_test.iloc[col_idx]
                
                #set attribute
                
                #open unseen workload file 
                with open(f"{self.args.output_dir}/rulebase.txt", 'a') as file:
                    file.write(f"{X_col}: {rulebase_throughput[X_col]}\n")
                    #file.write(f"{col_pair[0]}:{col_pair[1]}\n")
        return
    
    
    
    
    def plot_results(self, outname, pair):
        print(self.results)
        #convert everything inside to float
        for key in self.results:
            for key2 in self.results[key]:
                self.results[key][key2] = float(np.array((self.results[key][key2])))
        

        # Keys for x-axis
        x = list(self.results.keys())
        

        # Labels for each bar group
        bar_labels = list(next(iter(self.results.values())).keys())

        # Number of bars per group
        n_bars = len(bar_labels)
        print("num of bars", n_bars)

        # X positions for each group
        x_pos = np.arange(len(x))

        # Bar width
        bar_width = 0.05
        # Define the figure and the grid layout
        #fig = plt.figure(figsize=(10, 10))
        #gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 2, 2])
        fig, ax = plt.subplots(figsize=(20, 10))
        # Create subplots
        # Create the subplots
        #ax1 = fig.add_subplot(gs[0:3, 0])  # The first subplot spans the first two rows
        #ax2 = fig.add_subplot(gs[3:5, 0])    # The second subplot spans the last row
        print(bar_labels)
        #label color dict
        colors = ['b', 'orange', 'g']
        #add 20 colors
        colors += ['r', 'c', 'm', 'y', 'pink', 'purple', 'brown', 'olive', 'lime', 'teal', 'lavender', 'tan', 'maroon', 'navy']
        
        color_dict = {label: colors[i] for i, label in enumerate(bar_labels)}
        # Plot each bar
        for i, label in enumerate(bar_labels):
            values = [self.results[key][label] for key in x]
            ax.bar(x_pos + i * bar_width, values, width=bar_width, label=label, color=color_dict[label])

        
        ax.text(x=1, y=20, s=f'{self.maxPair[1]}', ha='center', rotation=90, bbox=dict(facecolor='0.85'))
        ax.text(x=1+ bar_width, y=20, s=f'avg of all collocated workloads', ha='center', rotation=90, bbox=dict(facecolor='0.85'))
        ax.text(x=1+2*bar_width, y=20, s=f'{self.predictedPair[1]}', ha='center', rotation=90, bbox=dict(facecolor='0.85'))
        #add rulebase pairs
        if self.args.rulebase:
            for i, col in enumerate(list(self.Xtest_cols)):
                ax.text(x=1+(i+3)*bar_width, y=20, s=f'{col}:{self.rulebase_pair[col][1]}', ha='center', rotation=90, bbox=dict(facecolor='0.85'))
            #ax.text(x=0+(i+3)*bar_width, y=20, s=f'{self.rulebase_pair[col][0]}', ha='center', rotation=90, bbox=dict(facecolor='0.85'))
            

        #get relative throughput
        relative_sum = {key: self.results['workload1+workload2'][key] / self.results['workload1+workload2']['Best shared with 100-100% MPS thread%'] for key in self.results['workload1+workload2'].keys()}
        print(relative_sum)
        ax.text(x=2, y=20, s=f'baseline = 100%', ha='center', rotation=90, bbox=dict(facecolor='0.85'))
        for i in range(1, len(list(relative_sum.values()))):
            ax.text(x=2+bar_width*i, y=20, s=f'relative gain = {100*list(relative_sum.values())[i]:.2f}%', ha='center', rotation=90, bbox=dict(facecolor='0.85'))
            if i >= 3:
                #cols
                ax.text(x=2+bar_width*i, y=50, s=f'col = {list(self.Xtest_cols)[i-3]}%', ha='center', rotation=90, bbox=dict(facecolor='0.85'))

        # Set x-ticks and labels
        ax.set_xticks(x_pos + bar_width * (n_bars - 1) / 2)
        #set x[0] as primary
        
        ax.set_xticklabels([f"{pair[0]}", "collocated workload", f"aggregated throughput"], rotation=0    )
        #rotate x
       
        #ax.set_xlabel(f'Collocated Workload: \n Ground Truth={self.maxPair[1]}\n random select={self.randPair[1]}\n stage1 prediction={self.predictedPair[1]}', loc='center')
        ax.set_ylabel('Throughput')
        ax.set_title(f'Throughput of {pair[0]} + collocated workload')
        ax.legend()
        ax.grid(alpha=0.7)

        #plot relative throughput of workload1+workload2 in ax2
        # Plot each bar
        """
        relative_sum = self.results['workload1+workload2'] / self.results['workload1+workload2']['Best shared with 100-100% MPS thread%']
        #plot relative sum in ax2
        ax2.bar(x_pos, relative_sum, width=bar_width, label='relative throughput', color='orange')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{pair[0]}", "collocated workload", f"aggregated throughput"], rotation=0)
        ax2.set_xlabel(f'Random {self.exclusivePair[1]}\n random select={self.randPair[1]}\n stage1 prediction={self.predictedPair[1]}', loc='center')
        ax2.set_ylabel('Relative Throughput')
        ax2.set_title(f'Relative Throughput of {pair[0]} + collocated workload')
        ax2.legend()    
        """
        
        plt.tight_layout()
        plt.savefig(outname)

        #dump self.results into a json
        
        with open(f"{outname[:-4]}.json", 'w') as file:
            json.dump(self.results, file)
        # Display the plot
        #save mse,r2
        with open(f"{outname[:-4]}_mse_r2.txt", 'w') as file:
            file.write(f"MSE: {self.mse}\n")
            file.write(f"R2: {self.r2}\n")
        
        return