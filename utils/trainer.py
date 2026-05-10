import pandas as pd
from utils.loaddata import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import h2o
from h2o.automl import H2OAutoML
import datetime
from pathlib import Path
import os


class Trainer():
    def __init__(self, args) -> None:
        self.args = args
        now = datetime.datetime.now()
        #get day, month, year
        day = now.day
        month = now.month
        year = now.year
        #get time
        current_time = now.strftime("%H:%M:%S")
        self.timestamp = f"{day}-{month}-{year}_{current_time}"
        # Initialize a logger for this class/module
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Dataloader initialized with debug logging enabled.")


    def processData(self, data, target):
        print("processing training data")
        data = pd.read_csv(data)
        
        #data = filter_data(df, workload = "", isbatchThroughput=False, custom_col_exclude=self.args.custom_col_exclude, n_combination=self.args.n_combination)
        #categorical_columns = ['workload1', 'workload2', 'idx1', 'idx2']
        
        #data, columns_excluded = preprocess_data(data=data, categorical_columns=categorical_columns, target='sum_throughput', correlation=0.2)
        #print(f"columns excluded {columns_excluded}")

        self.X_train, self.X_test, self.y_train, self.y_test, self.test_workloads, columns_excluded = train_test_custom_split(data=data,
                                                                                    test_workload="", 
                                                                                    target=target, 
                                                                                    RANDOMSEED=30,
                                                                                    CORRELATION=self.args.correlation,
                                                                                    n_combination=self.args.n_combination
                                                                                    )
        
        
        return  columns_excluded
    
    def train(self, modeltype, X_train, X_test, y_train, y_test):
        #get time
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        if self.args.debug:
            pass
            #X_train_path = f"{self.args.output_dir}/X_train.csv"
            #X_train.to_csv(X_train_path, index=False)
            #y_train_path = f"{self.args.output_dir}/y_train.csv"
            #y_train.to_csv(y_train_path, index=False)
            #y_test_path = f"{self.args.output_dir}/y_test.csv"
            #y_test.to_csv(y_test_path, index=False)
            #X_test_path = f"{self.args.output_dir}/X_test.csv"
            #X_test.to_csv(X_test_path, index=False)
        start_time = datetime.datetime.now()
        if modeltype == "KACE" or modeltype == "linear":
            model = LinearRegression()
            model.fit(X_train, y_train)
        elif modeltype == "NN":
            import tensorflow as tf
            tf.random.set_seed(50)
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            # Define the model
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='linear')  # Output layer for regression
            ])

            model.compile(optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_squared_error'])
            #add dropout layer
            model.add(Dropout(0.2))
            #apply earlystopping
            from tensorflow.keras.callbacks import EarlyStopping
            early_stopping = EarlyStopping(patience=40)
            # Train the model with seeds
            history = model.fit(X_train, y_train, epochs=250, validation_split=0.2, verbose=1, callbacks=[early_stopping])
                        
        elif modeltype == "hotcloud" or modeltype == "RF":
            model = RandomForestRegressor(random_state=42)
            #random search - random select from the combination of hyperparameters
            if self.args.label_policy != "separate_throughputpower_regression":
            
                # Number of trees in random forest
                n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
                # Number of features to consider at every split
                max_features = ['sqrt', 'log2']
                # Maximum number of levels in tree
                max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
                max_depth.append(None)
                # Minimum number of samples required to split a node
                min_samples_split = [2, 5, 10]
                # Minimum number of samples required at each leaf node
                min_samples_leaf = [1, 2, 4]
                # Method of selecting samples for training each tree
                bootstrap = [True, False]
                # Create the random grid
                random_grid = {'n_estimators': n_estimators,
                            'max_features': max_features,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'bootstrap': bootstrap}
                model = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
            
            model.fit(X_train, y_train)

        elif modeltype == "extratrees":
            self.logger.debug(f"extra trees maxdepth : {self.args.max_depth}")
            self.logger.debug(f"extra trees min_samples_split: {self.args.min_samples_split}")
            self.logger.debug(f"extra trees max_features: {self.args.max_features}")
            self.logger.debug(f"extra trees n_estimators: {self.args.n_estimators}")
            self.logger.debug(f"extra trees bootstrap: {self.args.bootstrap}")
            self.logger.debug(f"extra trees min_samples_leaf: {self.args.min_samples_leaf}")
            #modify here
            model = ExtraTreesRegressor(random_state=42, 
                                        max_depth=self.args.max_depth, 
                                        min_samples_split=self.args.min_samples_split,
                                        max_features=self.args.max_features,
                                        n_estimators=self.args.n_estimators,
                                        bootstrap=self.args.bootstrap,
                                        min_samples_leaf=self.args.min_samples_leaf)
             
            model.fit(X_train, y_train)

        elif modeltype == "AutoML":
            assert(len(X_train) == len(y_train))
            h2o.init()
            # Reset index to avoid misalignment issues
            #save y_train to a file
            #y_train.to_csv(f"{self.args.output_dir}/y_train_beforereset.csv", index=False)
            #X_train.to_csv(f"{self.args.output_dir}/X_train_beforereset.csv", index=False)
            
            h2o_train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
            h2o_test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
            #save h2o_train to a file
            h2o_train_path = f"{self.args.output_dir}/h2o_train.csv"
            h2o_train.as_data_frame().to_csv(h2o_train_path, index=False)
            #save X_test to a file
            #save pd.concat([X_train, y_train.to_frame() for debug
            #place y_train to the last column of X_train
            
            

            # Identify predictors and response
            self.logger.info(f"h2o_train shape before train: {h2o_train.shape}")
            x = h2o_train.columns
            self.logger.info(f"x columns: {x}")
            #y = "sum_throughput"
            if self.args.label_policy not in  ["maxthroughput-powercap", "separate_throughputpower_regression", "power_regression"]:
                y = "sum_throughput"
            else:
                y = self.args.label_policy
            x.remove(y)
            #train the model
            aml = H2OAutoML(max_models=30, seed=30, 
                            stopping_metric='mse',     # Early stopping based on MAE
                            sort_metric='mse', max_runtime_secs=3*60)
            aml.train(x=x, y=y, training_frame=h2o_train)
            model = aml.leader
            import os
            os.makedirs(self.args.output_dir, exist_ok=True)
            #join getcwd with output_dir

            model_path = h2o.save_model(model=model, path=f"{str(Path(os.getcwd()/self.args.output_dir))}", force=True)
            self.logger.info(f"Best model saved to: {model_path}")
        elif modeltype == "threadclass":
            model = RandomForestClassifier(random_state=42)
            #random search - random select from the combination of hyperparameters
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 50)]
            # Number of features to consider at every split
            max_features = ['sqrt', 'log2']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
            model = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
            model.fit(X_train, y_train)
        else:
            raise ValueError("modeltype not supported")
        
        
        end_time = datetime.datetime.now()
        process_time = str(end_time-start_time)
        str_excol = "_".join(self.args.custom_col_exclude)

        #save end_time-start_time to a file
        import os# Get the parent directory from the output file path
        # Create parent directory if it doesn't exist
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(f'{self.args.output_dir}/{self.timestamp}_{modeltype}_model-corr{self.args.correlation}_datard{self.args.split_randomseed}_excol{str_excol}_train_time.txt', 'w') as f:
            f.write("Training time\n")
            f.write(f"{process_time}\n")
        
        if  modeltype != "AutoML":
            with open(f'{self.args.output_dir}/{self.timestamp}_{modeltype}_model-corr{self.args.correlation}_datard{self.args.split_randomseed}_excol{str_excol}.pkl', 'wb') as file:
                pickle.dump(model, file)



        #get MAPE of predicting X_train
        if modeltype != "AutoML":
            y_train_pred = model.predict(X_train)
        else:
            y_train_pred = model.predict(h2o_train)
            #y_train_pred = h2o.as_list(y_train_pred)
            y_train_pred = model.predict(h2o_train).as_data_frame().values.flatten()
        y_train = y_train.squeeze()
        y_train_pred = y_train_pred.squeeze()
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        assert len(y_train) == len(y_train_pred), f"Mismatch: y_train({len(y_train)}) vs y_train_pred({len(y_train_pred)})"
        absolute_percentage_errors = np.abs((np.array(y_train) - np.array(y_train_pred)) / np.array(y_train)) * 100
        # Compute the MAPE
        mape = np.mean(absolute_percentage_errors)
        mse = mean_squared_error(y_train, y_train_pred)
        r2 = r2_score(y_train, y_train_pred)
        percentiles = np.percentile(absolute_percentage_errors, [25, 50, 75])
        print(f"MAPE on train data: {mape:.2f}%")
        print(f"mse on train data: {mean_squared_error(y_train, y_train_pred)}")
        print(f"r2 on train data: {r2_score(y_train, y_train_pred)}")
        #save mse,r2 to a csv file
        with open(f'{self.args.output_dir}/train_metrics_{self.args.label_policy}.txt', "w") as f:
            f.write(f"MAPE: {mape:.2f}%\n")
            f.write(f"25th Percentile: {percentiles[0]:.2f}%\n")
            f.write(f"50th Percentile (Median): {percentiles[1]:.2f}%\n")
            f.write(f"75th Percentile: {percentiles[2]:.2f}%\n")
            f.write(f"MSE: {mse}\n")
            f.write(f"R2: {r2}\n")
        ###############################
        #prediction metrics on test data
        ###############################
        #load   the model from a file
        #with open(f'output/trained_models/{self.timestamp}linear_regression_model.pkl', 'rb') as file:
        #    model = pickle.load(file)
        if modeltype != "AutoML":
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(h2o_test)
            y_pred = h2o.as_list(y_pred)
        
        
        
        y_test = y_test.squeeze()
        y_pred = y_pred.squeeze()
        absolute_percentage_errors = np.abs((np.array(y_test) - np.array(y_pred)) / np.array(y_test)) * 100
        assert(len(y_test) == len(y_pred))
        # Compute the MAPE
        mape = np.mean(absolute_percentage_errors)
        print(f"MAPE on validation: {mape:.2f}%")

        # Compute the 25th, 50th (median), and 75th percentiles
        percentiles = np.percentile(absolute_percentage_errors, [25, 50, 75])
        print(f"25th percentile: {percentiles[0]:.2f}%")
        print(f"50th percentile (median): {percentiles[1]:.2f}%")
        print(f"75th percentile: {percentiles[2]:.2f}%")

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'MSE: {mse}')
        print(f'R^2: {r2}')
        
        #save mse,r2 to a csv file
        with open(f'{self.args.output_dir}/{self.timestamp}mse_r2-corr{self.args.correlation}_datard{self.args.split_randomseed}_excol{str_excol}.csv', 'w') as f:
            f.write("mse,r2\n")
            f.write(f"{mse},{r2}\n")

        return Path(f"{self.args.output_dir}/{self.timestamp}_{modeltype}_model-corr{self.args.correlation}_datard{self.args.split_randomseed}_excol{str_excol}.pkl")