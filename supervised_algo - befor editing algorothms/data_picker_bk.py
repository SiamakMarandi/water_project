import os
import glob
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from gaussrank import *
import logging
import pandas as pd
import numpy as np
import dataset
import evaluator
import visualiser
# import SVM_Regressor
import sys
#sys.path.insert(1, 'H:/Project/water_project/dataset')
import dataset 

def calculator(clf, dataset, dId_list, year, month, day, hour, computation_range, what_hour, dId, day_type):
    print("current directory : ", os.getcwd())
    cur_dir = os.getcwd()
    if not os.path.isdir(f"{cur_dir}/csv"):
        print('The directory is not present. Creating a new one..')
        os.mkdir(f"{cur_dir}/csv")
    else:
        files = glob.glob(f"{cur_dir}/csv/*.csv")

        for f in files:
            try:
                #f.unlink()
                os.remove(f)
            except OSError as e:
                print("no file exist ")
        # os.remove(f"{cur_dir}/csv/*.csv")  
    
    path = ("{dir}/csv".format(dir=cur_dir))
    result_df_final = pd.DataFrame(columns=["DeviceId", "What Hour", "Computation Range", "Predicted Water Consumtion", "Mean Absolout Error"])
    # dataset.drop(["Day_of_Week", "Is_weekend"],axis=1, inplace=True)   
    # print("dataset : ", dataset)
    # ========================
    # result_df = pd.DataFrame(columns=["DeviceId", "What Hour", "Computation Range", "predicted Water Consumtion", "Mean Absolout Error"])
    i = 0
    for dId in dId_list:
        result_df = pd.DataFrame(columns=["DeviceId", "What Hour", "Computation Range", "Predicted Water Consumtion", "Mean Absolout Error"])
        print("Device ID : ", dId)
        
        df_filtered = dataset[dataset['DeviceId'] == dId]
        df_filtered.reset_index(inplace=True, drop=True)
        # print("df_filtered : ", df_filtered)      

        y_df_filtered = df_filtered.loc[:, ["Value"]]
        x_df_filtered = df_filtered.loc[:, ["DeviceId", "Day", "Month", "Year", "hour", "Day_of_Week", "Is_weekend"]]
        for duration in computation_range:
            for wh in what_hour:
                try : 
                    print("what hour : ", wh)
                    indexHour = x_df_filtered[(x_df_filtered['Year'] == year) & (x_df_filtered['Month']== month) & 
                    (x_df_filtered["hour"] == hour) & (x_df_filtered["Day"] == day)].index
                    # print("df_filtered : ", x_df_filtered)
                    # print("df_filtered : ", y_df_filtered)
                    
                    print("indexHour : ", indexHour[0])
                    
                    ohe = OneHotEncoder(sparse=False)
                    x_df_filtered_ohe = ohe.fit_transform(x_df_filtered)
                    # indexNames = dataset[(dataset["DeviceId"] == dId)].index
                    # print("indexNames : ", indexHour)

                    # final_dataset = df_filtered.iloc[indexHour - computation_days + what_hour, computation_days]
                    start_index = indexHour[0] - (duration + wh)
                    # print("start_index : ", start_index)
                    x_dataset = x_df_filtered_ohe[start_index : start_index + duration]
                    # print("x_dataset : ", x_dataset)
                    # ===============
                    y_dataset = y_df_filtered[start_index : start_index + duration]
                    # ===============================
                    # print("X : ", X)
                    # print("y : ", y)

                    x_cols = y_df_filtered.columns[:]
                    x = y_df_filtered[x_cols]

                    s = GaussRankScaler()
                    x_ = s.fit_transform(x)
                    assert x_.shape == x.shape
                    y_df_filtered[x_cols] = x_
                    # ===============
                    # print('Number of data points in train data:', x)
                    #-----------------------------------Categorical to Binary-----------------------------------------

                    # Train and Test (x,y) / shuffle false because of importance roll of date in our study----------------------
                    # train_x, test_x, train_y, test_y = train_test_split(X_ohe, y, stratify=y, test_size=0.3, shuffle=False)
                    # #################################
                    x_predict = x_df_filtered_ohe[indexHour[0]]
                    print("x_predict : ", x_predict)
                    y_predict = y_df_filtered.iloc[indexHour[0]]
                    x_predict = x_predict.reshape(1,-1)
                    y_predict = y_predict.to_frame() 

                    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, shuffle=False, test_size=0.2, random_state=42)

                    x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, shuffle=False, test_size=0.2, random_state=42)
                    clf.fit(x_train, y_train)
                    evaluation_dict = evaluator.evaluate_preds(clf, x_train, y_train, x_test, y_test, x_cv, y_cv, x_predict)
                    # visualiser.plotter(clf, x_train, y_train, x_test, y_test)
                    result_df.at[i, 'DeviceId'] = dId
                    result_df.at[i, "What Hour"] = wh
                    result_df.at[i, "Computation Range"] = duration
                    result_df.at[i, 'Predicted Water Consumtion'] = evaluation_dict["predicted_value"][0]
                    result_df.at[i, 'Mean Absolout Error'] = evaluation_dict["mean_absolute_error"]


                    i = i + 1
                except Exception as e:
                    # logging.error("something went wrong", exc_info=e)
                    print("there is no value for this device ID : ", dId)
        
        print("path  :  ", path)
        result_df.to_csv(f'{path}\\result_{dId}.csv', index = False)
        # print("result min   :   ", result_df["Mean Absolout Error"].min())        
        min_row = result_df[result_df["Mean Absolout Error"] == result_df["Mean Absolout Error"].min()]
        # print("min row : ", min_row)
        result_df_final = pd.concat([result_df_final,min_row], axis=0)
        # print("final result : ", result_df_final)
       

    result_df_final = result_df_final.reset_index() 
    result_df_final.drop(["index"], axis=1, inplace=True)
    result_df_final = result_df_final.drop_duplicates(subset=['DeviceId'])
    result_df_final.to_csv(f'{path}\\result_final.csv', index = False)

    print(day_type)
    print("final mean : \n", round(result_df_final.mean(), 4))
    print("final sum : \n", round(result_df_final.sum(), 4))
    # print("result : ", result_df_final)
    # pk = result_df_final.drop(["DeviceId", "What Hour", "Predicted Water Consumtion"], axis=1)  
    df = pd.read_csv(f'csv\\result_{dId}.csv')      
    # print("df : ", df)   
    msg = "Chart of miminum MAE of single device ID" 
    df = df.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)
    visualiser.computation_range_plotter(df, msg)

    df = pd.read_csv(f'csv\\result_final.csv')
    # print("df   :   ", df)
    msg = "Chart of miminum MAE of all device ID"
    df = df.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)
    visualiser.computation_range_plotter(df, msg)

    pk = result_df_final.drop(["DeviceId", "What Hour", "Predicted Water Consumtion"], axis=1)
    df = result_df_final.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)   
    
    gk = pk.groupby(['Computation Range'], axis=0).count()    
    gk = pk.groupby(['Computation Range'], axis=0).sum()   
    gk.groupby(['Computation Range'], axis=0).sum().plot(kind="line", linewidth='2',
                label='MAE',marker="o",
                markerfacecolor="red", markersize=10)
    
    # print("sum : \n", gk)
    # plt.plot(gk["Computation Range"],gk["Mean Absolout Error"],'b-o',label='Accuracy over batch size for 1000 iterations');
    plt.xlabel('Computation Range')
    plt.ylabel('Mean Absolout Error')
    plt.title("Chart of sum of miminum MAE of all device ID")
    plt.legend()
    plt.show()



    
    # print("result df", result_df)
        
    


