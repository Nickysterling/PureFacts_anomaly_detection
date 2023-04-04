import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pdb
import os


# This function should check to make sure the AssignedBPSRate is between the min and max BPS rates
# def check_between_min_max(arr):


"""
This function merges the data from the fee_group_details.csv and fee_schedule_assignment_details.csv into one dataframe.
"""


def get_merged_data():
    group_details = pd.read_csv(os.path.join('data', 'fee_group_details.csv'))
    sch_assign_details = pd.read_csv(os.path.join(
        'data', 'fee_schedule_assignment_details.csv'))
    sch_assign = pd.read_csv(os.path.join(
        'data', 'fee_schedule_assignment.csv'))
    # Merge data into one df using the mapping data file, fee_schedule_assignment.csv
    mapping = sch_assign[['FeeScheduleAssignID', 'FeeGroupID']]
    # merge the mapping onto the sch_assign_details to add the FeeGroupID column
    sch_assign_details_w_fee_group_ids = sch_assign_details.merge(
        mapping, how='inner', on='FeeScheduleAssignID')
    # Now since the result from above has a FeeGroupID, and the fee group data file does too, we can merge on that
    merged_df = sch_assign_details_w_fee_group_ids.merge(
        group_details, how='inner', on='FeeGroupID')
    return merged_df


"""
Should use PCA to reduce the dimensions to 2d or 3d and you can visualize points in space that way. 
Plotly is great for making interactive 3d plots. You can use the color of points to show the clusters 
found with different algorithms.
"""
# def visualize_data(data):

"""
Very simple isolation forest approach. Just wanted to see if it works and what the output is.
We will heavily refactor this later and maybe use a different approach, but this is a good start.
"""
if __name__ == "__main__":
    data = get_merged_data()
    # split into 80% train and 20% test
    train = data.sample(frac=0.8, random_state=0)
    test = data.drop(train.index)
    # Random_state is used to set the seed for the random generator so that we can ensure that the results that we get can be reproduced
    random_state = np.random.RandomState(42)
    # Create the isolation forest model
    model = IsolationForest(n_estimators=100, max_samples='auto',
                            contamination=float(0.2), random_state=random_state, verbose=1)
    model.fit(train[['AssignedBPSRate']])
    # Add the prediction to the data
    test['scores'] = model.decision_function(test[['AssignedBPSRate']])
    # Add the anomaly column to the data
    test['anomaly_score'] = model.predict(test[['AssignedBPSRate']])
    # Get the outliers
    anomalies = test[test['anomaly_score'] == -1].head()
    print('anomalies found:', anomalies)
