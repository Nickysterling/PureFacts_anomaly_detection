from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
import pandas as pd
import numpy as np
import os
import pdb


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
    # only use 10% of the data for now
    num_rows = len(merged_df)
    # Get the first 10% of rows
    first_20_percent = merged_df.head(int(num_rows*0.2))
    return first_20_percent


# Checks if AssignedBPSRate column is not between max and min bps rate
def is_outside_max_min_bps(row):
    if row['AssignedBPSRate'] >= row['MinBPSRate'] and row['AssignedBPSRate'] <= row['MaxBPSRate']:
        return False
    return True


# check that for a decrease in AssignedBPSRate if the tiers are increasing for each FeeScheduleAssignID and return each row
def check_tiers(data):
    # get all the FeeScheduleAssignIDs
    fee_schedule_assign_ids = data['FeeScheduleAssignID'].unique()
    data['tier_anomaly'] = 0
    for fee_schedule_assign_id in fee_schedule_assign_ids:
        fee_schdule_anonmaly = False
        # get all the rows for a FeeScheduleAssignID
        rows = data[data['FeeScheduleAssignID'] == fee_schedule_assign_id]
        # sort the rows by AssignedBPSRate
        rows = rows.sort_values(by=['StartTier'])
        # Ignore rows where tier is arbitrarily large
        if rows['EndTier'].unique()[0] >= 99999999999 and rows['StartTier'].unique()[0] <= 0:
            continue
        # check if bps rate is monotonic decreasing when start tier is increasing
        for i in range(rows.index[0], rows.index[-1]):
            if i == rows.index[0]:
                continue
            elif (rows['EndTier'][i] >= rows['EndTier'][i-1] or rows['StartTier'][i] >= rows['StartTier'][i-1]) and rows['AssignedBPSRate'][i] > rows['AssignedBPSRate'][i-1]:
                fee_schdule_anonmaly = True
                break
        if fee_schdule_anonmaly:
            data.loc[data['FeeScheduleAssignID'] == rows['FeeScheduleAssignID'].iloc[0], 'tier_anomaly'] = 1
    print(len(data[data['tier_anomaly'] == 1]), 'rows have BPS rate that does not follow the tier pattern') 
    color = ['red' if label == 1 else 'yellow' for label in data['tier_anomaly']]
    plt.scatter(data['StartTier'], data['AssignedBPSRate'], c=color)
    plt.xlabel('Start Tier')
    plt.ylabel('Assigned BPS Rate')
    plt.show()
    return data


# finds the most important features that capture the most information in the data and reduces the dimensionality of the data
# so that it can be visualized in a 2D scatter plot
def PCA_rearrange(data, pred):
    # instantiate the PCA class with the number of components you want to keep
    pca = PCA(n_components=3)
    # fit the PCA model to your data
    pca.fit(data)
    # transform the data using the PCA model
    data_pca = pca.transform(data)
    # Access the explained_variance_ratio_ attribute
    explained_variance_ratios = pca.explained_variance_ratio_
    # Print the explained variance ratios
    print(explained_variance_ratios)
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=pred)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()


def find_eps_value_with_optics(data):
    # Create an instance of the OPTICS class
    optics = OPTICS(eps=0.2, min_samples=10)
    # Fit the model to the data
    optics.fit(data)
    # Get the reachability plot
    reachability = optics.reachability_
    # Plot the reachability plot
    plt.plot(reachability)
    plt.show()


def find_bps_out_of_range(data):
    data['outside_min_max'] = data.apply(lambda row: is_outside_max_min_bps(row), axis=1)
    print((len(data[data['outside_min_max'] == True])/len(data))*100,
          '% of rows have a BPS rate outside of its max and min BPS rate')
    color = ['yellow' if x == False else 'red' for x in data['outside_min_max']]
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=color)
    plt.xlabel('Min BPS Rate')
    plt.ylabel('Max BPS Rate')
    plt.show()
    return data


def dbscan_tiers(data):
    data = data[['StartTier', 'EndTier',
                 'AssignedBPSRate', 'FeeScheduleAssignID']]
    # eps is the maximum distance between two data pointsin the same cluster,
    # while min_samples is the minimum number of data points in a cluster.
    model = DBSCAN(eps=25, min_samples=5)
    # fit the model to the data and predict cluster labels
    pred = model.fit_predict(data)
    num_anomalies = len(data[pred == -1])
    print(num_anomalies,
          'rows have been predicted as anomalies through DBSCAN', (num_anomalies/len(data))*100, "% of rows are anomalies")
    # create a scatter plot of colored by cluster label
    color = ['red' if label == -1 else 'yellow' for label in pred]
    plt.scatter(data.iloc[:, 0].astype(int), data.iloc[:, 1].astype(int), c=color)
    plt.xlabel('Start Tier')
    plt.ylabel('End Tier')
    plt.show()

def dbscan_fee_group(data):
    data = data[['AssignedBPSRate', 'FeeScheduleAssignID', 'FeeGroupID']]
    # eps is the maximum distance between two data pointsin the same cluster,
    # while min_samples is the minimum number of data points in a cluster.
    model = DBSCAN(eps=25, min_samples=5)
    # fit the model to the data and predict cluster labels
    pred = model.fit_predict(data)
    num_anomalies = len(data[pred == -1])
    print(num_anomalies,
          'rows have been predicted as anomalies through DBSCAN', (num_anomalies/len(data))*100, "% of rows are anomalies")
    # create a scatter plot of colored by cluster label
    color = ['red' if label == -1 else 'yellow' for label in pred]
    plt.scatter(data.iloc[:, 1], data.iloc[:, 2], c=color)
    plt.xlabel('Fee ScheduleAssign ID')
    plt.ylabel('Fee Group ID')
    plt.show()


def dbscan_group_value(data):
    data = data[['CurrentGroupValue', 'AssignedBPSRate']]
    # Drop rows with NaN values
    data = data.dropna(subset=['CurrentGroupValue'])

    # eps is the maximum distance between two data points in the same cluster,
    # while min_samples is the minimum number of data points in a cluster.
    model = DBSCAN(eps=25, min_samples=5)
    # fit the model to the data and predict cluster labels
    pred = model.fit_predict(data)
    num_anomalies = len(data[pred == -1])
    # print("Length of data after dropping Nan CurrentGroupValue rows", len(data))
    print(num_anomalies,
          'rows have been predicted as anomalies through DBSCAN', 
          (num_anomalies/len(data))*100, "% of rows are anomalies")
    # create a scatter plot of colored by cluster label
    color = ['red' if label == -1 else 'yellow' for label in pred]
    plt.scatter(data.iloc[:, 0].astype(int), data.iloc[:, 1], c=color)
    plt.xlabel('Group Value')
    plt.ylabel('BPS Rate')
    plt.show()


def main():
    data = get_merged_data()
    # normalising data
    data = data.where((pd.notnull(data)), None)
    print('Number of rows in data:', len(data))
    # CurrentGroupValue can have NaN values so does not work well in clustering model
    # add in FeeGroupID to see if it can be used to predict anomalies?
    data = data[['MinBPSRate', 'MaxBPSRate', 'StartTier', 'EndTier', 'AssignedBPSRate', 
    'FeeScheduleAssignID', 'IsTieredRate', 'FeeGroupID', 'CurrentGroupValue', "FeeTypeID"]]
    data = find_bps_out_of_range(data)
    data = check_tiers(data)
    dbscan_group_value(data)
    # dbscan_tiers(data)
    dbscan_fee_group(data)


if __name__ == '__main__':
    main()
