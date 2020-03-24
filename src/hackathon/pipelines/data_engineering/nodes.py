# Copyright 2018-2019 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""

from typing import Any, Dict

import pandas as pd


def do_nothing(inputs, *args):
    if args:
        output = inputs, *args
    else:
        output = inputs
        
    return output


def merge_test_on_train(transformed_sample_submission_data, train_data):        
    return  (transformed_sample_submission_data
              .rename(columns={'datetime':'Occurrence Local Date Time'})
              .merge(train_data.loc[:,['road_segment_id',
                                       'longitude',
                                       'latitude']],
                     on=['road_segment_id'])
             )

def return_sparse_recommendation_train(transformed_sample_submission_data,
                                       train_data):
    test = (transformed_sample_submission_data
        .rename(columns={'datetime':'Occurrence Local Date Time'})
        .loc[:,['Occurrence Local Date Time','road_segment_id']]
        .drop_duplicates()
        .assign(prediction=0))

    train = (train_data
            .loc[:,['Occurrence Local Date Time','road_segment_id']]
            .assign(prediction=1)
            .groupby(['Occurrence Local Date Time','road_segment_id']).sum()
            .reset_index())


    long = pd.concat([train, test])

    unique = (long
              .groupby(['Occurrence Local Date Time','road_segment_id']).sum()
              .reset_index())

    sparse = unique.pivot(index='Occurrence Local Date Time', 
                          columns='road_segment_id', 
                          values='prediction').fillna(0)
    
    return sparse

def return_feature_sparse_melted(sparse_data: pd.DataFrame):
    new_index = (pd.Series(sparse_data.index).dt.dayofweek.astype(str) + 
             pd.Series(sparse_data.index).dt.hour.astype(str))

    index_lookup = pd.concat([pd.Series(new_index, name='DOW'),
                              pd.Series(sparse_data.index)], axis=1).set_index('Occurrence Local Date Time')

    new_sparse_data = (sparse_data
                       .set_index(new_index)
                       .reset_index()
                       .groupby('Occurrence Local Date Time').sum())

    melted = (new_sparse_data
              .reset_index()
              .melt('Occurrence Local Date Time')
              .assign(value = lambda x: x.value/x.value.max()))

    return index_lookup, melted

    

def split_data(data: pd.DataFrame, example_test_data_ratio: float) -> Dict[str, Any]:
    """Node for splitting the classical Iris data set into training and test
    sets, each split into features and labels.
    The split ratio parameter is taken from conf/project/parameters.yml.
    The data and the parameters will be loaded and provided to your function
    automatically when the pipeline is executed and it is time to run this node.
    """
    data.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]
    classes = sorted(data["target"].unique())
    # One-hot encoding for the target variable
    data = pd.get_dummies(data, columns=["target"], prefix="", prefix_sep="")

    # Shuffle all the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split to training and testing data
    n = data.shape[0]
    n_test = int(n * example_test_data_ratio)
    training_data = data.iloc[n_test:, :].reset_index(drop=True)
    test_data = data.iloc[:n_test, :].reset_index(drop=True)

    # Split the data to features and labels
    train_data_x = training_data.loc[:, "sepal_length":"petal_width"]
    train_data_y = training_data[classes]
    test_data_x = test_data.loc[:, "sepal_length":"petal_width"]
    test_data_y = test_data[classes]

    # When returning many variables, it is a good practice to give them names:
    return dict(
        train_x=train_data_x,
        train_y=train_data_y,
        test_x=test_data_x,
        test_y=test_data_y,
    )



