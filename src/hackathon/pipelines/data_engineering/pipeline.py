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

from kedro.pipeline import Pipeline, node
from .nodes import (split_data, 
                    do_nothing, 
                    merge_test_on_train, 
                    return_sparse_recommendation_train,
                    return_feature_sparse_melted)
from hackathon.transformers import (transform_sample_submission, 
                                    reverse_transform_sample_submission)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                do_nothing,
                "sanral_zip_load",
                'sanral_zip_save',
                tags=['extract']
            ),
            node(
                do_nothing,
                "road_segments_zip_load",
                'road_segments_zip_save',
                tags=['extract']
            ),
            node(
                do_nothing,
                "uber_movement_data_zip_load",
                'uber_movement_data_zip_save',
                tags=['extract']
            ),
            node(
                transform_sample_submission,
                "sample_submission_data",
                'transformed_sample_submission_data',
                tags=['transform']
            ),
            node(
                merge_test_on_train,
                ["transformed_sample_submission_data","train_data"],
                'merged_sample_submission_data',
                tags=['merge']),
            node(return_sparse_recommendation_train,
                 ['transformed_sample_submission_data',
                  'train_data'],
                'sparse_data',
                tags=['sparse']),
            node(return_feature_sparse_melted,
                'sparse_data',
                ['transformed_sparse_data_index', 'transformed_sparse_data'],
                tags=['surprise'])
        ]
    )
