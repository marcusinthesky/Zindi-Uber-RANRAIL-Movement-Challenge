from kedro.context import KedroContext, load_context, KedroContextError
from typing import Callable, Any
from kedro.io import AbstractTransformer
from pathlib import Path
import pandas as pd

def transform_sample_submission(data):
    return (data.loc[:,'datetime x segment_id']
                 .str.split(' x ', expand=True)
                 .rename(columns={0:'datetime', 1:'segment_id'})
                 .assign(datetime = lambda x: pd.to_datetime(x.datetime))
                 .assign(prediction = pd.np.nan)
                 .rename(columns = {'segment_id':'road_segment_id'}))

def reverse_transform_sample_submission(data):
    return (data.assign(datetime = lambda d: d.datetime.astype(str),
                         datetime_x_segment_id = lambda d: d.datetime + ' x ' + d.segment_id)
                 .rename(columns={'datetime x segment_id':'datetime_x_segment_id'})
                 .loc[:,['datetime_x_segment_id', 'prediction']]
                 .rename(columns = {'road_segment_id':'segment_id'}))

class SampleSubmissionTransformer(AbstractTransformer):
    def load(self, data_set_name: str, load: Callable[[], Any]) -> Any:
        start = time.time()
        data = load()
        print("Loading {} took {:0.3f}s".format(data_set_name, time.time() - start))
        return (data
                 .loc[:,'datetime x segment_id']
                 .str.split(' x ', expand=True)
                 .rename(columns={0:'datetime', 1:'segment_id'})
                 .assign(datetime = lambda x: pd.to_datetime(x.datetime))
                 .assign(prediction = pd.np.nan))

    def save(self, data_set_name: str, save: Callable[[Any], None], data: Any) -> None:
        start = time.time()
        
        save(data.assign(datetime = lambda d: d.datetime.astype(str),
                         datetime_x_segment_id = lambda d: d.datetime + ' x ' + d.segment_id)
                 .rename(columns={'datetime x segment_id':'datetime_x_segment_id'})
                 .loc[:,['datetime_x_segment_id', 'prediction']])
        print("Saving {} took {:0.3}s".format(data_set_name, time.time() - start))
        
        
# project_context = load_context(Path.cwd(), env=None)

# project_context.catalog.add_transformer(SampleSubmissionTransformer(), data_set_names='sample_submission_data')