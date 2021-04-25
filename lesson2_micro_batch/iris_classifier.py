import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class IrisMicroBatchClassifier(BentoService):
    """
    A minimum prediction service exposing a Scikit-learn model
    """

    @api(
        input=DataframeInput(),
        mb_max_latency=10000, 
        mb_max_batch_size=1000,
        batch=True)
    def predict(self, df: pd.DataFrame):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        return self.artifacts.model.predict(df)