import xgboost as xgb
import numpy as np
from clearml import Task, StorageManager
import pickle


class Preprocess(object):
    def __init__(self):
        label_encoder_task = Task.get_task("778a88dc010e4df690c84077772a3ec2")
        artifact_name = 'label_encoder'
        artifact_url = str(label_encoder_task.artifacts[artifact_name].get("url"))
        encoder_path = str(StorageManager.get_local_copy(artifact_url))

        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)

    def preprocess(self, body, state, collect_custom_statistics_fn=None):
        try:
            return xgb.DMatrix([list(body.values())])
        except ValueError as e:
            print(f"Failed to parse JSON: {e}")
            return None

    def postprocess(self, data, state: dict, collect_custom_statistics_fn=None):
        if isinstance(data, np.ndarray):
            if data.ndim > 1 and data.shape[1] > 1:
                predicted_class_index = np.argmax(data, axis=1)[0]
                predicted_label = self.encoder.inverse_transform([predicted_class_index])[0]
                return dict(y=predicted_label)
            else:
                predicted_label = self.encoder.inverse_transform(data.flatten())[0]
                return dict(y=predicted_label)
            return dict(y=labels.tolist())
        else:
            return dict(y=data)
