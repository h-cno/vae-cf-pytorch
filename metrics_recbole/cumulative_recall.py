# from recbole.evaluator.base_metric import AbstractMetric
from recbole.evaluator.metrics import Recall
from recbole.utils import EvaluatorType


class CumlativeRecall(Recall):
    
    # 前のラウンドのtopkのindexをdataobjectに加える
    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("recall", result)
        return metric_dict

    def used_info(self, dataobject):
        