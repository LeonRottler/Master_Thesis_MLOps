from clearml import Task, Dataset, PipelineDecorator, OutputModel
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, DiscreteParameterRange, \
    HyperParameterOptimizer, GridSearch, RandomSearch


class HPO:
    def __init__(self, train_task_id):
        self.train_task_id = train_task_id

    def run_optimizer(self):
        optimizer = HyperParameterOptimizer(
            base_task_id=self.train_task_id,
            hyper_parameters=[
                UniformParameterRange("General/learning_rate", min_value=0.001, max_value=0.2),
                UniformIntegerParameterRange("General/max_depth", min_value=1, max_value=5),
                UniformIntegerParameterRange("General/n_estimators", min_value=400, max_value=800)
            ],
            objective_metric_title="F1",
            objective_metric_series="f1_score2",
            objective_metric_sign="max",

            execution_queue="services",
            optimization_time_limit=1.0,

            optimizer_class=RandomSearch
        )

        optimizer.set_report_period(1)
        optimizer.start()
        optimizer.set_time_limit(1.0)
        optimizer.wait()
        best_experiment = optimizer.get_top_experiments_details(1)
        optimizer.stop()

        return best_experiment


if __name__ == "__main__":
    hpo = HPO(train_task_id="eef39efb8bcb408e886d773715704d93").run_optimizer()
    print(hpo[0]["task_id"])
    print(hpo[0]["hyper_parameters"])
    print(Task.get_task(hpo[0]["task_id"]).get_models())
