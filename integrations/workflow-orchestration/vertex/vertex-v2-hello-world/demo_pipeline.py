# coding: utf-8
import os

from comet_ml import login

import google.cloud.aiplatform as aip
from kfp import compiler, dsl

# Login to Comet if needed
login()


COMET_PROJECT_NAME = "comet-example-vertex-v2-hello-world"


@dsl.component(packages_to_install=["comet_ml"])
def data_preprocessing(a: str = None) -> str:
    import math
    import random
    import time

    import comet_ml

    experiment = comet_ml.start()

    for i in range(60):
        experiment.log_metric("accuracy", math.log(i + random.random()))
        time.sleep(0.1)
    experiment.end()

    return a


@dsl.component(packages_to_install=["comet_ml"])
def model_training(a: str = None) -> str:
    import math
    import random
    import time

    import comet_ml

    experiment = comet_ml.start()

    for i in range(60):
        experiment.log_metric("accuracy", math.log(i + random.random()))
        time.sleep(0.1)
    experiment.end()

    return a


@dsl.component(packages_to_install=["comet_ml"])
def model_evaluation(a: str = None, b: str = None) -> str:
    import math
    import random
    import time

    import comet_ml

    experiment = comet_ml.start()

    for i in range(60):
        experiment.log_metric("accuracy", math.log(i + random.random()))
        time.sleep(0.1)
    experiment.end()

    return a


@dsl.pipeline(name="comet-integration-example")
def pipeline():
    import comet_ml.integration.vertex

    logger = comet_ml.integration.vertex.CometVertexPipelineLogger(
        # api_key=XXX,
        project_name=COMET_PROJECT_NAME,
        # workspace=XXX
        share_api_key_to_workers=True,
    )

    task_1 = logger.track_task(data_preprocessing(a="test"))

    task_2 = logger.track_task(model_training(a=task_1.output))

    task_3 = logger.track_task(model_training(a=task_1.output))

    _ = logger.track_task(model_evaluation(a=task_2.output, b=task_3.output))


if __name__ == "__main__":
    print("Running pipeline")
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="demo_pipeline.json"
    )

    job = aip.PipelineJob(
        display_name="comet-integration-example",
        template_path="demo_pipeline.json",
        pipeline_root=os.getenv("PIPELINE_ROOT"),
        project=os.getenv("GCP_PROJECT"),
        enable_caching=False,
    )

    job.submit()
