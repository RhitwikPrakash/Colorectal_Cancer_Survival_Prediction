from kfp import dsl
from kfp.dsl import Input, Output, Dataset, component

# ✅ DATA PROCESSING COMPONENT
@component(
    base_image="gonudocker/cancer-survival-prediction:latest"
)
def data_processing_op(processed_data: Output[Dataset]):
    import pandas as pd
    from pathlib import Path

    df = pd.DataFrame({'age': [65, 72], 'tumor_size': [5.4, 7.2]})
    output_path = Path(processed_data.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_path)


# ✅ MODEL TRAINING COMPONENT
@component(
    base_image="gonudocker/cancer-survival-prediction:latest"
)
def model_training_op(processed_data: Input[Dataset]):
    import pandas as pd
    df = pd.read_pickle(processed_data.path)
    print("Training on data:", df.head())


# ✅ PIPELINE
@dsl.pipeline(
    name='Colorectal Cancer Survival Prediction Pipeline',
    description='A pipeline to process data and train model'
)
def mlops_pipeline():
    dp_task = data_processing_op() # type: ignore
    model_training_op(processed_data=dp_task.outputs['processed_data'])


# ✅ COMPILATION
if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=mlops_pipeline, # type: ignore
        package_path='mlops_pipeline.yaml'
    )
    print("✅ Pipeline compiled successfully.")
