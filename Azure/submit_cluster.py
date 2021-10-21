from azureml.core import ScriptRunConfig, Experiment, Workspace, Environment, Dataset, Run

workspace = Workspace.from_config()
workspace.get_details()

experiment = Experiment(workspace, 'Hexafarms_leaf')
cluster = workspace.compute_targets['Hexaleaf']

dataset = Dataset.get_by_name(workspace, name='Leaf-Segmentation')



env = Environment.from_docker_image(name="Hexa_env", image="ccomkhj/mmplant5:v1", container_registry=None, conda_specification=None, pip_requirements=None)

src = ScriptRunConfig(source_directory=".",
                      script='tools/train.py',
                      compute_target="Hexaleaf",
                      environment=env,
                      arguments=[
                        '--config', "configs/deeplabv3/deeplabv3_r50-d8_480x480_1k_LeafDataset.py",
                        '--data_path', "data/LCCV",
                       ],
                      )

run = experiment.submit(config=src)
run.wait_for_completion(show_output=True)

'''
가능한문제 원인
1. 폴더가 업로드 되지 않고 있다. (근데 왠지 폴더 업로드는 문제가 없을 것 같다...)
2. /와 \의 혼동으로 인한 문제
3. https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data?view=azure-ml-py
'''