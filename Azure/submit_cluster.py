from azureml.core import ScriptRunConfig, Experiment, Workspace, Environment, Dataset

workspace = Workspace.from_config()
workspace.get_details()

experiment = Experiment(workspace, 'Hexafarms_leaf')
cluster = workspace.compute_targets['Hexaleaf']

env = Environment.from_docker_image(name="Hexa_env", image="ccomkhj/mmplant6:v1", container_registry=None, conda_specification=None, pip_requirements=None)

src = ScriptRunConfig(source_directory=".",
                      script='tools/train.py',
                      compute_target="Hexaleaf",
                      environment=env,
                      arguments=[
                        '--config', "configs/deeplabv3plus/deeplabv3plus_r50-d8_480x480_40k_LeafDataset_T2.py",
                        '--datapath', "data/LCCV",
                        '--work-dir', 'outputs'
                       ])

run = experiment.submit(config=src)
run.wait_for_completion(show_output=True)

best_run, fitted_model = run.get_output()
