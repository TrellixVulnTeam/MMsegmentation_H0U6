from azureml.core import ScriptRunConfig, Experiment, Workspace, Environment

workspace = Workspace.from_config()
workspace.get_details()

# create or load an experiment
experiment = Experiment(workspace, 'Hexafarms_leaf')
# create or retrieve a compute target
cluster = workspace.compute_targets['Hexaleaf']

'''
mmsegmentation이 설치가 안되있다 현재 docker images는 ...
'''

# env = Environment.from_existing_conda_environment(name="Hexa_env",
#                                                     conda_environment_name="mmplant")

# env = Environment.from_conda_specification(name="Hexa_env", file_path="environment.yml")
# env = Environment.from_pip_requirements(name="Hexa_env", file_path="requirements.txt")


''' Use Docker image'''
env = Environment.from_docker_image(name="Hexa_env", image="ccomkhj/mmplant2:v3", container_registry=None, conda_specification=None, pip_requirements=None)

# # create or retrieve an environment
# env = Environment.get(workspace=workspace, name='fastai')
# # configure and submit your training 3
# config = ScriptRunConfig(source_directory='.',
#                     command=['ls', '-l'],
#                     compute_target=cluster,
#                     environment=env)

src = ScriptRunConfig(source_directory="tools",
                      script='train.py',
                      compute_target="Hexaleaf",
                      environment=env)

# Set compute target
# # Skip this if you are running on your local computer
# script_run_config.run_config.target = my_compute_target

run = experiment.submit(config=src)
run.wait_for_completion(show_output=True)