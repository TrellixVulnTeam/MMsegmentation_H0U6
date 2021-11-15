from azureml.core import Workspace, Experiment, experiment
import azureml.core

print("version:", azureml.core.VERSION)



ws = Workspace.from_config()
experiment_name = "Hexa_Leaf_Segmentation"
experiment = Experiment(ws, experiment_name)

from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget

amlcompute_cluster_name = 'cpu-cluster'
provisioning_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D2_V2",
max_nodes = 4)

compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, provisioning_configuration=provisioning_config)

compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
