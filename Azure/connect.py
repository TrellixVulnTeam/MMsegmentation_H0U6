from azureml.core import Workspace

subscription_id: "dfc1bdc8-c35c-4e7f-ae62-7e8553e4f353"
resource_group  = "Hexa_resource"
workspace_name  = "Hexafarms_Leaf"

try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    ws.write_config()
    print('Library configuration succeeded')
except:
    print('Workspace not found')

