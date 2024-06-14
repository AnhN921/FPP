from glob_inc.utils import *


def send_task(task_name, client, this_client_id):
    print_log("publish to " + "dynamicFL/req/"+this_client_id)
    client.publish(topic="dynamicFL/req/"+this_client_id, payload=task_name)


def send_model(path, client, this_client_id):
    f = open(path, "rb")
    data = f.read()
    f.close()
    client.publish(topic="dynamicFL/model/all_client", payload=data)
