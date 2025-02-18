from mak.utils.helper import get_config, set_seed, parse_args
args = parse_args()
config_sim = get_config(args.config) 
set_seed(seed=config_sim['common']['seed'])

import flwr as fl
from logging import INFO
from flwr.common.logger import log
from datasets.utils.logging import disable_progress_bar
import os
from mak.utils.helper import get_device_and_resources
from mak.utils.helper import gen_dir_outfile_server, get_model, get_strategy,save_simulation_history,get_dataset, get_size_weights
from mak.utils.pytorch_transformations import get_transformations
from mak.clients import get_client_fn
from mak.servers.custom_server import ServerSaveData
from mak.utils.dataset_info import dataset_info


def main(config_sim):
    fds, centralized_testset = get_dataset(config_sim=config_sim)
    
    if config_sim['server']['strategy'] == 'FedLaw':
        size_weights = get_size_weights(federated_dataset=fds,num_clients=config_sim['server']['num_clients']) #for fedlaw only
    else:
        size_weights = []
    
    dataset_name = fds._dataset_name
    shape = dataset_info[dataset_name]["input_shape"]

    model = get_model(config_sim,shape = shape)
    apply_transforms = get_transformations(dataset_name = dataset_name)
    device, ray_init_args, client_res = get_device_and_resources(config_sim=config_sim)
    generated_info = {"shape" : shape, "device": str(device)}
    config_sim["generated_info"] = generated_info
    out_file_path, saved_models_path = gen_dir_outfile_server(config=config_sim)
    if config_sim["common"]["save_log"]:
        fl.common.logger.configure(identifier="FLNCLAB", filename=os.path.join(saved_models_path,'log.txt'))
        
    log(INFO,f" =>>>>> Dataset : {dataset_name} Shape : {shape}") 

    try:
        dir_alpha = fds._partitioners['train']._alpha[0]
    except (AttributeError):
        dir_alpha = "NA"

    log(INFO,f" =>>>>> Model : {model._get_name()} Device : {device}")
    log(INFO,f" =>>>>> Dataset : {dataset_name} Partitoner : {str(fds._partitioners['train']).split('.')[-1]} Alpha : {dir_alpha}")
    log(INFO,f" =>>>>> Ray init args : {ray_init_args} Client Res : {client_res}")

    strategy = get_strategy(config=config_sim,test_data=centralized_testset,save_model_dir=saved_models_path,out_file_path= out_file_path,device=device,apply_transforms=apply_transforms,size_weights=size_weights)
    server = ServerSaveData(
        strategy=strategy, client_manager=fl.server.client_manager.SimpleClientManager(),out_file_path=out_file_path,target_acc=config_sim['common']['target_acc'])
    
    log(INFO,f" =>>>>> Using Strategy : {strategy.__class__} Server : {server.__class__}")
    
    hist = fl.simulation.start_simulation(
        client_fn=get_client_fn(config_sim = config_sim, model=model,dataset=fds,device=device,apply_transforms=apply_transforms),
        num_clients=config_sim['server']['num_clients'],
        client_resources=client_res,
        config=fl.server.ServerConfig(num_rounds=config_sim['server']['num_rounds']),
        strategy=strategy,
        server=server,
        actor_kwargs={
            "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
        },
        ray_init_args = ray_init_args,
    )

    simu_data_file_path = out_file_path.replace('.csv','_metrics.csv')
    save_simulation_history(hist=hist,path = simu_data_file_path)

if __name__ == "__main__":
    main(config_sim=config_sim)