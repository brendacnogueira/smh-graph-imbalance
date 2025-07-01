from utils import *
from sklearn.model_selection import KFold, train_test_split
import pickle
import json

from itertools import product

rename_dict={"Freesolv":'expt',"Lipo":'y',"ESOL":"measured log solubility in mols per litre","Melting":"melting"}
hyperparameters={
    "lerning_rate":[0.01,0.005,0.001], #[0.01,0.005,0.001],
    "batch_size":[16], 
    "hidden_dim":[32,64], #[32,64]
    "num_layers":[2,5], #[2,5]
    "epochs":[500]
}

hyperparams_xgboost={
    'n_estimators': [10, 50, 100, 250], 
    'learning_rate': [.001, .01, .1], 
    'max_depth': [3, 5, 10],
}
gammas=[1.0,0.5]
aug_sampling=[0.20,0.15,0.1]
binarization_cut=[0.3,0.2,0.1]
rel_thres_list= [0.95,0.99]#[0.95,0.99]



def train_model_aug(params,dataset_augmented,targets_train,dataset_train,dataset_valid,dataset_test, targets_val, targets_test, df_all_aug,df_aug_losses):
    
    
    for lr,batch_size,hidden_dim,num_layers, epochs in product(hyperparameters["lerning_rate"],hyperparameters['batch_size'],hyperparameters['hidden_dim'],hyperparameters['num_layers'],hyperparameters['epochs']):
        
        print("\n===== Training Augmented Model (with synthetic data) =====")

        augmented_model, augmented_train_losses, augmented_val_losses = train_model_new(
            dataset_augmented,dataset_valid, epochs=epochs,
            lr=lr, batch_size=batch_size,hidden_dim=hidden_dim, num_layers=num_layers

        )
        augmented_model.eval()
        with torch.no_grad():
            augmented_train_preds = [augmented_model(data.x.float(), data.edge_index, batch=None).cpu().item() for data in dataset_train]
            augmented_val_preds = [augmented_model(data.x.float(), data.edge_index, batch=None).cpu().item() for data in dataset_valid]

            augmented_test_preds =[augmented_model(data.x.float(), data.edge_index, batch=None).cpu().item() for data in dataset_test]
       
       
        df_train = pd.DataFrame({
            'prediction': augmented_train_preds,

            'target': targets_train,
        })
        df_train['split']='train'

        df_val = pd.DataFrame({
            'prediction': augmented_val_preds,

            'target': targets_val,
        })
        df_val['split']='val'

        df_test = pd.DataFrame({
            'prediction': augmented_test_preds,

            'target': targets_test,
        })
        df_test['split']='test'
        print("test ",len(df_test))

        # Concatenate all into one DataFrame
        df_aug = pd.concat([df_train, df_val, df_test], ignore_index=True)

        df_aug['learning_rate']=lr
        df_aug['batch_size']=batch_size
        df_aug['hidden_dim']=hidden_dim
        df_aug['num_layers']=num_layers
        df_aug['epochs']=epochs
        df_aug['fold']=i

        df_aug["method"]=method
        df_aug['aug_params']=str(params)
        df_all_aug = pd.concat([df_all_aug, df_aug], ignore_index=True)

        df_losses=pd.DataFrame({
                "train":augmented_train_losses,
                "val":augmented_val_losses,
        })
        df_losses['learning_rate']=lr
        df_losses['batch_size']=batch_size
        df_losses['hidden_dim']=hidden_dim
        df_losses['num_layers']=num_layers
        df_losses['epochs']=epochs
        df_losses['fold']=i
        df_losses["method"]=method
        df_losses['aug_params']=str(params)
        df_aug_losses=pd.concat([df_aug_losses,df_losses],ignore_index=True)

    return df_all_aug,df_aug_losses


for dataset_name in ['Lipo']: #'ESOL','Lipo','Freesolv','Melting'
    print(f"###Dataset {dataset_name}###")
    synthetic_data=[]
    dataset={}
    dataset['all']=pd.DataFrame()
    for dat in ['train','valid','test']:
        data_pd=pd.read_csv(f'datasets/{dataset_name}/{dat}.csv')
        
        data_pd=data_pd.rename(columns={rename_dict[dataset_name]:'y'})
        if dataset_name=="Melting":
            data_pd=data_pd.rename(columns={"SMILES":'smiles'})
        dataset[dat]=data_pd
        dataset['all']=pd.concat([dataset['all'],dataset[dat]])
    dataset['all']=dataset['all'].reset_index()
    #targets_train=dataset['train'].y.values
    #targets_val=dataset['valid'].y.values
    #targets_test=dataset['test'].y.values
    kf=KFold(n_splits=5,shuffle=True, random_state=42)
    df_all_orig=pd.DataFrame()
    df_all_aug=pd.DataFrame()

    df_ori_losses=pd.DataFrame()
    df_aug_losses=pd.DataFrame()
    for i, (train_index, test_index) in enumerate(kf.split(dataset['all'])):
        print(f"### {i} fold")
        train_index, val_index= train_test_split(train_index, test_size=0.25, shuffle=True, random_state=42)
        #targets_train=dataset['train'].y.values
        #targets_val=dataset['valid'].y.values
        #targets_test=dataset['test'].y.values
        targets_train=dataset['all'].y.values[train_index]
        targets_val=dataset['all'].y.values[val_index]
        targets_test=dataset['all'].y.values[test_index]
        

        ph_=phi_control(pd.Series(targets_train, dtype="float64"), extr_type="low")    
        print(f"Dataset split: {len(targets_train)} train, {len(targets_val)} validation, {len(targets_test)} test")

        device='cuda'
        node_indicator= get_node_indicator(dataset['all'].smiles.values)

        graphs_train=smiles_to_nx(dataset['all'].smiles.values[train_index],node_indicator) #node_indicator
        graphs_val=smiles_to_nx(dataset['all'].smiles.values[val_index],node_indicator)
        graphs_test=smiles_to_nx(dataset['all'].smiles.values[test_index],node_indicator)
        print("Extracting graph features...")

        dataset_train=create_dataset(graphs_train, targets_train,device)
        dataset_valid=create_dataset(graphs_val, targets_val,device)

        dataset_test=create_dataset(graphs_test, targets_test,device)
        print(len(dataset_test))
        print("\n===== Training Baseline Model (without synthetic data) =====")
        for lr,batch_size,hidden_dim,num_layers, epochs in product(hyperparameters["lerning_rate"],hyperparameters['batch_size'],hyperparameters['hidden_dim'],hyperparameters['num_layers'],hyperparameters['epochs']):
            
            # Evaluate baseline model
            print(f"{lr,batch_size,hidden_dim,num_layers, epochs}")
            baseline_model, baseline_train_losses, baseline_val_losses = train_model_new(
                dataset_train,dataset_valid, epochs=epochs,
                lr=lr, batch_size=batch_size,hidden_dim=hidden_dim, num_layers=num_layers
            )

            baseline_model.eval()
            with torch.no_grad():

                baseline_train_preds = [baseline_model(data.x.float(), data.edge_index, batch=None).cpu().item() for data in dataset_train]
                baseline_val_preds = [baseline_model(data.x.float(), data.edge_index, batch=None).cpu().item() for data in dataset_valid]

                baseline_test_preds =[baseline_model(data.x.float(), data.edge_index, batch=None).cpu().item() for data in dataset_test]

                df_train = pd.DataFrame({
                    'prediction': baseline_train_preds,
                    'target': targets_train,
                })
                df_train['split']='train'

                df_val = pd.DataFrame({
                    'prediction': baseline_val_preds,
                    'target': targets_val,
                })
                df_val['split']='val'

                df_test = pd.DataFrame({
                    'prediction': baseline_test_preds,
                    'target': targets_test,
                })
                df_test['split']='test'

                # Concatenate all into one DataFrame
                df_orig = pd.concat([df_train, df_val, df_test], ignore_index=True)
                df_orig['learning_rate']=lr
                df_orig['batch_size']=batch_size
                df_orig['hidden_dim']=hidden_dim
                df_orig['num_layers']=num_layers
                df_orig['epochs']=epochs
                df_orig['fold']=i
                df_orig['aug_params']=0

                df_all_orig = pd.concat([df_all_orig, df_orig], ignore_index=True)
            
            df_losses=pd.DataFrame({
                "train":baseline_train_losses,
                "val":baseline_val_losses,
            })
            df_losses['learning_rate']=lr
            df_losses['batch_size']=batch_size
            df_losses['hidden_dim']=hidden_dim
            df_losses['num_layers']=num_layers
            df_losses['epochs']=epochs
            df_losses['fold']=i
            df_losses['aug_params']=0
            df_ori_losses=pd.concat([df_ori_losses,df_losses],ignore_index=True)

        
        print("\n===== Training SMH Model and Generating Synthetic Samples =====")
        smh = SimpleSMH()
        smh_model,best_params = smh.fit(graphs_train, targets_train,ph_)
        
        smh.only_aug=False
        for method in ['shm','smogn']:
            print(f"{method}")
            if method=="shm":
                for gamma, aug_samp,cut in product(gammas,aug_sampling,binarization_cut):
                    print(f"fold:{i}, {gamma, aug_samp,cut}")
                    smh.gamma=gamma
                    smh.cut=cut
                  
                    params=[gamma,aug_samp,cut,best_params]
                    
                    print("Generating synthetic samples for underrepresented target ranges...")
                    synthetic_graphs, synthetic_target_values, indices = smh.generate_samples(
                        targets_train, n_samples=int(aug_samp*len(targets_train)), 

                    )
                    synthetic_data.append({ "method":method,
                                                "fold":i,
                                                "aug_params":params, 
                                                'original_graphs':graphs_train,
                                                'targets':targets_train,
                                                'synthetic_graphs': synthetic_graphs,
                                                'synthetic_target_values': synthetic_target_values,
                                                'samples_augmented':indices})
                    
                    augmented_graphs=list(graphs_train) + list(synthetic_graphs)
   
                    #targets_augmented = np.append(targets_train, synthetic_target_values)

                    dataset_synthetic = creat_dataset_from_synthetic(synthetic_graphs, synthetic_target_values,node_indicator)
                    
                    dataset_augmented=dataset_train+dataset_synthetic
                    
                    df_all_aug,df_aug_losses=train_model_aug(params,dataset_augmented,targets_train,dataset_train,dataset_valid,dataset_test, targets_val, targets_test, df_all_aug,df_aug_losses)
                    

            elif method=="smogn":
                
                for rel_thres in rel_thres_list:
                    print(f"Fold:{i}, rel: {rel_thres}")
                    params=rel_thres
                    synthetic_graphs, synthetic_target_values, indices = smh.generate_samples_smogn(
                        targets_train, rel_thres=rel_thres, 

                    )
                    
                    synthetic_data.append({  "method":method,
                                                "fold":i,
                                                "aug_params":params, 
                                                'original_graphs':graphs_train,
                                                'targets':targets_train,
                                                'synthetic_graphs': synthetic_graphs,
                                                'synthetic_target_values': synthetic_target_values,
                                                'samples_augmented':indices})
                    augmented_graphs=list(graphs_train) + list(synthetic_graphs)
   
                    #targets_augmented = np.append(targets_train, synthetic_target_values)

                    dataset_synthetic = creat_dataset_from_synthetic(synthetic_graphs, synthetic_target_values,node_indicator)
                    
                    dataset_augmented=dataset_train+dataset_synthetic

                    df_all_aug,df_aug_losses=train_model_aug(params,dataset_augmented,targets_train,dataset_train,dataset_valid,dataset_test, targets_val, targets_test, df_all_aug,df_aug_losses)
                    print("Folds df:",df_all_aug['fold'].unique())

    df_all_orig.to_csv(f'results/{dataset_name}/baseline_predictions_all.csv', index=False)
    df_ori_losses.to_csv(f'results/{dataset_name}/baseline_losses_all.csv', index=False)
         
    df_all_aug.to_csv(f'results/{dataset_name}/augmented_predictions_all.csv', index=False)
    df_aug_losses.to_csv(f'results/{dataset_name}/augmented_losses_all.csv', index=False)

    with open(f'results/{dataset_name}/syntetic_results_all.pkl', 'wb') as file:
        pickle.dump(synthetic_data, file) 




