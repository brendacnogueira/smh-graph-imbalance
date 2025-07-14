# Spectral Manifold Harmonization for Graph Imbalanced Regression
This is the code for Spectral Manifold Harmonization for Graph Imbalanced Regression.
##  Requirements
```
conda env create -f environment.yml
```
## Usage
Run experiments

```
python main_cv.py
```

### Create Augmentations

Create augmentation and train the model: 
```
from utils import *
node_indicator= get_node_indicator(dataset.smiles.values)

graphs_train=smiles_to_nx(dataset_train.smiles.values,node_indicator)
ph_=phi_control(pd.Series(targets_train, dtype="float64"), extr_type="low") #extr_type="high"/"both"/"low"
smh = SimpleSMH()
smh_model,best_params = smh.fit(graphs_train, targets_train,ph_)

synthetic_graphs, synthetic_target_values, indices = smh.generate_samples(targets_train, n_samples=int(n_augmentations))

augmented_graphs=list(graphs_train) + list(synthetic_graphs)
                   
dataset_synthetic = creat_dataset_from_synthetic(synthetic_graphs, synthetic_target_values,node_indicator)
                    
dataset_augmented=dataset_train+dataset_synthetic

augmented_model, augmented_train_losses, augmented_val_losses = train_model_new(
            dataset_augmented,dataset_valid, epochs=epochs,
            lr=lr, batch_size=batch_size,hidden_dim=hidden_dim, num_layers=num_layers

)


```
