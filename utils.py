import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from sera import *
from rdkit import Chem
import networkx as nx
import pysmiles
import rdkit
import scipy.sparse as sp

import random
from sklearn.neighbors import KernelDensity
import torch.nn.functional as F
from torch_geometric.nn import  global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import  GCNConv, GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU
from sklearn.model_selection import GridSearchCV

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import wilcoxon

from torch_geometric.utils import from_networkx
\
import smogn_
from torch_geometric.utils.convert import from_networkx

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set the seed at the start of your script
set_seed(42)

import pandas as pd

class GINRegression(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(GINRegression, self).__init__()
        self.num_layers=num_layers
        nn1 = Sequential(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)

        nn2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)

        self.linear1 = Linear(hidden_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, 1)  # Output one regression value per graph

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        for num in range(self.num_layers):
            x = F.relu(self.conv2(x, edge_index))
        x_node = global_add_pool(x, batch)
        x_node = F.relu(self.linear1(x_node))
        return self.linear2(x_node).squeeze(-1)

class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Pooling to get [batch_size, hidden_channels]
        x = self.lin(x)  # Graph-level output
        return x
   
def train_model_new(dataset_train, dataset_valid, criterion=nn.MSELoss(), epochs=300, lr=0.001, batch_size=16, hidden_dim=64, num_layers=1):
    device = "cuda"

    model = GINRegression(input_dim=dataset_train[0].num_node_features, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_state = None
    epochs_since_improvement = 0
    patience = 20

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs= model(batch.x.float(), batch.edge_index, batch=batch.batch).squeeze()
            loss = criterion(outputs, batch.y.view(-1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(dataset_train)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_loss_val = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                out = model(batch.x.float(), batch.edge_index, batch=batch.batch).squeeze()
                val_loss = criterion(out.cpu(), batch.y.view(-1).cpu().float()).item()
                running_loss_val += val_loss

        epoch_val_loss = running_loss_val / len(dataset_valid)
        val_losses.append(epoch_val_loss)

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if (epoch + 1) % 20 == 0 or epochs_since_improvement >= patience:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if epochs_since_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses

# ==========================================================
# 5. Simplified SMH implementation
# ==========================================================
import numpy as np
import pandas as pd
import xgboost as xgb

import numpy as np
from scipy.optimize import minimize

def reconstruct_laplacian(coefficients, eigenvalues, eigenvectors):
    """Reconstructs Laplacian from spectral coefficients."""
    n_nodes = eigenvectors.shape[0]
    laplacian = np.zeros((n_nodes, n_nodes))
    for i, (c, l) in enumerate(zip(coefficients, eigenvalues)):
        v = eigenvectors[:, i]
        laplacian += c * l * np.outer(v, v)
    return (laplacian + laplacian.T) / 2  # ensure symmetry

def adjacency_from_laplacian(laplacian):
    """Converts Laplacian to adjacency matrix."""
    adjacency = np.eye(laplacian.shape[0]) - laplacian
    np.fill_diagonal(adjacency, 0)
    adjacency = np.maximum(adjacency, 0)
    return adjacency

def fiedler_value(laplacian):
    """Returns the Fiedler value (second smallest eigenvalue)."""
    eigvals = np.linalg.eigvalsh(laplacian)
    return eigvals[1] if len(eigvals) > 1 else 0

def edge_density(adjacency):
    """Computes edge density of a graph."""
    n = adjacency.shape[0]
    return np.sum(adjacency) / (n * (n - 1))

def objective(s_new, s_original):
    """Minimization objective: L2 distance to original."""
    return np.sum((s_new - s_original) ** 2)

def constraint_connectivity(s_new, eigenvalues, eigenvectors):
    """Connectivity constraint: Fiedler value must be > 0."""
    lap = reconstruct_laplacian(s_new, eigenvalues, eigenvectors)
    return fiedler_value(lap) - 1e-3

def constraint_sparsity(s_new, eigenvalues, eigenvectors, tau):
    """Sparsity constraint: edge density <= tau."""
    lap = reconstruct_laplacian(s_new, eigenvalues, eigenvectors)
    adj = adjacency_from_laplacian(lap)
    return tau - edge_density(adj)

def project_spectral_coefficients(s_original, eigenvalues, eigenvectors, tau=0.2):
    """Projects spectral coefficients to domain-valid region."""
    cons = [
        {'type': 'ineq', 'fun': constraint_connectivity, 'args': (eigenvalues, eigenvectors)},
        {'type': 'ineq', 'fun': constraint_sparsity, 'args': (eigenvalues, eigenvectors, tau)},
    ]
    result = minimize(objective, s_original, args=(s_original,), constraints=cons, method='SLSQP')
    return result.x if result.success else None


class SimpleSMH:
    """Simplified SMH for generating balanced samples."""

    def __init__(self,gamma=1.0,cut=0.3):
        #self.spectral_dim = spectral_dim
        #self.hidden_dims = hidden_dims
        #self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Will be initialized during fit
        self.mapper = None
        self.eigenvalues_dict = {}
        self.eigenvectors_dict = {}
        self.target_scaler = StandardScaler()
        self.ph_=None
        self.index_to_target = {}
        self.gamma=gamma
        self.cut=cut
        self.only_aug=None
        self.original_len={}
    

    def _compute_spectral_decomposition(self, graph,max):
        """Compute eigenvalues and eigenvectors of the graph Laplacian."""
        # Get Laplacian
        laplacian = nx.normalized_laplacian_matrix(graph).toarray()
        laplacian = (laplacian + laplacian.T) / 2  # Ensure symmetry
       
        # Compute eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        # Sort by eigenvalues
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Limit dimensions if needed
        #if self.spectral_dim < len(eigenvalues):
        #    eigenvalues = eigenvalues[:self.spectral_dim]
        #    eigenvectors = eigenvectors[:, :self.spectral_dim]
        #if len(eigenvalues) < self.spectral_dim:
        #    print(eigenvalues)
        #    pad_len = self.spectral_dim - len(eigenvalues)
        #    eigenvalues = np.pad(eigenvalues, (0, pad_len), mode='constant', constant_values=0)
        #    eigenvectors = np.pad(eigenvectors, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
        if max<len(eigenvalues):
            max=len(eigenvalues)

        return eigenvalues, eigenvectors,max
    

    
    def fit(self, graphs, targets, ph_, verbose=True):
        """Fit the SMH model to the data using mini-batch training."""
        
        self.ph_ = ph_
        self.index_to_target = {i: targets[i] for i in range(len(graphs))}
        # Step 1: Compute spectral decomposition for all graphs
        max_len = 0
        list_eigenvalues = []

        for i, graph in enumerate(graphs):
            eigenvalues, eigenvectors, max_len = self._compute_spectral_decomposition(graph, max_len)
            self.eigenvalues_dict[i] = eigenvalues
            self.eigenvectors_dict[i] = eigenvectors
            self.original_len[i] = len(eigenvalues)
    
        
        self.spectral_dim = max_len
        

        for i in range(len(graphs)):
            eigenvalues = self.eigenvalues_dict[i]
            if len(eigenvalues) < self.spectral_dim:
                pad_len = self.spectral_dim - len(eigenvalues)
                eigenvalues = np.pad(eigenvalues, (0, pad_len), mode='constant', constant_values=0)
            list_eigenvalues.append(eigenvalues)
        
        X = np.array(targets).reshape(-1, 1).astype(np.float32)   # target values as features/input
        y = np.vstack(list_eigenvalues).astype(np.float32)        # eigenvalues as output/labels
        phs = phi(pd.Series(targets,dtype=np.float64), phi_parms=ph_).astype(np.float32)
        dtrain = xgb.DMatrix(X, label=y)
     
        # Step 2: Define custom objective and eval metric
        def relevance_weighted_objective(): #phi_parms
            def custom_obj(preds: np.ndarray, y_true:np.ndarray,sample_weight: np.ndarray): #dtrain: xgb.DMatrix
                #y_true = dtrain.get_label().astype(np.float64) 
                
                #y_true=y_true.reshape(preds.shape[0], preds.shape[1])
                #x_input=dtrain.get_data().toarray().flatten()

                #phs = phi(pd.Series(x_input,dtype=np.float64), phi_parms=phi_parms).astype(np.float32)
                
                preds=preds.reshape(y_true.shape)
                grad = 2 * sample_weight[:,None] * (preds - y_true)  # add new axis to phs for broadcasting
                hess = 2 * sample_weight[:,None] * np.ones_like(preds)
                return grad, hess
            return custom_obj

        #def relevance_weighted_eval():#phi_parms
        def eval_metric(preds, y_true,sample_weight): #dtrain
            #y_true = dtrain.get_label().astype(np.float64) 
            #y_true=y_true.reshape(preds.shape[0], preds.shape[1])
            #x_input=dtrain.get_data().toarray().flatten()
            
            #phs = phi(pd.Series(x_input,dtype=np.float64), phi_parms=phi_parms).astype(np.float32)
            weighted_mse = np.mean(sample_weight[:,None] * (preds - y_true) ** 2)
            return weighted_mse
            # return 'relevance_weighted_mse', float(weighted_mse)
            #return eval_metric

        custom_obj = relevance_weighted_objective() #phi_parms=ph_
        #feval = relevance_weighted_eval() #phi_parms=ph_
        

        # Step 3: Train with low-level API `xgb.train()`
        evals_result = {}
        params = {
            'booster': 'gbtree',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 1e-4,     # L2 regularization
            'tree_method': 'auto',
            'seed': 42, 
            'num_boost_round' : 100,

        }
        #self.mapper = xgb.train(
        #    params=params,
        #    dtrain=dtrain,
        #    obj=custom_obj,
        #    feval=feval,
        #    evals=[(dtrain, 'train')],
        #    evals_result=evals_result,
        #    verbose_eval=verbose
        #)
        h_parameters={
            'n_estimators': [10, 50, 100, 250], 
            'learning_rate': [.001, .01, .1], 
            'max_depth': [3, 5, 10],
            'reg_lambda': [1e-4], 
            'booster': ['gbtree'],
            
        }

        model=xgb.XGBRegressor(random_state=42, objective=custom_obj)

        cv_results = GridSearchCV(model,
                                  param_grid=h_parameters,
                                  cv=5,
                                  
                                  n_jobs=-1
                                  
                                )
        cv_results.fit(X,y,sample_weight=phs)

       
        best_params = cv_results.cv_results_['params'][cv_results.best_index_]
        
        model = model.set_params(**best_params)
        model.fit(X,y,sample_weight=phs)
        self.mapper = model
        return model, best_params

    
    def compute_relevance_weights(self, target_value, target_values, relevance_values):
        """
        Compute weights based on target similarity and relevance (Equation 5).
        
        Args:
            target_value: Target value to generate samples for
            target_values: Array of target values in training set
            relevance_values: Relevance value for each target
            
        Returns:
            Weights for each sample
        """
        # Compute kernel similarity (K(y, yi) in the paper)
        kernel_sim = np.exp(-self.gamma * (target_value - target_values)**2)
        
        # Combine similarity and relevance
        weights = kernel_sim #* relevance_values
        
        # Normalize
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones_like(weights) / len(weights)
        
        return weights

    def compute_conditional_distribution(self, target_value, spectral_coeffs, target_values):
        """
        Compute the conditional distribution p(s|y) = N(μ(y), Σ(y)) as in Equations 3-4.
        
        Args:
            target_value: Target value to condition on
            spectral_coeffs: Array of spectral coefficients
            target_values: Array of target values
            
        Returns:
            mean_vector: Mean of the conditional distribution
            cov_matrix: Covariance matrix of the conditional distribution
        """
        # Compute relevance for all targets
        
        relevance_values = phi(pd.Series(target_values), phi_parms=self.ph_)
        
        # Compute weights using Equation 5
        weights = self.compute_relevance_weights(target_value, target_values, relevance_values)
        
        # Get mean from the mapping network
        target_tensor = torch.tensor(target_value, dtype=torch.float32, device=self.device).view(1)
        with torch.no_grad():
            mean_vector = self.mapper.predict(
               target_tensor.view(1, 1).cpu().numpy()
            ).squeeze()# xgb.DMatrix(target_tensor.view(1, 1).cpu().numpy())
            
        #mean_vector=mean_vector.cpu().numpy()

        # Compute weighted covariance matrix using Equation 4
        n_coeffs = spectral_coeffs.shape[1]
        cov_matrix = np.zeros((n_coeffs, n_coeffs))
        
        for i in range(len(target_values)):
            diff = spectral_coeffs[i] - mean_vector
            outer_product = np.outer(diff, diff)
            cov_matrix += weights[i] * outer_product
        
        # Ensure symmetry and positive-definiteness
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        cov_matrix += 1e-6 * np.eye(n_coeffs)
        
        return mean_vector, cov_matrix
 

    def generate_samples(self,target_values,n_samples=200,gamma=1.0):
        phs=phi(pd.Series(target_values), phi_parms=self.ph_)
        
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(target_values.reshape(-1, 1))
        log_density = kde.score_samples(target_values.reshape(-1, 1))
        density=np.exp(log_density)
        epsilon=1e-6
        weights = phs / (density + epsilon)
        weights=weights/weights.sum()
        indices=np.random.choice([i for i in range(len(target_values))], size=n_samples, p=weights)
        
        new_target=[target_values[i] for i in indices]
        

        """Generate synthetic graphs for specific target values."""
        # Scale target values
        new_target_array = np.array(new_target).reshape(-1, 1)
        #scaled_targets = self.target_scaler.transform(new_target_array).flatten()

        # Select base graphs
        base_indices = list(range(len(self.eigenvalues_dict)))

 
        # Prepare arrays for spectral coefficients and their corresponding targets
        all_coeffs = []
        all_scaled_targets = []
        all_targets=[]
        
        for i in range(len(self.eigenvalues_dict)):
            if len(self.eigenvalues_dict[i]) < self.spectral_dim:
                pad_len = self.spectral_dim - len(self.eigenvalues_dict[i])
                self.eigenvalues_dict[i] = np.pad(self.eigenvalues_dict[i], (0, pad_len), mode='constant', constant_values=0)
                #self.eigenvectors_dict[i]= np.pad(self.eigenvectors_dict[i], ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
            all_coeffs.append(self.eigenvalues_dict[i])

            #all_scaled_targets.append(self.index_to_target[i])
            all_targets.append(self.index_to_target[i])
        
        all_coeffs = np.array(all_coeffs)
        #all_scaled_targets = np.array(all_scaled_targets)
        all_targets = np.array(all_targets)
   
        # Generate samples
        generated_graphs = []
        generated_targets = []
        #print("indices: ",len(indices))
        #print("Mean: ",np.mean(all_coeffs, axis=0), "std:",np.std(all_coeffs, axis=0))
        list_indices=[]
        for i, idx in enumerate(indices): #new_target_array

            target_value=target_values[idx]
            # Select base graph
            base_idx = idx
            eigenvalues = self.eigenvalues_dict[base_idx]
            eigenvectors = self.eigenvectors_dict[base_idx]
            original_len=self.original_len[base_idx]
            
            
            # Add some noise for variety
            #noise = np.random.normal(0, 0.1, size=spectral_coeffs.shape)
            #modified_coeffs = spectral_coeffs + noise
            
            # Compute the conditional distribution p(s|y) using Equations 3-5
            mean_vector, cov_matrix = self.compute_conditional_distribution(
                target_value, all_coeffs, all_targets
            )
            mean_vector = mean_vector.flatten()
            
            # Sample spectral coefficients from the conditional distribution
            sampled_coeffs = np.random.multivariate_normal(mean_vector, cov_matrix)
            #target_tensor = torch.FloatTensor([target_value]).to(self.device)
            #with torch.no_grad():
            #    sampled_coeffs = self.mapper(target_tensor.view(1, 1)).squeeze().cpu().numpy()
            
            # Generate graph
            graph = self._spectral_to_graph(eigenvalues, eigenvectors, sampled_coeffs,original_len)
            
            # Store with original (unscaled) target value
            #original_target = self.target_scaler.inverse_transform([[target_value]])[0, 0]
            if graph!=None:
                generated_graphs.append(graph)
                #generated_targets.append(original_target)
                generated_targets.append(float(target_value))
                list_indices.append(idx)
        print(len(generated_graphs))
        return generated_graphs, np.array(generated_targets), list_indices
    
    def generate_samples_smogn(self,target_values,rel_thres=0.9):
        
        # Prepare arrays for spectral coefficients and their corresponding targets
        all_coeffs = []
        all_scaled_targets = []
        all_targets=[]
        
        for i in range(len(self.eigenvalues_dict)):
            if len(self.eigenvalues_dict[i]) < self.spectral_dim:
                pad_len = self.spectral_dim - len(self.eigenvalues_dict[i])
                self.eigenvalues_dict[i] = np.pad(self.eigenvalues_dict[i], (0, pad_len), mode='constant', constant_values=0)
                #self.eigenvectors_dict[i]= np.pad(self.eigenvectors_dict[i], ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
            all_coeffs.append(self.eigenvalues_dict[i])
            #all_scaled_targets.append(self.index_to_target[i])
            all_targets.append(self.index_to_target[i])
        
        all_coeffs = np.array(all_coeffs)
        #all_scaled_targets = np.array(all_scaled_targets)
        all_targets = np.array(all_targets)
        df=pd.DataFrame(all_coeffs)
        df.columns = [f'Feature_{i+1}' for i in range(df.shape[1])]
        df['target']=all_targets
        df_aug, indices, n_aug=smogn_.smoter(data = df, y = "target",under_samp=False,samp_method="balance",rel_thres=rel_thres)
        
        # Generate samples
        generated_graphs = []
        generated_targets = []
        #print("indices: ",len(indices))
        #print("Mean: ",np.mean(all_coeffs, axis=0), "std:",np.std(all_coeffs, axis=0))
        list_indices=[]
    
        j=0
        for i, idx in enumerate(indices): #new_target_array
            # Select base graph
            base_idx = idx
            eigenvalues = self.eigenvalues_dict[base_idx]
            eigenvectors = self.eigenvectors_dict[base_idx]
            original_len=self.original_len[base_idx]
             
            #with torch.no_grad():
            #    sampled_coeffs = self.mapper(target_tensor.view(1, 1)).squeeze().cpu().numpy()
            
            # Generate graph
            aug_data=df_aug.iloc[j:j+n_aug]
            #print(df.iloc[idx,:])
            #print(aug_data)
            vectors=aug_data.drop(columns='target').values
            targets=aug_data.target.values
            for sampled_coeffs, target in zip(vectors, targets):
                graph = self._spectral_to_graph(eigenvalues, eigenvectors, sampled_coeffs,original_len)
                
                # Store with original (unscaled) target value
                #original_target = self.target_scaler.inverse_transform([[target_value]])[0, 0]
                if graph!=None:
                    generated_graphs.append(graph)
                    #generated_targets.append(original_target)
                    generated_targets.append(float(target))
                    list_indices.append(idx)
            j=j+n_aug
        print(len(generated_graphs))
        return generated_graphs, np.array(generated_targets), list_indices


    def _spectral_to_graph(self, eigenvalues, eigenvectors, coefficients,original_len):
        """Convert spectral representation to graph."""
        # Prepare for reconstruction
        #n_values = min(len(eigenvalues), len(coefficients))
        #eigenvalues = eigenvalues[:n_values]
        #coefficients = coefficients[:n_values]
        #eigenvectors = eigenvectors[:, :n_values]

        n_values = original_len
        eigenvalues = eigenvalues[:original_len]
        coefficients = coefficients[:original_len]
        eigenvectors = eigenvectors[:, :original_len]

        # Initialize reconstructed Laplacian
        n_nodes = eigenvectors.shape[0]
        laplacian = np.zeros((n_nodes, n_nodes))

        # Reconstruct Laplacian
        for i in range(n_values):
            lambda_i = eigenvalues[i]
            v_i = eigenvectors[:, i]
            coeff_i = coefficients[i]

            outer_product = np.outer(v_i, v_i)
            if self.only_aug:
                laplacian += coeff_i * outer_product 
            else:
                laplacian += coeff_i * lambda_i * outer_product 

        # Ensure symmetry
        laplacian = (laplacian + laplacian.T) / 2

        ## Convert to adjacency matrix
        adjacency = np.eye(n_nodes) - laplacian
#
        ## Remove self-loops
        np.fill_diagonal(adjacency, 0)
#
        ## Ensure values are valid
        adjacency = np.maximum(adjacency, 0)

        
        # Binarize
        adjacency = (adjacency > self.cut).astype(float)
      

        # Create graph
        graph = nx.from_numpy_array(adjacency)
         
        return graph
    


def Sanitize_smiles(smiles):
    '''

    :param smiles:
    :return: standardized smiles
    '''
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)
    if 'se' in smiles:
        smiles = smiles.replace('se', 'Se')
    return smiles
def Mol2Graph(mol, node_labels, H_flag):
    '''

    :param mol: molecule
    :param node_labels: node label map
    :return: graph
    :des: convert atoms to nodes; convert bonds to edges
    '''
    graph = nx.Graph()
    if H_flag == 1:
        mol = Chem.AddHs(mol)
    # node_label-element map
    node_labels = dict(zip(node_labels.values(), node_labels.keys()))
    # edge_order-bond map
    bond_labels = dict(zip(rdkit.Chem.rdchem.BondType.values.values(), rdkit.Chem.rdchem.BondType.values.keys()))
    bonds = mol.GetBonds()
    # Get edge list with orders
    bond_list = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), {'order': int(bond.GetBondType())}) for bond in bonds]
    for id, node in enumerate(mol.GetAtoms()):
        graph.add_node(id)
        if node.HasProp("__origIdx"):
            graph.nodes[id]['ori_pos'] = node.GetIntProp("__origIdx")
        graph.nodes[id]['element'] = mol.GetAtoms()[id].GetSymbol()
        graph.nodes[id]['charge'] = mol.GetAtoms()[id].GetFormalCharge()
        graph.nodes[id]['hcount'] = mol.GetAtoms()[id].GetTotalNumHs()
        graph.nodes[id]['aromatic'] = mol.GetAtoms()[id].GetIsAromatic()
        graph.nodes[id]['RadElec'] = mol.GetAtoms()[id].GetNumRadicalElectrons()
        node_label_one_hot = [0] * len(node_labels)
        node_label = int(node_labels.get(graph.nodes[id]['element']))
        node_label_one_hot[node_label] = 1
        graph.nodes[id]['label'] = node_label_one_hot
    graph.add_edges_from(bond_list)
    
    return graph

def Graph2Mol(graph_o):
    '''

    :param graph: molecule graph
    :return: molecule
    '''
    flag = 0
    # Get atoms and adjacency
    graph = graph_o
    new_node_labels = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, new_node_labels)
    node_list = list(nx.get_node_attributes(graph, 'element').values())
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    bonds = nx.get_edge_attributes(graph, 'order')
    for i, edge in enumerate(graph.edges()):
        adjacency_matrix[edge] = bonds[edge]
        adjacency_matrix[edge[1], edge[0]] = bonds[edge]
    adjacency_matrix = np.array(adjacency_matrix)
    # create empty editable mol object
    mol = Chem.RWMol()
    # add atoms to mol and keep track of index
    node_to_idx = {}
    for id, node in enumerate(graph.nodes):
        a = Chem.Atom(node_list[id])
        molIdx = mol.AddAtom(a)
        try:
            mol.GetAtoms()[id].SetFormalCharge(graph.nodes[id]['charge'])
        except:
            flag = 1
        try:
            mol.GetAtoms()[id].SetIsAromatic(graph.nodes[id]['aromatic'])
        except:
            flag = 2
        try:
            hcount = graph.nodes[id]['hcount']
            mol.GetAtoms()[id].SetNumExplicitHs(hcount)
        except:
            flag = 3
        try:
            RadElec = graph.nodes[id]['RadElec']
            mol.GetAtoms()[id].SetNumRadicalElectrons(RadElec)
        except:
            flag = 4
        node_to_idx[id] = molIdx
    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            # only traverse half the matrix
            if iy <= ix:
                continue
            if bond == 0:
                continue
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], rdkit.Chem.rdchem.BondType.values[bond])
    mol = mol.GetMol()
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None:
        
        Chem.Kekulize(mol, True)
        
    return mol
def get_node_indicator(smiles):
    elements=[]
    for id, smiles in enumerate(smiles):
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)
            
            if 'se' in smiles:
                smiles = smiles.replace('se', 'Se')
            
            try:
                graph = pysmiles.read_smiles(smiles, reinterpret_aromatic=True)
            except ValueError:
                print('value error{}'.format(smiles))

            elements.extend(list(nx.get_node_attributes(graph, 'element').values()))
            if "*" in smiles:
                elements.extend(['*'])
            
    elements = list(set(elements))
    elements_num = len(elements)
    elements_id = range(elements_num)
    node_indicator = dict(zip(elements_id, elements))
    return node_indicator
def smiles_to_nx(smiles,node_indicator):
    graphs=[]
    for id, smiles in enumerate(smiles):
            smiles = Sanitize_smiles(smiles)
            mol = Chem.MolFromSmiles(smiles)
           
            Chem.Kekulize(mol)
            graph = Mol2Graph(mol, node_indicator, H_flag=0)
            mol = Graph2Mol(graph)
            smiles1 = Chem.MolToSmiles(mol, allHsExplicit=True)

            if 'se' in smiles:
                smiles = smiles.replace('se', 'Se')
            smiles1 = Sanitize_smiles(smiles1)

            graphs.append(graph)
    
    Graphs = [g for g in graphs if g.number_of_nodes() > 0]
    for i, g in enumerate(Graphs):
        Graphs[i].graph['id'] = i
        Graphs[i].graph['stand'] = 1
        Graphs[i].graph['right'] = 1
    return Graphs

def Network2TuDataset(graph,  y, device, edge_feat_dim=4):
    g = graph
    adj = sp.coo_matrix(nx.adjacency_matrix(g))
    if len(g) == 1:
        edge_index = torch.LongTensor([[0], [0]]).to(device)
        edgetype = np.array([0])
    else:
        edge_index = torch.vstack((torch.LongTensor(adj.row), torch.LongTensor(adj.col))).to(device)
        try:
            edgetype = np.array([g.get_edge_data(edge_index[0][num].item(), edge_index[1][num].item())['order'] for num in range(edge_index.shape[1])])
        except:
            edgetype = []
            for num in range(edge_index.shape[1]):
                try:
                    type = g.get_edge_data(edge_index[0][num].item(), edge_index[1][num].item())['order']
                    edgetype.append(type)
                except:
                    type = 1
                    edgetype.append(type)
            edgetype = np.array(edgetype)
    edge_feat = torch.LongTensor(edgetype).to(device)
    y = torch.tensor(y).to(device)
    
    feat = list(nx.get_node_attributes(g, 'label').values())
    x = torch.FloatTensor(feat).to(device)
    
    data =Data(x=x, edge_index=edge_index, y=y)# edge_attr=edge_feat
    return data




def create_dataset(graphs, targets,device):
    dataset=[]
    for graph, target in zip(graphs,targets):
        data=Network2TuDataset(graph, target, device)
        #data=nx_to_pyg_data(graph,target)
        dataset.append(data)
    return dataset

def creat_dataset_from_synthetic(graphs,targets,node_indicator,default_element='C', default_order=1, device='cuda'):
    dataset=[]
    num_node_features = len(node_indicator)
    
    for graph, target in zip(graphs, targets):
        # Set node features: one-hot for default atom (e.g., Carbon)
        for node in graph.nodes():
            graph.nodes[node]['x'] = torch.tensor(
                [0.0] * num_node_features, dtype=torch.float
            )

        # Set edge features: all synthetic bonds get dummy value = 1
        for u, v in graph.edges():
            graph[u][v]['edge_attr'] = torch.tensor([1,1], dtype=torch.long)

        # Convert to PyG Data
        data = from_networkx(graph).to(device)
        data.weight=None
        data.edge_attr=None
        #if data.edge_attr==None:
        #    data.edge_attr = torch.ones(data.edge_index.shape[1], 2, dtype=torch.long).to(device)
        data.y=torch.tensor(target).to(device)
        
        dataset.append(data)
    return dataset

