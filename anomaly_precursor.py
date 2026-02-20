import pandas as pd
import numpy as np

from typing import Literal

from tsfresh import extract_features
from sklearn.tree import DecisionTreeClassifier
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.feature_extraction.settings import MinimalFCParameters

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

random_seed = 42
min_samples_leaf=10
min_samples_split=20
max_depth=6
max_leaf_nodes=None
min_impurity_decrease=0.0
class_weight={0: 1, 1: 3}
# should be high precision with moderately label events or moderate precision with very label events

DT_PARAMS = {
    'min_samples_leaf': min_samples_leaf,
    'min_samples_split': min_samples_split,
    'max_depth': max_depth,
    'max_leaf_nodes': max_leaf_nodes,
    'min_impurity_decrease': min_impurity_decrease,
    'class_weight': class_weight,
    'random_state': random_seed
}

PRECISION_PARAMS = {
    'normal': {
        'min_precision': 0.5,
        'bad_percentage': 0.4,
        'bad_threshold': 0.005
    },
    'light': {
        'min_precision': 0.4,
        'bad_percentage': 0.3,
        'bad_threshold': 0
    },
    'moderate': {
        'min_precision': 0.3,
        'bad_percentage': 0.2,
        'bad_threshold': -0.01
    },
    'severe': {
        'min_precision': 0.2,
        'bad_percentage': 0.1,
        'bad_threshold': -0.02
    }
}

window_length = 20
data_step = 1

def generate_input_data(L = 1000 , horizon = 5, delay = 1 , incomplete_x10 : bool = True):
    """
    generate input data for anomaly precursor
    Args:
        incomplete_x10: whether to make x10 incomplete , compute label based on y
    Returns:
        df: pandas dataframe with columns: time, y, x1..x10
    
    """
    df = pd.DataFrame({
        'x1' : np.random.randn(L),
        'x2' : np.random.randn(L),
        'x3' : np.random.randn(L),
        'x4' : np.random.randn(L),
        'x5' : np.random.randn(L),
        'x6' : np.random.randn(L),
        'x7' : np.random.randn(L),
        'x8' : np.random.randn(L),
        'x9' : np.random.randn(L),
        'x10' : np.random.randn(L),
        'y' : np.random.randn(L),
        'time' : np.arange(L),
        'id' : 1 
    })
    if incomplete_x10:
        df['x10'] = np.where(np.arange(L) > L / 2 , np.random.randn(L) > 0, np.nan)

    for col in df.columns.difference(['time' , 'id' , 'y']):
        if df[col].isna().sum() > 0:
            df[f'{col}_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(0)

    threshold = df['y'].rolling(horizon).sum().expanding(1).apply(lambda x : x.quantile(0.1))
    df['label'] = (df['y'].rolling(horizon).sum().shift(-horizon+1-delay) < threshold.where(threshold < -0.01, -0.01)).fillna(False).astype(int)

    df = df.set_index(['id' , 'time']).sort_index()

    print(f"label event ratio: {df['label'].mean():.3f}")
    return df

def get_input_data():
    """
    generate input data for anomaly precursor
    Args:
        incomplete_x10: whether to make x10 incomplete , compute label based on y
    Returns:
        df: pandas dataframe with columns: time, y, x1..x10
    
    """
    from src.proj import CALENDAR
    dates = CALENDAR.td_within(20100101 , 20250418)
    index_inputs : dict[str, pd.DataFrame] = {}
    index_inputs['microcap'] = pd.read_feather('data/DataBase/DB_index_daily_custom/microcap_400.feather').\
        rename(columns = {'trade_date' : 'date'}).set_index('date').reindex(dates)
    for i , code in enumerate(['000300.SH' , '000905.SH' , '000852.SH' , '399006.SZ' , '000688.SH' , '000985.CSI' , '000016.SH']):
        index_inputs[code] = pd.read_feather(f'data/DataBase/DB_index_daily_ts/{code}.feather').loc[:,['trade_date' , 'pct_chg']].\
            rename(columns = {'trade_date' : 'date'}).set_index('date').reindex(dates)

    df = pd.DataFrame(index = pd.Index(dates , name = 'date'))
    df['x0'] = index_inputs['000985.CSI']['pct_chg'] / 100
    df['x1'] = index_inputs['microcap']['pct_chg'] / 100
    df['x2'] = index_inputs['000300.SH']['pct_chg'] / 100 - df['x0']
    df['x3'] = index_inputs['000905.SH']['pct_chg'] / 100 - df['x0']
    df['x4'] = index_inputs['000852.SH']['pct_chg'] / 100 - df['x0']
    df['x5'] = index_inputs['399006.SZ']['pct_chg'] / 100 - df['x0']
    df['x6'] = index_inputs['000688.SH']['pct_chg'] / 100 - df['x0']
    df['x7'] = index_inputs['000016.SH']['pct_chg'] / 100 - df['x0']
 
    from src.res.factor.util.agency.portfolio_accountant import PortfolioAccount
    acc = PortfolioAccount.load('models/gru_avg/snapshot/detailed_alpha/t50_fmp_test/account/top.best.univ.top_50.lag0.tar')
    y = acc.df.reset_index().loc[:,['end','pf']].rename(columns = {'end' : 'date' , 'pf' : 'y'}).set_index('date').reindex(dates)
    df['y'] = y['y'].where(~y['y'].isna() , index_inputs['000852.SH']['pct_chg'] / 100)
    df = df.assign(time = np.arange(len(df)) , id = 1).set_index(['id' , 'time'] , append=True)

    from src.proj import DB
    market_risk = DB.load('market_daily' , 'risk')
    if not market_risk.empty:
        df = df.join(market_risk.set_index('date') , how = 'left')

    return df

def preprocess(df : pd.DataFrame , horizon = 5 , delay = 1 , bad_percentage : float = 0.5 , bad_threshold : float = 0):
    # standardize the data based on training set
    cols = df.columns.to_list()
    threshold = df['y'].rolling(horizon).sum().rolling(750 , min_periods=1).quantile(bad_percentage)
    df['y_rolling'] = df['y'].rolling(horizon).sum()
    df['threshold'] = threshold.where(threshold < bad_threshold, bad_threshold)

    df['label'] = (df['y_rolling'].shift(-horizon+1-delay) < df['threshold']).fillna(False).astype(int)
    df.loc[df.index[:20] , 'label'] = -1
    df['y_raw'] = df['y']

    means = df.query('date < 20170101').loc[:,cols].mean(axis = 0)
    stds = df.query('date < 20170101').loc[:,cols].std(axis = 0)
    means = means.fillna(0)
    stds = stds.fillna(0).where(stds.fillna(0) != 0 , 1)

    df.loc[:,cols] = (df.loc[:,cols] - means) / stds

    for col in cols:
        if df[col].isna().sum() > 0:
            df[f'{col}_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(0)

    df = df.sort_index()
    print(f"label event ratio: {len(df.query('label == 1')) / len(df.query('label != -1')):.3f} , compare to bad_percentage: {bad_percentage:.2f} , bad_threshold: {bad_threshold:.2f}")
    return df.drop(columns=['y_raw' , 'threshold' , 'y_rolling'])

def get_rule(tree_obj, feature_names, leaf_id):
    """
    Return the decision path for a given leaf as a string.
    
    Parameters
    ----------
    tree_obj : sklearn.tree._tree.Tree
        The underlying tree object (dt.tree_).
    feature_names : list of str
        Names of the input features.
    leaf_id : int
        ID of the target leaf node.
    """
    # Walk upward from leaf to root
    node = leaf_id
    path = []
    while node != 0:   # 0 is always the root
        # Find the parent of 'node'
        parent = np.where((tree_obj.children_left == node) | (tree_obj.children_right == node))[0]
        if len(parent) == 0:
            break   # safety, should not happen
        parent = parent[0]
        
        # Determine if node is left or right child
        feature_idx = tree_obj.feature[parent]
        threshold_val = tree_obj.threshold[parent]
        if node == tree_obj.children_left[parent]:
            path.append(f"{feature_names[feature_idx]} ≤ {threshold_val:.3f}")
        else:
            path.append(f"{feature_names[feature_idx]} > {threshold_val:.3f}")
        
        node = parent   # move up
    
    # Reverse to get path from root → leaf
    path.reverse()
    return " AND ".join(path)

class ClassicDecisionTree:
    def __init__(self , df : pd.DataFrame , lable_type : Literal['normal' , 'light' , 'moderate', 'severe'] = 'normal'):
        self.min_precision = PRECISION_PARAMS[lable_type]['min_precision']
        self.bad_percentage = PRECISION_PARAMS[lable_type]['bad_percentage']
        self.bad_threshold = PRECISION_PARAMS[lable_type]['bad_threshold']
        
        self.df = preprocess(df , horizon = 5 , delay = 1 , bad_percentage = self.bad_percentage , bad_threshold = self.bad_threshold)
        
        self.dataset : dict[str, pd.DataFrame] = {}

    def feature_extraction(self):
        features : list[pd.DataFrame] = []
        for window_size in [5, 10, 20]:
            df_rolled = roll_time_series(
                self.df.reset_index(drop=False).drop(columns=['label'] , errors='ignore').dropna(axis=1 , how = 'any'),
                column_id='id',
                column_sort='time',
                max_timeshift=window_size,  # produces windows of length 1..20
                min_timeshift=1   # optional: only keep windows with ≥5 observations
            )
            df_features = extract_features(df_rolled, 
                                           column_id='id', 
                                           column_sort='time', 
                                           default_fc_parameters=MinimalFCParameters())
            assert isinstance(df_features, pd.DataFrame)
            df_features.columns = [f'{col}_{window_size}' for col in df_features.columns]
            df_features = df_features
            features.append(df_features)
        self.features = features

    def data_split(self):
        df = self.df
        for df_feature in self.features:
            df = df.join(df_feature.reset_index(drop = False).rename(columns = {'level_0' : 'id' , 'level_1' : 'time'}).set_index(['id' , 'time']))
        self.dataset['train'] = df.query('date < 20220101').query('label != -1').iloc[::data_step]
        self.dataset['valid'] = df.query('date >= 20220101').query('label != -1')

    def fit(self):
        # 5. Decision tree for high precision patterns
        dt = DecisionTreeClassifier(
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            random_state=random_seed,
        )
        dt.fit(self.dataset['train'].drop(['time','y','label'], axis=1 , errors='ignore'), self.dataset['train']['label'])
        self.dt = dt

    def evaluate(self):
        # 6. Extract paths with >= min_precision label in training leaves
        # Evaluate precision on validation set
        y_pred = self.dt.predict(self.dataset['valid'].drop(['time','y','label'], axis=1 , errors='ignore'))
        if y_pred.sum() == 0:
            precision = 0
        else:
            precision = (self.dataset['valid']['label'] & y_pred).sum() / y_pred.sum()
        print(f"Overall precision: {precision:.2f}")

        # Inspect leaf probabilities
        leaf_ids_train = self.dt.apply(self.dataset['train'].drop(['time','y','label'], axis=1 , errors='ignore'))
        train_labels = self.dataset['train']['label']

        leaf_stats = {}
        for leaf in np.unique(leaf_ids_train):
            mask = leaf_ids_train == leaf
            leaf_precision = train_labels[mask].mean()
            leaf_support = mask.sum()
            leaf_stats[leaf] = {'precision': leaf_precision, 'support': leaf_support}

        self.high_prec_leaves = [leaf for leaf, stats in leaf_stats.items() if stats['precision'] >= self.min_precision]
        print(f"\nFound {len(self.high_prec_leaves)} leaves with ≥{self.min_precision:.0%} precision in training.")

        # Now define feature names (uncommented)
        feature_names = self.dataset['train'].columns.tolist()

        for leaf in self.high_prec_leaves[:5]:
            rule = get_rule(self.dt.tree_, feature_names, leaf)
            print(f"Leaf {leaf}: precision={leaf_stats[leaf]['precision']:.3f}, "
                f"support={leaf_stats[leaf]['support']}, rule: {rule}")

    def validate(self):
        val_leaf_ids = self.dt.apply(self.dataset['valid'].drop(['time','y','label'], axis=1 , errors='ignore'))
        val_labels = self.dataset['valid']['label']
        val_precisions = []
        val_nums = []
        for leaf in self.high_prec_leaves:
            mask = val_leaf_ids == leaf
            if mask.sum() > 0:
                prec = val_labels[mask].mean()
                val_nums.append(mask.sum())
                val_precisions.append(prec)
                print(f"Leaf {leaf} validation precision: {prec:.3f} (n={mask.sum()})")
            else:
                val_precisions.append(0)
                print(f"Leaf {leaf} has no validation samples")
        print(f"ratio of label events in validation set: {self.dataset['valid']['label'].mean():.2%} ({self.dataset['valid']['label'].sum().astype(int).item()}/{len(self.dataset['valid']['label'])})")
        if val_precisions:
            print(f"Average val precision of high-precision leaves: {sum(val_precisions) / sum(val_nums):.2%} ({sum(val_precisions)}/{sum(val_nums)} samples)")
        else:
            print("No high-precision leaves appeared in validation set.")

    def run(self):
        self.feature_extraction()
        self.data_split()
        self.fit()
        self.evaluate()
        self.validate()

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).permute(0, 2, 1)  # (batch, channels, length)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNNFeatureExtractor(nn.Module):
    def __init__(self, n_channels, embedding_dim=64):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, embedding_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.global_pool(x).squeeze(-1)   # (batch, 128)
        x = self.dropout(x)
        x = self.fc(x) 
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets) -> torch.Tensor:
        # inputs: raw logits (no sigmoid)
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)  # p_t
        focal = self.alpha * bce * (1 - pt) ** self.gamma 
        if self.reduction == 'mean':
            return focal.mean()
        else:
            return focal.sum()

class GRUFeatureExtractor(nn.Module):
    def __init__(self, n_features, embedding_dim=64, hidden_size=128, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (batch, features, length) -> permute to (batch, length, features)
        x = x.permute(0, 2, 1)                # (batch, length, features)
        # GRU forward
        out, _ = self.gru(x)                   # out: (batch, length, hidden_size)
        # Use last time step's output
        last_out = out[:, -1, :]                # (batch, hidden_size)
        last_out = self.bn(last_out)
        last_out = self.dropout(last_out)
        emb = self.fc(last_out)                 # (batch, embedding_dim)
        return emb

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, n_features, embedding_dim=64, hidden_size=128, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 2, 1)                # (batch, length, features)
        out, _ = self.lstm(x)                  # out: (batch, length, hidden_size)
        last_out = out[:, -1, :]                # (batch, hidden_size)
        last_out = self.bn(last_out)
        last_out = self.dropout(last_out)
        emb = self.fc(last_out)
        return emb

class TransformerFeatureExtractor(nn.Module):
    def __init__(self, n_features, embedding_dim=64, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        # Project input features to d_model dimension
        self.input_proj = nn.Linear(n_features, d_model)
        # Positional encoding (learnable or fixed) – here we use a simple learnable positional embedding
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))  # max length 1000, adjust as needed
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, features, length) -> (batch, length, features)
        x = x.permute(0, 2, 1)                # (batch, length, features)
        # Project to d_model
        x = self.input_proj(x)                 # (batch, length, d_model)
        # Add positional encoding (trim to actual length)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        # Transformer expects (batch, length, d_model)
        x = self.transformer_encoder(x)         # (batch, length, d_model)
        # Use the representation of the last token (or global average)
        x = x.mean(dim=1)                       # (batch, d_model)
        x = self.dropout(x)
        emb = self.fc(x)                         # (batch, embedding_dim)
        return emb

# -------------------------------
# 4. Main NNDecisionTree Class
# -------------------------------
class NNDecisionTree:
    def __init__(self, df: pd.DataFrame, lable_type : Literal['normal' , 'light' , 'moderate', 'severe'] = 'normal' , 
                 nn_type : Literal['cnn', 'gru', 'lstm', 'transformer'] = 'cnn', embedding_dim=64, 
                 learning_rate=1e-4, epochs=100, early_stopping=10, batch_size : int | None = None, 
                 focal_alpha=0.25, focal_gamma=2.0,
                 decision_tree_params=None):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns 'time', 'y', 'label' and all feature columns.
        nn_type : str
            One of {'cnn', 'gru', 'lstm', 'transformer'}.
        embedding_dim : int
            Dimension of the embedding extracted from the neural net.
        learning_rate : float
        epochs : int
        batch_size : int
        focal_alpha, focal_gamma : focal loss parameters.
        decision_tree_params : dict
            Parameters for DecisionTreeClassifier (e.g., min_samples_leaf, max_depth).
        random_seed : int
        """
        self.min_precision = PRECISION_PARAMS[lable_type]['min_precision']
        self.bad_percentage = PRECISION_PARAMS[lable_type]['bad_percentage']
        self.bad_threshold = PRECISION_PARAMS[lable_type]['bad_threshold']
        self.df = preprocess(df , horizon = 5 , delay = 1 , bad_percentage = self.bad_percentage , bad_threshold = self.bad_threshold)
        self.nn_type = nn_type
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        if decision_tree_params is None:
            self.dt_params = DT_PARAMS
        else:
            self.dt_params = decision_tree_params

        self.dataset = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                   'mps' if torch.backends.mps.is_available() else 'cpu')
        self.high_prec_leaves = []

    def data_split(self):
        """
        Create rolling windows and train/validation split.
        Assumes df is sorted by time.
        """
        feature_columns = [c for c in self.df.columns if c not in ['time', 'y', 'label']]

        # Split indices chronologically (80% train, 20% val)
        split_idx = int(0.8 * len(self.df))
        df_train = self.df.iloc[:split_idx].query('label != -1').iloc[::data_step]
        df_val = self.df.iloc[split_idx - window_length:].query('label != -1')  # allow overlap for windows

        X_data, y_data = [], []
        train_indices = []

        # Training windows
        for t in range(window_length, len(df_train)):
            window = df_train[feature_columns].iloc[t-window_length:t].values
            label = df_train['label'].iloc[t]
            X_data.append(window)
            y_data.append(label)
            train_indices.append(len(X_data)-1)

        val_indices = []
        # Validation windows
        for t in range(window_length, len(df_val)):
            window = df_val[feature_columns].iloc[t-window_length:t].values
            label = df_val['label'].iloc[t]
            X_data.append(window)
            y_data.append(label)
            val_indices.append(len(X_data)-1)

        X_data = np.array(X_data)          # (n_samples, L, n_features)
        y_data = np.array(y_data)

        self.dataset['X_data'] = X_data
        self.dataset['y_data'] = y_data
        self.dataset['train_idx'] = np.array(train_indices)
        self.dataset['val_idx'] = np.array(val_indices)

        self.dataset['X_train'] = X_data[train_indices]
        self.dataset['X_val'] = X_data[val_indices]
        self.dataset['y_train'] = y_data[train_indices]
        self.dataset['y_val'] = y_data[val_indices]

        print(f"Total windows: {X_data.shape[0]}, Features: {X_data.shape[2]}")
        print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    def _build_nn(self, n_features):
        """Factory method to create the specified backbone."""
        if self.nn_type == 'cnn':
            return CNNFeatureExtractor(n_features, self.embedding_dim)
        elif self.nn_type == 'gru':
            return GRUFeatureExtractor(n_features, self.embedding_dim)
        elif self.nn_type == 'lstm':
            return LSTMFeatureExtractor(n_features, self.embedding_dim)
        elif self.nn_type == 'transformer':
            return TransformerFeatureExtractor(n_features, self.embedding_dim)
        else:
            raise ValueError(f"Unknown nn_type: {self.nn_type}")

    def fit_nn(self):
        """Train the neural feature extractor (with a temporary classifier head)."""
        n_features = self.dataset['X_train'].shape[2]

        # Datasets & loaders
        train_ds = TimeSeriesDataset(self.dataset['X_train'], self.dataset['y_train'])
        val_ds = TimeSeriesDataset(self.dataset['X_val'], self.dataset['y_val'])
        train_loader = DataLoader(train_ds, batch_size=self.batch_size if self.batch_size else len(train_ds), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size if self.batch_size else len(val_ds), shuffle=False)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Model, optimizer, loss
        backbone = self._build_nn(n_features).to(self.device)
        # Temporary classifier head
        classifier = nn.Linear(self.embedding_dim, 1).to(self.device)
        # Combine for training (we'll handle forward manually)
        optimizer = optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=self.lr)
        criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)

        self.model = backbone  # store backbone for later
        best_val_loss = float('inf')
        best_epoch = 0
        best_state_dicts = {}

        for epoch in range(self.epochs):
            backbone.train()
            classifier.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                emb = backbone(X_batch)          # (batch, embedding_dim)
                logits = classifier(emb)          # (batch, 1)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)

            # Optional validation loss
            backbone.eval()
            classifier.eval()
            val_loss = 0
            y_num = 0
            real_bad = 0
            pred_bad = 0
            precise_bad = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    emb = backbone(X_batch)
                    logits = classifier(emb)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()
                    y_num += len(y_batch)
                    real_bad += y_batch.sum().item()
                    pred_bad += (logits.sigmoid() > self.min_precision).sum().item()
                    precise_bad += ((logits.sigmoid() > self.min_precision) * y_batch).sum().item()
            avg_val_loss = val_loss / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_state_dicts['backbone'] = backbone.state_dict()
                best_state_dicts['classifier'] = classifier.state_dict()
            else:
                if epoch - best_epoch >= self.early_stopping:
                    print(f"Early stopping at epoch {epoch+1}")
                    print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Real Bad: {int(real_bad):0d}/{int(y_num):0d} | Pred Bad: {int(pred_bad):0d}/{int(y_num):0d} | Precise Bad: {int(precise_bad):0d}/{int(pred_bad):0d}")
                    self.model.load_state_dict(best_state_dicts['backbone'])
                    break

            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Real Bad: {int(real_bad):0d}/{int(y_num):0d} | Pred Bad: {int(pred_bad):0d}/{int(y_num):0d} | Precise Bad: {int(precise_bad):0d}/{int(pred_bad):0d}")

    def extract_embeddings(self):
        """Forward all data through the trained backbone to get embeddings."""
        self.model.eval()
        with torch.no_grad():
            X_full = torch.FloatTensor(self.dataset['X_data']).permute(0, 2, 1).to(self.device)
            embeddings = self.model(X_full).cpu().numpy()   # (n_samples, embedding_dim)
        self.dataset['embeddings'] = embeddings
        self.dataset['emb_train'] = embeddings[self.dataset['train_idx']]
        self.dataset['emb_val'] = embeddings[self.dataset['val_idx']]

    def fit_decision_tree(self):
        """Train decision tree on embeddings and find high-precision leaves."""
        dt = DecisionTreeClassifier(**self.dt_params)
        dt.fit(self.dataset['emb_train'], self.dataset['y_train'])

        # Compute leaf stats on training set
        leaf_ids = dt.apply(self.dataset['emb_train'])
        leaf_stats = {}
        for leaf in np.unique(leaf_ids):
            mask = leaf_ids == leaf
            leaf_precision = self.dataset['y_train'][mask].mean()
            leaf_support = mask.sum()
            leaf_stats[leaf] = {'precision': leaf_precision, 'support': leaf_support}

        # Keep leaves with precision >= min_precision
        high_prec = [leaf for leaf, s in leaf_stats.items() if s['precision'] >= self.min_precision]
        self.dt = dt
        self.leaf_stats = leaf_stats
        self.high_prec_leaves = high_prec

        print(f"Found {len(high_prec)} leaves with ≥{self.min_precision:.0%} precision (train).")

        # Print a few example rules
        feature_names = [f"embed_{i}" for i in range(self.embedding_dim)]
        for leaf in high_prec[:5]:
            rule = get_rule(dt.tree_, feature_names, leaf)
            print(f"Leaf {leaf}: prec={leaf_stats[leaf]['precision']:.3f}, sup={leaf_stats[leaf]['support']}, rule: {rule}")

    def validate_decision_tree(self):
        """Evaluate high-precision leaves on validation set."""
        val_leaf_ids = self.dt.apply(self.dataset['emb_val'])
        val_labels = self.dataset['y_val']
        val_nums = []
        val_precisions = []
        for leaf in self.high_prec_leaves:
            mask = val_leaf_ids == leaf
            if mask.sum() > 0:
                prec = val_labels[mask].mean()
                val_nums.append(mask.sum())
                val_precisions.append(val_labels[mask].sum())
                print(f"Leaf {leaf} val prec: {prec:.3f} (n={mask.sum()})")
            else:
                print(f"Leaf {leaf} has no validation samples")
        print(f"ratio of label events in validation set: {val_labels.mean():.2%} ({val_labels.sum().astype(int).item()}/{len(val_labels)})")
        if val_precisions:
            print(f"Average val precision of high-precision leaves: {sum(val_precisions) / sum(val_nums):.2%} ({sum(val_precisions)}/{sum(val_nums)} samples)")
        else:
            print("No high-precision leaves appeared in validation set.")

    def run(self):
        self.data_split()
        self.fit_nn()
        self.extract_embeddings()
        self.fit_decision_tree()
        self.validate_decision_tree()