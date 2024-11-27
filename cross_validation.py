import numpy as np

def make_fold(X, y, k):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices) 
    fold_sizes = n_samples // k
    folds = []

    # Diviser les indices en k folds
    split_indices = [indices[i * fold_sizes : (i + 1) * fold_sizes] for i in range(k)]
    
    for i in range(k):
        # Indices du fold de test
        test_indices = split_indices[i]
        
        # Indices pour les folds d'entraînement (tous sauf le i-ème)
        train_indices = np.concatenate([split_indices[j] for j in range(k) if j != i])

        # Créer les ensembles d'entraînement et de test
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        folds.append((X_train, X_test, y_train, y_test))

    return folds

def cross_validation(X, y, k, model, metric):
    scores=[]
    folds = make_fold(X, y, k)
    for i, fold in enumerate(folds):
        X_train, X_test, y_train, y_test = fold
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = metric(y_test, y_pred)
        scores.append({f"Fold_{i}" : score})
    return scores