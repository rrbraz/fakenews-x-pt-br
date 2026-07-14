import copy

import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_classifier(trial, classifier_type, seed=42):
    """
    Create and configure a classifier based on the type and trial suggestions
    
    Args:
        trial: Optuna trial instance
        classifier_type: Type of classifier to create
        seed: Random seed for reproducibility
        
    Returns:
        model: Configured classifier instance
    """
    if classifier_type == 'LogisticRegression':
        C = trial.suggest_float('C', 1e-5, 100, log=True)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
        max_iter = trial.suggest_int('max_iter', 100, 1000)

        return LogisticRegression(
            C=C,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            random_state=seed
        )

    elif classifier_type == 'SVC':
        C = trial.suggest_float('C', 1e-5, 100, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])

        return SVC(
            C=C,
            kernel=kernel,
            gamma=gamma if kernel == 'rbf' else 'scale',
            class_weight=class_weight,
            probability=True,
            random_state=seed
        )

    elif classifier_type == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])

        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=seed
        )

    elif classifier_type == 'GradientBoosting':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)

        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=seed
        )

    elif classifier_type == 'MultinomialNB':
        alpha = trial.suggest_float('alpha', 0.01, 10.0, log=True)

        return MultinomialNB(alpha=alpha)

    elif classifier_type == 'DecisionTree':
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])

        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=seed
        )

    elif classifier_type == 'KNeighbors':
        n_neighbors = trial.suggest_int('n_neighbors', 3, 15)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])

        return KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights
        )

    elif classifier_type == 'LightGBM':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        num_leaves = trial.suggest_int('num_leaves', 10, 100)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 10.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0)

        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=seed,
            verbose=-1,  # Suppress LightGBM output
            force_col_wise=True  # For better performance with wide datasets
        )

    elif classifier_type == 'XGBoost':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 10.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0)

        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=seed,
            verbosity=0,  # Suppress XGBoost output
            use_label_encoder=False,
            eval_metric='logloss'
        )

    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def optimize_sklearn_classifier(
    train_features, train_labels, val_features, val_labels, 
    classifier_type, n_trials=10, seed=42
):
    """
    Optimize hyperparameters for a scikit-learn classifier using Optuna
    
    Args:
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        classifier_type: Type of classifier to optimize
        n_trials: Number of trials for optimization
        seed: Random seed for reproducibility
        
    Returns:
        best_model: Best model found
        best_score: Best validation F1 score
        best_params: Best hyperparameters found
    """
    best_model = None
    best_score = 0
    
    def objective(trial):
        nonlocal best_model, best_score
        
        # Initialize the model with current hyperparameters
        model = get_classifier(trial, classifier_type, seed=seed)
        
        # Train the model
        try:
            model.fit(train_features, train_labels)
            
            # Predict on validation set
            val_preds = model.predict(val_features)
            
            # Calculate F1 score
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            
            if val_f1 > best_score:
                best_score = val_f1
                best_model = copy.deepcopy(model)
            
            return val_f1
        except Exception as e:
            # In case of error, discard this trial
            print(e)
            raise optuna.exceptions.TrialPruned(f"Trial failed with error: {e}")
    
    # Create and run the study
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    
    return best_model, best_score, best_params, study


def optimize_multiple_classifiers(
    train_features, train_labels, val_features, val_labels,
    classifier_types=None, n_trials_per_classifier=10, seed=42
):
    """
    Optimize hyperparameters for multiple scikit-learn classifiers
    
    Args:
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        classifier_types: List of classifier types to optimize (default: all supported types)
        n_trials_per_classifier: Number of trials per classifier
        seed: Random seed for reproducibility
        
    Returns:
        classifier_results: Dictionary with results for each classifier
        best_classifier_type: Type of the best classifier
        best_model: Best model found
    """
    if classifier_types is None:
        classifier_types = [
            'LogisticRegression', 
            'SVC', 
            'RandomForest', 
            'GradientBoosting', 
            'MultinomialNB', 
            'DecisionTree', 
            'KNeighbors',
            'LightGBM',
            'XGBoost'
        ]
    
    classifier_results = {}
    best_model = None
    best_model_f1 = 0
    best_classifier_type = None
    
    for classifier_type in classifier_types:
        print(f"\n---- Optimizing {classifier_type} ----")
        
        model, score, params, study = optimize_sklearn_classifier(
            train_features, train_labels, val_features, val_labels,
            classifier_type, n_trials=n_trials_per_classifier, seed=seed
        )
        
        # Store results
        classifier_results[classifier_type] = {
            'model': model,
            'f1': score,
            'accuracy': accuracy_score(val_labels, model.predict(val_features)),
            'params': params,
            'study': study
        }
        
        print(f"Best F1 Score for {classifier_type}: {score:.4f}")
        print(f"Best hyperparameters found:")
        for param, value in params.items():
            print(f"  {param}: {value}")
        
        # Update best model if this one is better
        if score > best_model_f1:
            best_model_f1 = score
            best_model = model
            best_classifier_type = classifier_type
    
    return classifier_results, best_classifier_type, best_model


def evaluate_sklearn_model(model, features, labels):
    """
    Evaluate a scikit-learn model on a dataset
    
    Args:
        model: Scikit-learn model to evaluate
        features: Features to evaluate on
        labels: True labels
        
    Returns:
        preds: Predicted labels
        accuracy: Accuracy score
        f1: F1 score
    """
    # Make predictions
    preds = model.predict(features)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    
    return preds, accuracy, f1