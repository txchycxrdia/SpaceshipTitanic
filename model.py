import os
import shutil

import numpy as np
import pandas as pd
import requests

from sklearn.impute import SimpleImputer

import optuna
from optuna.integration import CatBoostPruningCallback

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from flask import Flask, request, jsonify, send_file

import uuid
import argparse
import warnings

app = Flask(__name__)


class My_Classifier_Model:
    def __init__(self):
        self.__X_train = None
        self.__y_train = None

    @staticmethod
    def __format_data(data):
        data[['GroupId', 'IdWithinGroup']] = data['PassengerId'].str.split('_', expand=True)
        data[['Deck', 'Num', 'Side']] = data['Cabin'].str.split('/', expand=True)
        data.drop(['Name', 'PassengerId', 'Cabin', 'IdWithinGroup', 'Num'], axis=1, inplace=True)
        data['GroupId'] = data['GroupId'].astype('float')

        # криосон
        data.loc[
            ((data['RoomService'] == 0.0) | data['RoomService'].isnull()) &
            ((data['FoodCourt'] == 0.0) | data['FoodCourt'].isnull()) &
            ((data['ShoppingMall'] == 0.0) | data['ShoppingMall'].isnull()) &
            ((data['Spa'] == 0.0) | data['Spa'].isnull()) &
            ((data['VRDeck'] == 0.0) | data['VRDeck'].isnull()) &
            (data['CryoSleep'].isnull()),
            'CryoSleep'
        ] = True

        data.loc[
            ((data['RoomService'] > 0.0) |
             (data['FoodCourt'] > 0.0) |
             (data['ShoppingMall'] > 0.0) |
             (data['Spa'] > 0.0) |
             (data['VRDeck'] > 0.0)) & (data['CryoSleep'].isnull()),
            'CryoSleep'
        ] = False

        # родная планета по палубе
        data.loc[
            (data['Deck'] == 'G') & (data['HomePlanet'].isnull()),
            'HomePlanet'
        ] = 'Earth'

        europa_decks = ['A', 'B', 'C', 'T']
        data.loc[
            (data['Deck'].isin(europa_decks)) & (data['HomePlanet'].isnull()),
            'HomePlanet'
        ] = 'Europa'

        data.loc[
            (data['Deck'] == 'F') & (data['HomePlanet'].isnull()),
            'HomePlanet'
        ] = 'Mars'  # спорно

        # палуба по родной планете
        home_planet_deck = data.groupby(
            ['HomePlanet', 'Deck']
        ).size().unstack().fillna(0)

        earth = home_planet_deck.loc['Earth']
        earth_proba = list(earth / sum(earth))

        europa = home_planet_deck.loc['Europa']
        europa_proba = list(europa / sum(europa))

        mars = home_planet_deck.loc['Mars']
        mars_proba = list(mars / sum(mars))

        decks = data['Deck'].unique()
        deck_values = sorted(decks[~pd.isnull(decks)])
        planet_proba = dict(
            zip(['Earth', 'Mars', 'Europa'], [earth_proba, mars_proba, europa_proba])
        )
        for planet in planet_proba.keys():
            planet_null_decks_shape = data.loc[
                (data['HomePlanet'] == planet) & (data['Deck'].isnull()),
                'Deck'
            ].shape[0]

            data.loc[
                (data['HomePlanet'] == planet) & (data['Deck'].isnull()),
                'Deck'
            ] = np.random.choice(deck_values, planet_null_decks_shape, p=planet_proba[planet])

        # возраст по медианному на планете
        for planet in ['Europa', 'Earth', 'Mars']:
            planet_median = data[data['HomePlanet'] == planet]['Age'].median()
            data.loc[
                (data["Age"].isnull()) & (data["HomePlanet"] == planet),
                "Age"
            ] = planet_median

        # заполнить оставшиеся пропуски
        categorical_columns = ['HomePlanet', 'Destination', 'Deck', 'Side']
        numerical_columns = list(set(data.columns) - set(categorical_columns) - set('CryoSleep'))

        for col in numerical_columns:
            si = SimpleImputer(strategy='median')
            data[[col]] = si.fit_transform(data[[col]])
            data[[col]] = si.transform(data[[col]])

        for col in categorical_columns:
            si = SimpleImputer(strategy='most_frequent')
            data[[col]] = si.fit_transform(data[[col]])
            data[[col]] = si.transform(data[[col]])

        # логарифмируем...
        for col in numerical_columns[1:-1]:
            data[col] = np.log(1 + data[col])

        return data

    def __objective(self, trial: optuna.Trial) -> float:
        _X_train, X_valid, _y_train, y_valid = train_test_split(
            self.__X_train, self.__y_train, test_size=0.25
        )

        params = {
            'objective': trial.suggest_categorical(
                'objective',
                ['Logloss', 'CrossEntropy']
            ),
            'colsample_bylevel': trial.suggest_float(
                'colsample_bylevel',
                0.01,
                0.1,
                log=True
            ),
            'depth': trial.suggest_int(
                'depth',
                1,
                12
            ),
            'boosting_type': trial.suggest_categorical(
                'boosting_type',
                ['Ordered', 'Plain']
            ),
            'bootstrap_type': trial.suggest_categorical(
                'bootstrap_type',
                ['Bayesian', 'Bernoulli', 'MVS']
            ),
            'used_ram_limit': '8gb',
            'eval_metric': 'Accuracy',
            'logging_level': 'Silent'
        }

        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float(
                'bagging_temperature', 0, 10
            )

        model = CatBoostClassifier(**params, cat_features=['HomePlanet', 'Destination', 'Deck', 'Side'])
        pruning_callback = CatBoostPruningCallback(trial, 'Accuracy')
        model.fit(
            _X_train,
            _y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
        )
        pruning_callback.check_pruned()
        predictions = model.predict(X_valid)
        prediction_labels = np.rint(predictions)
        accuracy = accuracy_score(y_valid, prediction_labels)

        return accuracy

    def train(self, dataset, path='./model/model.cbm'):
        train = pd.read_csv(dataset)
        X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1:]
        X_train = self.__format_data(X_train)
        self.__X_train = X_train
        self.__y_train = y_train
        warnings.filterwarnings("ignore")
        optuna.logging.disable_default_handler()
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction='maximize'
        )
        study.optimize(self.__objective, n_trials=100)
        model = CatBoostClassifier(**study.best_trial.params,
                                   cat_features=['HomePlanet', 'Destination', 'Deck', 'Side'],
                                   logging_level='Silent')
        model.fit(X_train, y_train.astype(int))
        model.save_model(path)

    def predict(self, dataset, path='./data/results.csv'):
        X_test = self.__format_data(pd.read_csv(dataset))
        model = CatBoostClassifier().load_model('./data/model/model.cbm')
        predicted = model.predict(X_test)
        result = pd.DataFrame()
        result['PassengerId'] = pd.read_csv(dataset)['PassengerId']
        result['Transported'] = pd.Series(predicted).astype(bool)
        result.to_csv(path, index=False)


model = My_Classifier_Model()


@app.route('/train', methods=['POST'])
def train():
    request_id = str(uuid.uuid4())
    try:
        os.mkdir(f'./request_data/{request_id}')
        file = request.files['file']
        file_path = f'./request_data/{request_id}/' + file.filename
        file.save(file_path)
        model.train(file_path, path=f'./request_data/{request_id}/model.cbm')
        response = send_file(f'./request_data/{request_id}/model.cbm', as_attachment=True)
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(f'./request_data/{request_id}')


@app.route('/predict', methods=['POST'])
def predict():
    request_id = str(uuid.uuid4())
    try:
        os.mkdir(f'./request_data/{request_id}')
        file = request.files['file']
        file_path = f'./request_data/{request_id}/' + file.filename
        file.save(file_path)
        model.predict(file_path, path=f'./request_data/{request_id}/results.csv')
        response = send_file(f'./request_data/{request_id}/results.csv', as_attachment=True)
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(f'./request_data/{request_id}')


def send_request(url, dataset, command, save_path):
    url.rstrip('/')
    if command == 'train' or command == 'predict':
        response = requests.post(url + '/' + command, files={'file': open(dataset, 'rb')})
    else:
        return 'Wrong command'
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return 'Request completed successfully'
    else:
        return response.content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand')

    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.add_argument("--dataset", help='Path to the training dataset', required=True)

    parser_predict = subparsers.add_parser('predict', help='Predict using the model')
    parser_predict.add_argument("--dataset", help='Path to the evaluation dataset', required=True)

    parser_request = subparsers.add_parser('request', help='Send request to server with this model')
    parser_request.add_argument("--url", help="URL of the server", required=True)
    parser_request.add_argument("--dataset", help='Path to the dataset', required=True)
    parser_request.add_argument('--command', help='Command to execute (train or predict)', required=True)
    parser_request.add_argument("--save_path", help='Path to save response data', required=True)

    parser_run = subparsers.add_parser('run', help='Run Flask application')

    args = parser.parse_args()

    if args.subcommand == "train":
        model.train(args.dataset)
    elif args.subcommand == "predict":
        model.predict(args.dataset)
    elif args.subcommand == "request":
        print(send_request(args.url, args.dataset, args.command, args.save_path))
    elif args.subcommand == "run":
        app.run(host='localhost', port=5000, debug=False)
    else:
        parser.print_help()
