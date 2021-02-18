import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn import tree
from matplotlib import pyplot as plt


if __name__ == "__main__":
    input_file = "data/cereal.csv"

    # easily pull in csv data with pandas dataframes
    dataset = pd.read_csv(input_file)

    # print(dataset.head())
    # print(dataset.describe())

    X = dataset.drop(['name', 'type', 'rating'], axis=1)

    mfr_encoder = LabelBinarizer()
    mfr_encoder.fit(X['mfr'])
    transformed = mfr_encoder.transform(X['mfr'])
    ohe_df = pd.DataFrame(transformed)
    X = pd.concat([X, ohe_df], axis=1).drop(['mfr'], axis=1)

    # print(X.head())
    # print(X.dtypes)

    y = dataset['rating']
    # print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    # print(df)

    mfr_names = ['American Home Food Products',
                 'General Mills',
                 'Kelloggs',
                 'Nabisco',
                 'Post',
                 'Quaker Oats',
                 'Ralston Purina']
    feature_names = list(X.columns.values)[:-7] + mfr_names
    text_representation = tree.export_text(dt, feature_names=feature_names)
    print(text_representation)

    fig = plt.figure(figsize=(100, 100))
    fig.set_dpi(100.0)
    _ = tree.plot_tree(dt,
                       feature_names=feature_names,
                       filled=True)
    fig.savefig("decision_tree.png")
