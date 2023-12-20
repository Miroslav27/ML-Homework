

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import joblib
    from sklearn.preprocessing import MinMaxScaler


    ds = pd.read_csv("kaggle/input/kc-house-data/kc_house_data.csv") #Folder Should be created!!!

    print(ds.shape, ds.describe(), ds.head(), ds.info(), ds.isna().sum(), sep="\n")

    ds = ds.fillna(0)
    ds.isna().sum()
    ds.drop("date", axis=1, inplace=True)
    ds.head()

    mult_price_cor_columns = abs(ds.corrwith(ds.price)) > 0.3  # those columns will have impact
    print("Correleation:\n", mult_price_cor_columns)
    multi_ds = ds.loc[:, mult_price_cor_columns]

    plt.scatter(ds.price, ds.sqft_living, alpha=0.1)
    plt.title('Scatter Plot of price vs sqrt_living')
    plt.xlabel('price')
    plt.ylabel('sqrt_living')
    plt.savefig('kaggle/working/Price_vs_Sqft_scatterplot.png', ) #Folder Should be created!!!
    plt.show()



    # Create a pairplot with correlation coefficients
    sns.pairplot(ds[['price', 'sqft_living']])
    plt.savefig('kaggle/working/snspairplot_Price_vs_Sqft_pairplot.png', )
    plt.show()
    sns.regplot(x='price', y='sqft_living', data=ds, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    plt.savefig('kaggle/working/Price_vs_Sqft_snsregplot.png', )
    plt.show()

    # Calculate the linear regression line
    slope, intercept = np.polyfit(ds['sqft_living'], ds['price'], 1)

    # Create a scatter plot
    plt.scatter(ds['sqft_living'], ds['price'], alpha=0.5)

    # Plot the linear regression line
    plt.plot(ds['sqft_living'], slope * ds['sqft_living'] + intercept, color='red', label='Linear Regression Line')
    plt.ticklabel_format(style='plain', axis='both')
    plt.title('Price vs Sqft')
    plt.xlabel('sqft_living', )
    plt.ylabel('price')
    plt.savefig('kaggle/working/Price_vs_Sqft_plot.png', )
    plt.show()



    X = np.array(ds.sqft_living).reshape(-1, 1)
    y = np.array(ds.price)
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Visualize the results
    plt.scatter(X_test, y_test, color='black', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.title('Linear Regression: Actual vs. Predicted')
    plt.xlabel('sqft_living')
    plt.ylabel('price')
    plt.legend()
    plt.savefig('kaggle/working/linear_regression_model_plot.png', )
    plt.show()

    # Save the trained model to a file using joblib
    joblib.dump(model, 'kaggle/working/linear_regression_model_single.joblib')



    scaler = MinMaxScaler(feature_range=(0, 1))

    y = multi_ds.price
    X = multi_ds.drop(columns="price")

    model_multi = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(X_test_scaled[:5])

    model_multi.fit(X_train_scaled, y_train)
    y_pred = model_multi.predict(X_test_scaled)

    mse_multi = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse_multi}')

    # Visualize the results
    plt.scatter(X_test_scaled[:, 2], y_test, color='black', label='Actual')
    plt.scatter(X_test_scaled[:, 2], y_pred, color='red', label='Predicted', alpha=0.15)
    plt.title('Linear Regression: Actual vs. Predicted')
    plt.xlabel('sqft_living')
    plt.ylabel('price')
    plt.legend()
    plt.savefig('kaggle/working/linear_regression_model_multi_scaled_plot.png', )
    plt.show()
    joblib.dump(model, 'kaggle/working/linear_regression_model_multi_scaled.joblib')

    X_b = np.c_[np.ones((len(X), 1)), X]

    # Обчислити аналітичне рішення за допомогою normal equation
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    # theta містить знайдені значення параметрів
    print("Знайдені параметри:")
    print(theta)

    y = y_train
    X = X_train_scaled

    # Додаємо стовбець з одиницями для зсуву в матрицю ознак X
    X_b = np.c_[np.ones((len(X), 1)), X]

    # Обчислюємо аналітичне рішення за допомогою normal equation
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print(theta)

    # Зробимо прогнози для нових даних
    X_new = X_test_scaled
    X_new_b = np.c_[np.ones((len(X_new), 1)), X_new]
    y_pred_a = X_new_b.dot(theta)

    mse_multi_analythic = mean_squared_error(y_test, y_pred_a)
    print(f'Mean Squared Error: {mse_multi_analythic}')

    plt.plot(X[:, 2], y, "b.", label="Actual")
    plt.plot(X_new[:, 2], y_pred_a, "r+", label="Prediction", alpha=0.3)
    plt.xlabel("sqft_living")
    plt.ylabel("price")
    plt.legend()
    plt.show()

    plt.plot(X_new[:100, 2], y_pred_a[:100], "r+", label="analytic", alpha=0.6)
    plt.plot(X_test_scaled[:100, 2], y_pred[:100], "bo", label="linear regression", alpha=0.1)
    plt.savefig('kaggle/working/linear_regression_vs_analytic_plot.png', )
    plt.show()
    print(mean_squared_error(y_pred[:100], y_pred_a[:100]), )


    def loss_j(y_test, y_pred):
        squared_errors = (y_test - y_pred) ** 2
        mse = np.mean(squared_errors)
        return mse


    y = np.random.rand(100)
    h_x = np.random.rand(100)
    mse_result = loss_j(y, h_x)

    print("Mean Squared Error:", mse_result)
    print("Mean Squared Error Analytic:", loss_j(y_test, y_pred_a))
    print("Mean Squared Error Multi:", loss_j(y_test, y_pred))
