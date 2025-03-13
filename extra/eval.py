from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f'MAE: {mae:.2f} ppb')
    print(f'RMSE: {rmse:.2f} ppb')
    print(f'R²: {r2:.2f}')
    return y_pred

def plot_results(history, y_test, y_pred):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual DON Concentration (ppb)')
    plt.ylabel('Predicted DON Concentration (ppb)')
    plt.title('Actual vs. Predicted DON Concentration')
    plt.show()