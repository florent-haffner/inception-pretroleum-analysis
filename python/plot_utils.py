import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks", palette='muted')


def plot_val(test_labels, test_predictions, rmse_limit=2.8, axes=[7, 16]):
    plt.xlabel('Observed values')
    plt.ylabel('Predicted values')
    plt.xlim(axes)
    plt.ylim(axes)

    mask_color = np.logical_or((test_predictions >= (test_labels + rmse_limit)),
                               (test_predictions <= (test_labels - rmse_limit)))
    per_in = 1 - np.sum(mask_color) / mask_color.size
    color = np.where(mask_color, 'out', f'in ({per_in * 100:.2f}%)')
    sns.scatterplot(x=test_labels.flatten(), y=test_predictions.flatten(),
                    hue=color.flatten(), palette=['#212c3d', '#c05e31'])
    mini = min(test_labels.flatten().min(), test_predictions.flatten().min())
    maxi = max(test_labels.flatten().max(), test_predictions.flatten().max())

    plt.plot([mini, maxi], [mini, maxi], linestyle='--', color='gray')
    plt.plot([mini, maxi], [mini - rmse_limit, maxi - rmse_limit], linestyle='--', color='black')
    plt.plot([mini, maxi], [mini + rmse_limit, maxi + rmse_limit], linestyle='--', color='black')

    # Custom legend labels
    legend_labels = {'out': f'Beyond limit ({np.sum(mask_color)})',
                     'in': f'Inside interval ({per_in * 100:.0f}%)'}
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#c05e31',
                                  markersize=10, label=legend_labels['out']),
                       plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#212c3d',
                                  markersize=10, label=legend_labels['in'])]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.show()

def plot_err(test_labels, test_predictions, bins=35, deviation=1, n_count=50):
    plt.figure()
    error = test_predictions - test_labels
    plt.hist(error, bins)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.xlim([-deviation, deviation])
    plt.ylim([0, n_count])
    plt.show()


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
