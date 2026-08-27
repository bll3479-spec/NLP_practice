import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

def plot_comparison(histories, names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['tomato', 'steelblue', 'seagreen']

    for hist, name, color in zip(histories, names, colors):
        axes[0].plot(hist['train_loss'], label=name, color=color)
        axes[1].plot(hist['valid_acc'],  label=name, color=color)

    axes[0].set_title('Training Loss');    axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True)
    axes[1].set_title('Validation Accuracy (%)'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('spam_training_curve.png', dpi=100)
    plt.show()
    print('학습 곡선 저장: spam_training_curve.png')


def plot_confusion_matrices(results, names):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (labels, preds), name in zip(axes, results, names):
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
        ax.set_title(f'{name}'); ax.set_ylabel('실제'); ax.set_xlabel('예측')

    plt.tight_layout()
    plt.savefig('spam_confusion_matrix.png', dpi=100)
    plt.show()
    print('혼동 행렬 저장: spam_confusion_matrix.png')
