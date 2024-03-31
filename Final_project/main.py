from torchvision.models import vit_b_32, ViT_B_32_Weights
from fastai.vision.all import *
from fastai.imports import *
from fastai.torch_core import *
from fastai.learner import *

@patch
@delegates(subplots)
def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
    metrics = np.stack(self.values)
    names = self.metric_names[1:-1]
    n = len(names) - 1
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None: nrows = int(np.ceil(n / ncols))
    elif ncols is None: ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
        ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
        ax.set_title(name if i > 1 else 'losses')
        ax.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    # Define paths
    path = Path('/Users/albertlin/Projects/ML/Project/fastAI/data')
    dls = ImageDataLoaders.from_folder(path/'train', valid_pct=0.2, seed=42, item_tfms=Resize(224), batch_tfms=[*aug_transforms(size=224, max_rotate=20), Normalize.from_stats(*imagenet_stats)])

    learn = vision_learner(dls, resnet50, metrics=accuracy,wd=0.3)
    learn.fine_tune(epochs=10)
    # learn = vision_learner(dls, resnet50 , metrics=accuracy, pretrained=True, weights= ResNet50_Weights.IMAGENET1K_V2)
    
    learn.recorder.plot_metrics()

    plt.plot(learn.recorder.lrs, learn.recorder.losses)
    plt.title("Learning rate vs Losses of ResNet50")
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()

    interp = ClassificationInterpretation.from_learner(learn)
    full_confusion = interp.confusion_matrix()
    most_confused_items = interp.most_confused()
    most_confused_items = most_confused_items[:10]

    for item in most_confused_items:
        actual, predicted, occurrences = item
        actual_index, predicted_index = interp.vocab.o2i[actual], interp.vocab.o2i[predicted]
        confusion_value = full_confusion[actual_index, predicted_index]
        print(f"Actual: {actual}, Predicted: {predicted}, Occurrences: {occurrences}, Confusion Value: {confusion_value}")

    learn.export('./model_vgg.pkl')
