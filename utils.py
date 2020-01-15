import itertools
import matplotlib.pyplot as plt


names = ('Neutral', 'Anger', 'Contempt', 'Disgust', 'Fear',
         'Happiness', 'Sadness', 'Surprise')


#Nice Confusion Matrix 
def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):
  if normalize:
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    print('Normalized confusion matrix')
  else:
    print('Confusion matrix, without normalization')
  print(cm)

  plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation = 45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment = 'center',
             color = 'white' if cm[i, j] > thresh else 'black')
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  #source: https://deeplizard.com/learn/video/0LhiS6yu2qQ


#Performance measures
def precision(names, cm, show = False):
  precision= []
  for i in range(len(names)):
    tp_fp = cm[:, i].sum().float()
    for j in range(len(names)):
      if i == j:
        tp = cm[i, j].float()
    precision.append(round(float(tp/tp_fp), 3))
  if show:
    for i in range(len(names)):
      print('Precision for label %s: ' %names[i], round(float(precision[i]), 3))
  else:
    return precision


def recall(names, cm, show = False):
  recall = []
  for i in range(len(names)):
    tp_fn = cm[i, :].sum().float()
    for j in range(len(names)):
      if i == j:
        tp = cm[i, j].float()
    recall.append(round(float(tp/tp_fn), 3))
  if show:
    for i in range(len(names)):
      print('Recall for label %s: ' %names[i], round(float(recall[i]),3))
  else:
    return recall


def f1_score(names, cm, show = False):
  precision = []
  recall = []
  f1_score = []
  for i in range(len(names)):
    tp_fp = cm[:, i].sum().float()
    tp_fn = cm[i, :].sum().float()
    for j in range(len(names)):
      if i == j:
        tp = cm[i, j].float()
    precision.append(tp/tp_fp)
    recall.append(tp/tp_fn)
    f1_score.append(round(float(2 * (precision[i] * recall[i])/(precision[i] + recall[i])),3))
  if show:
    for i in range(len(names)):
      print('F1-score for label %s: ' %names[i], round(float(f1_score[i]),3))
  else:
    return f1_score


#Getting what's correct (NN predictions)
def get_num_correct(preds, labels):
    return preds.argmax(dim = 1).eq(labels).sum().item()
    #source: https://deeplizard.com/learn/video/XfYmia3q2Ow