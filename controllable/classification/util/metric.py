#to calculate accuracy

def get_accuracy(preds, labels):
  total_acc = 0.0
  
  for i in range(len(labels)):
    if labels[i] == preds[i]:
      total_acc+=1.0
  
  return total_acc / len(labels)