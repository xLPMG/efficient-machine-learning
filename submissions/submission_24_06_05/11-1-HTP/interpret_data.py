import numpy as np
import csv

classProbsFile = "class_probs.raw"
pathResult = "output/htp_int8/output/Result_"
pathLabel = "/opt/data/imagenet/raw_test/batch_size_32/labels_"

printValues = False

def read_raw(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        class_probs = np.frombuffer(raw_data, dtype=np.float32)
        return class_probs
    
def read_csv(file_path):
    results = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            results.append(row)
    return results

for i in range(10):
    # Calculates probabilities for each class
    calculated_probs = read_raw(pathResult+str(i)+"/"+classProbsFile)
    # calculated_probs has 32k entries, 1k classes for 32 samples
    # -> 32 rows and 1000 columns
    reshaped_probabilities = calculated_probs.reshape(32, 1000)
    max_probabilities = np.argmax(reshaped_probabilities, axis=1)
    
    # read the correct labels
    given_labels = read_csv(pathLabel+str(i)+".csv")
    print("==============================")
    print("Result ",i)
    if printValues:
        print("Index, Estimate, Correct label, Match?")
    
    num_correct = 0
    for i in range(min(len(max_probabilities), len(given_labels))):
        prob = max_probabilities[i]
        label = int(given_labels[i][0])
        result = "-"
        if prob == label:
            result = "MATCH"
            num_correct+=1
        if printValues:
            print("%2d: %4d %4d %s" % (i, prob, label, result))
    print("Accuracy: ",str((num_correct / 32) * 100),"%")

    
    
