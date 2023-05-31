from tqdm import tqdm
#from skseq.sequences.sequence_list import SequenceList
from skseq.label_dictionary import LabelDictionary
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from skseq import sequence_list_c


def get_data_target_sets(data):
    """
    Extracts sentences and tags from the provided data object based on sentence IDs.

    Args:
        data: A data object containing sentences and corresponding tags.

    Returns:
        A tuple (X, y) containing lists of sentences (X) and tags (y).
    """
    X = []  # Contains the sentences
    y = []  # Contains the tags

    ids = data.sentence_id.unique()  # Get unique sentence IDs from the data

    # Use tqdm to create a progress bar
    progress_bar = tqdm(ids, desc="Processing", unit="sentence")

    for sentence in progress_bar:  # Iterate over each unique sentence ID
        # Append the words for the current sentence to X
        X.append(list(data[data["sentence_id"] == sentence]["words"].values))
        # Append the tags for the current sentence to y
        y.append(list(data[data["sentence_id"] == sentence]["tags"].values))

    return X, y  # Return the lists of sentences and tags


def create_corpus(sentences, tags):
    """
    Creates a corpus by generating dictionaries for words and tags in the given sentences and tags.

    Args:
        sentences: A list of sentences.
        tags: A list of corresponding tags for the sentences.

    Returns:
        A tuple (word_dict, tag_dict, tag_dict_rev) containing dictionaries for words, tags,
        and a reversed tag dictionary.

    Example:
        sentences = [['I', 'love', 'Python'], ['Python', 'is', 'great']]
        tags = ['O', 'O', 'B']
        word_dict, tag_dict, tag_dict_rev = create_corpus(sentences, tags)
        # word_dict: {'I': 0, 'love': 1, 'Python': 2, 'is': 3, 'great': 4}
        # tag_dict: {'O': 0, 'B': 1}
        # tag_dict_rev: {0: 'O', 1: 'B'}
    """
    word_dict = {}  # Contains unique words with corresponding indices
    tag_dict = {}  # Contains unique tags with corresponding indices

    # Generate word dictionary
    for sentence in sentences:
        for word in sentence:
            if word not in word_dict:
                word_dict[word] = len(word_dict)

    # Generate tag dictionary
    for tag_list in tags:
        for tag in tag_list:
            if tag not in tag_dict:
                tag_dict[tag] = len(tag_dict)

    tag_dict_rev = {v: k for k, v in tag_dict.items()}  # Reverse tag dictionary

    return word_dict, tag_dict, tag_dict_rev


def create_sequence_list(word_dict, tag_dict, X, y):
    """
    Creates a sequence list object by adding sequences from X and y to it.

    Args:
        word_dict: A dictionary mapping words to their corresponding indices.
        tag_dict: A dictionary mapping tags to their corresponding indices.
        X: A list of input sequences (sentences).
        y: A list of corresponding target sequences (tags).

    Returns:
        A sequence list object populated with sequences from X and y.
    """
    seq = sequence_list_c.SequenceList(LabelDictionary(word_dict), LabelDictionary(tag_dict))

    # Use tqdm to create a progress bar
    progress_bar = tqdm(range(len(X)), desc="Adding sequences", unit="sequence")

    for i in progress_bar:
        # Add the sequence (X[i], y[i]) to the sequence list
        seq.add_sequence(X[i], y[i], LabelDictionary(word_dict), LabelDictionary(tag_dict))

    return seq


def show_features(feature_mapper, seq, feature_type=["Initial features", "Transition features", "Final features", "Emission features"]):
    """
    Displays the features extracted from a sequence using a feature mapper.

    Args:
        feature_mapper: An object responsible for mapping feature IDs to feature names.
        seq: A sequence object containing the input sequence.
        feature_type: Optional. A list of feature types to display. Default is ["Initial features", "Transition features", "Final features", "Emission features"].

    Returns:
        None
    """
    inv_feature_dict = {word: pos for pos, word in feature_mapper.feature_dict.items()}

    for feat, feat_ids in enumerate(feature_mapper.get_sequence_features(seq)):
        print(feature_type[feat])  # Print the current feature type

        for id_list in feat_ids:
            for k, id_val in enumerate(id_list):
                print(id_list, inv_feature_dict[id_val])  # Print the feature IDs and their corresponding names

        print("\n")  # Add a newline after printing all features of a certain type


def get_tiny_test():
    """
    Creates a tiny test dataset.

    Args:
        None

    Returns:
        A tuple (X, y) containing lists of sentences (X) and tags (y).
    """
    X = [['The programmers from Barcelona might write a sentence without a spell checker . '],
         ['The programmers from Barchelona cannot write a sentence without a spell checker . '],
         ['Jack London went to Parris . '],
         ['Jack London went to Paris . '],
         ['Bill gates and Steve jobs never though Microsoft would become such a big company . '],
         ['Bill Gates and Steve Jobs never though Microsof would become such a big company . '],
         ['The president of U.S.A though they could win the war . '],
         ['The president of the United States of America though they could win the war . '],
         ['The king of Saudi Arabia wanted total control . '],
         ['Robin does not want to go to Saudi Arabia . '],
         ['Apple is a great company . '],
         ['I really love apples and oranges . '],
         ['Alice and Henry went to the Microsoft store to buy a new computer during their trip to New York . ']]

    y = [['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'I-per', 'O', 'O', 'B-geo', 'O'],
            ['B-per', 'I-per', 'O', 'O', 'B-geo', 'O'],
            ['B-per', 'I-per', 'O', 'B-per', 'I-per', 'O', 'O', 'I-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'I-per', 'O', 'B-per', 'I-per', 'O', 'O', 'I-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'I-geo', 'I-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'I-geo', 'O', 'O', 'O', 'O'],
            ['B-per', 'O', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'O'],
            ['B-org', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'O', 'B-per', 'O', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-geo',
             'I-geo', 'O']]

    return [i[0].split() for i in X], y


def predict_SP(model, X):
    """
    Predicts the tags for the input sequences using a StructuredPerceptron model.

    Args:
        model: A trained StructuredPerceptron model.
        X: A list of input sequences (sentences).

    Returns:
        A list of predicted tags for the input sequences.
    """
    y_pred = []

    # Use tqdm to create a progress bar
    progress_bar = tqdm(range(len(X)), desc="Predicting tags", unit="sequence")

    for i in progress_bar:
        # Predict the tags for the current input sequence
        predicted_tag = model.predict_tags_given_words(X[i])
        y_pred.append(predicted_tag)

    y_pred = [np.ndarray.tolist(array) for array in y_pred]
    y_pred = np.concatenate(y_pred).ravel().tolist()

    return y_pred


def accuracy(true, pred):
    """
    Computes the accuracy of predicted tags compared to true tags, excluding instances where true[i] == 'O'.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.

    Returns:
        The accuracy score, which measures the proportion of correct predictions.
    """
    # Get indexes of those that are not 'O'
    idx = [i for i, x in enumerate(true) if x != 'O']

    # Get the true and predicted tags for those indexes
    true = [true[i] for i in idx]
    pred = [pred[i] for i in idx]

    # Use sklearn's accuracy_score to compute the accuracy
    return accuracy_score(true, pred)


def plot_confusion_matrix(true, pred, tag_dict_rev):
    """
    Plots a confusion matrix using a heatmap.

    Args:
        true: A list or array of true labels.
        pred: A list or array of predicted labels.
        tag_dict_rev: A dictionary mapping tag values to their corresponding labels.

    Returns:
        None
    """
    # Get all unique tag values from true and pred lists
    unique_tags = np.unique(np.concatenate((true, pred)))

    # Create a tick label list with all unique tags
    tick_labels = [tag_dict_rev.get(tag, tag) for tag in unique_tags]

    # Get the confusion matrix
    cm = confusion_matrix(true, pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def f1_score_weighted(true, pred):
    """
    Computes the weighted F1 score based on the true and predicted tags.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.

    Returns:
        The weighted F1 score.
    """
    # Get the weighted F1 score using sklearn's f1_score function
    return f1_score(true, pred, average='weighted')


def evaluate(true, pred, tag_dict_rev):
    """
    Computes and prints evaluation metrics and displays a confusion matrix based on the true and predicted tags.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.
        tag_dict_rev: A dictionary mapping tag indexes to tag labels.

    Returns:
        None
    """
    # Compute the accuracy and F1 score using predefined functions
    acc = accuracy(true, pred)
    f1 = f1_score_weighted(true, pred)

    # Print the evaluation results
    print('Accuracy: {:.4f}'.format(acc))
    print('F1 Score: {:.4f}'.format(f1))

    # Plot the confusion matrix
    plot_confusion_matrix(true, pred, tag_dict_rev)


def print_tiny_test_prediction(X, model, tag_dict_rev):
    """
    Prints the predicted tags for each input sequence.

    Args:
        X: A list of input sequences.
        model: The trained model used for prediction.
        tag_dict_rev: A dictionary mapping tag indexes to tag labels.

    Returns:
        None
    """
    y_pred = []
    for i in range(len(X)):
        # Predict the tags for the current input sequence
        predicted_tag = model.predict_tags_given_words(X[i])
        y_pred.append(predicted_tag)

    for i in range(len(X)):
        sentence = X[i]
        tag_list = y_pred[i]
        prediction = ''
        for j in range(len(sentence)):
            prediction += sentence[j] + "/" + tag_dict_rev[tag_list[j]] + " "

        print(prediction + "\n")
