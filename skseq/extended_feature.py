from skseq.id_feature import IDFeatures


# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------
class ExtendedFeatures(IDFeatures):
    """
    A class that extends the functionality of IDFeatures by adding additional features.
    """

    def add_emission_features(self, sequence, pos, y, features):
        """
        Adds emission features to the existing features based on the given sequence, position, label, and list of features.

        Args:
            sequence (Sequence): The input sequence.
            pos (int): The position in the sequence.
            y (int): The label ID.
            features (list): The list of existing features.

        Returns:
            list: The updated list of features.
        """

        x = sequence.x[pos]

        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)

        # Get word name from ID.
        if isinstance(x, str):
            x_name = x
        else:
            x_name = self.dataset.x_dict.get_label_name(x)

        word = str(x_name)

        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        # OUR FEATURES

        # Check if word starts with uppercase
        if str.istitle(word):
            feat_name = "firstupper::%s" % (y_name)
            feat_name = str(feat_name)

            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # Check if is digit
        if str.isdigit(word):
            feat_name = "digit::%s" % y_name
            feat_name = str(feat_name)

            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # Check if is upper
        if str.isupper(word):
            feat_name = "upper::%s" % y_name
            feat_name = str(feat_name)

            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # Check if contains numbers
        if any(char.isdigit() for char in word) and not str.isdigit(word):
            # Generate feature name.
            feat_name = "insidedigit::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        # Check if contains non-ascii/punctuation/hyphens
        if not all(
                char.isalnum() or char in ['-', '.', ',', ':', ';', '!', '?', '"', '(', ')', '[', ']', '{', '}', '\'']
                for char in word):
            # Generate feature name.
            feat_name = "nonascii::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        # Check if is lowercase
        if str.islower(word):
            feat_name = "lower::%s" % y_name
            feat_name = str(feat_name)

            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # Check if contains only alphanumeric characters
        if str.isalnum(word):
            feat_name = "alphanum::%s" % y_name
            feat_name = str(feat_name)

            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # Check if word contains specific suffixes
        if str.endswith(word, 'day'):
            feat_name = "suffixday::%s" % y_name
            feat_name = str(feat_name)

            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        if str.endswith(word, 'ber'):
            feat_name = "suffixber::%s" % y_name
            feat_name = str(feat_name)

            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # Check if is a certain word (e.g., the, to, an, etc.)
        # (Different functions can be added here for specific words)

        # Check if is a stopword
        # (Function for checking stopwords can be added here)

        return features
