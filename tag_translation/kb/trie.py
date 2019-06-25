import wordninja
from operator import truediv


class TrieNode(object):
    """
    Trie Node
    """

    def __init__(self, char):
        # The character
        self._char = char

        # The children nodes,keys are chars, values are nodes
        self._children = {}

        # Marks if the node is the last character of the word
        self._is_word = False

        # Keeps track of the actual word
        self._word = None

    @property
    def isWord(self):
        return self._is_word

    @isWord.setter
    def isWord(self, value):
        self._is_word = value


class Trie(object):
    """
        Trie implementation, adjusted for genre representation
    """

    def __init__(self):
        self._root = TrieNode("*")


    def tokenize(self, string):
        """
            Given a string tokenize it into words that are present in the Trie. If not successful, returns an empty string ''
        """
        def _tokenize(index, node=None, acc=[]):
            if node is None:
                node = self._root

            if index >= len(string):
                if node.isWord:
                    return acc
                else:
                    return []
            else:
                char = string[index]
                if char in node._children:
                    if node.isWord:

                        # It will be greedy trying to match the longest word possible
                        acc1 = _tokenize(index + 1, node=node._children[char], acc=[char])

                        if acc1:
                            ## if that recursive branch succeeds then return the result
                            acc.extend(acc1)
                            return acc
                        else:
                            ## because the above recursion failed be less greedy and take
                            ## the next best match path
                            acc.append(" ")
                            return _tokenize(index, node=None, acc=acc)
                    else:
                        acc.append(char)
                        return _tokenize(index + 1, node=node._children[char], acc=acc)
                else:
                    if node.isWord:
                        acc.append(" ")
                        return _tokenize(index, node=None, acc=acc)
                    else:
                        return []

        # Result is a list of chars, with space to delimit each word
        acc =  _tokenize(index=0)
        if len(acc) == 0:
            tokens = []
        else:
            tokens = ''.join(acc).split(' ')
        return tokens


    def _has_prefix(self, prefix, check_has_word):
        """
            Check if a prefix exists in the trie
            if check_has_word is True, then it checks whether the word to exist
        """
        if prefix is None:
            return False

        chars = list(prefix)
        node = self._root
        for char in chars:
            if char not in node._children:
                return False
            node = node._children[char]

        if check_has_word:
            return node.isWord
        else:
            return True

    def has_prefix(self, prefix):
        """
            Check if a prefix exists in the trie
        """
        return self._has_prefix(prefix, False)

    def has_word(self, prefix):
        """
            Check if a word exists in the trie
        """
        return self._has_prefix(prefix, True)


    def _add_word(self, word):
        """
            Add a word in the trie
        """
        chars = list(word)
        node = self._root

        for char in chars:
            if char not in node._children:
                node._children[char] = TrieNode(char)
            node = node._children[char]

        node.isWord = True
        node._word = word

    def get_words(self, prefix=None):
        """
            Return all the words in the trie, having the given prefix
            If prefix is None, it returns all the words
        """
        def _get_word_list(node, prefix, result):
            if node.isWord:
                result.append(prefix)
            for char in node._children.keys():
                _get_word_list(node._children[char], prefix + char, result)

        if prefix is None:
            return None

        result = []
        node =  self._root
        chars = list(prefix)

        for char in chars:
            if char not in node._children:
                return result
            node = node._children[char]
        _get_word_list(node, prefix, result)
        return result

    def get_all_words(self):
        """
            Return all the words in the trie
        """
        return self.get_words("")

    def print_words(self, prefix=""):
        """
            Print all words that start with given prefix
            If prefix is "", all the words of the trie are printed
        """
        result = self.get_words(prefix)
        for concept in result:
            print(concept)


    def _mark_known_words(self, words):
        """
            Returns a list of tuple,
            each  composed of the word and a boolean flag marking if the word is in the trie or not
        """
        words_with_in_trie_flags = []
        for word in words:
            if self.has_word(word):
                words_with_in_trie_flags.append((word, True))
            else:
                # try to tokenize instead
                tokens = self.tokenize(word)
                if len(tokens) > 0 and self._acceptable_tokenization(word, tokens):
                    for token in tokens:
                        words_with_in_trie_flags.append((token, True))
                else:
                    words_with_in_trie_flags.append((word, False))
        return words_with_in_trie_flags

    def decode_tag(self, word):
        """
            Decode a genre in tokens following the same procedure as when the trie was built
        """
        tokens = self.tokenize(word)
        wiki_tokens = wordninja.split(word)

        if len(tokens) == 1 and len(wiki_tokens) == 1:
            return tokens

        marked = self._mark_known_words(wiki_tokens)
        wiki_tokens = [word for word, known in marked]

        # if all words are known prioritize the wiki
        for token, known in marked:
            if not known:
                return tokens

        if len(tokens) <= len(wiki_tokens):
            return tokens

        return wiki_tokens

    def _is_short_string(self, string, upper_limit = 3):
        """
            Check if a given string is short
            upper_limit specifies the upper bound length of a short string
        """
        return len(string) <= upper_limit

    def _is_wikipedia_word(self, string):
        """
            Check if a given word exists in wikipedia
        """

        # Use wordninja to split the word
        # It is a probalistic approach to split based on Wikipedia;
        # It works mostly for English words
        words = wordninja.split(string)

        # If there is only one returned word, the string itself
        # then it means that the string exists in wikipedia as a word
        return len(words) == 1

    def _tokens_in_order_form_word(self, tokens, word):
        """
            Check if the given list of tokens form the word through concatenation
        """
        return word == ''.join(tokens)

    def _are_many_short_tokens(self, tokens, word, average_length = 3.):
        """
            Check if word is split in too many short tokens
            average length is set to 3 chars
        """
        # Input checks
        if not self._tokens_in_order_form_word(tokens, word):
            raise ValueError("The tokens do not appear to be extracted from the given word", tokens, word)

        # Word is split in too many short tokens if there are many tokens with length < average_length
        return len(tokens) > round(len(word)/average_length + .2)

    def _count_short_tokens_exceeds_threshold(self, tokens, short_max_length = 2, threshold = 2):
        """
            Check if the number of short tokens exceeds a threshold set by default to 2
            A token is short if its lenght is <= short_max_length
        """
        return sum([len(token) <= short_max_length for token in tokens]) >= threshold


    def _has_short_suffix(self, tokens, len_chars = 2):
        """
            Check if the last token, the suffix, is 1-char long
        """
        return len(tokens) > 1 and len(tokens[-1]) <= len_chars


    def _known_tokens_single__middle_letters(self, marked_wiki_tokens):
        """
            Check if the known tokens are single letters
        """

        # Ignore if the first letter is a token because often genres have prefixes of 1-letter long
        for token, known in marked_wiki_tokens[1:]:
            if known and len(token) == 1:
                return True
        return False

    def _acceptable_tokenization(self, word, tokens):
        """
            A tokenization is acceptable if multiple rules are met
                not too many short tokens
                the number of short tokens does not exceed a threshold
                the last token is not 1-char long
        """
        return not (self._are_many_short_tokens(tokens, word)
                or self._count_short_tokens_exceeds_threshold(tokens)
                or self._has_short_suffix(tokens))

    def _get_valid_tokenization_for_start_with_known(self, tokens):
        """
            Returns a list of valid tokens for the case when the first token is known

            The idea is that unknown tokens after known tokens are not considered as relevant
            and are appended instead to the previous known token
        """
        valid_tokens = []

        for token, known in tokens:
            if known:
                # If it is known just add it to the list of valid tokens
                valid_tokens.append(token)
            else:
                # If it is unknown append it to the last known token as a suffix
                valid_tokens[-1] += token

        return valid_tokens


    def _get_valid_tokenization_for_start_with_unknown(self, tokens):
        """
            Returns a list of valid tokens for the case when the first token is unknown

            The idea is that unknown tokens before known tokens are considered as relevant
            and are stored as standalone concepts
        """
        valid_tokens = []

        # If the last tokens are unknow append them to the last known token as suffix
        last_unknown = ''
        for token, known in tokens:
            if known:
                if last_unknown != '':
                    valid_tokens.append(last_unknown)
                    last_unknown = ''

                valid_tokens.append(token)
            else:
                last_unknown += token

        # Check for situations like Unknown+
        if len(valid_tokens) == 0:
            valid_tokens.append(last_unknown)
        # Check for situation like (Uknown+ Known+)+ Unknown+
        elif len(tokens) >= 2 and not tokens[-1][1]:
            valid_tokens[-1] += last_unknown

        return valid_tokens

    def _get_valid_tokenization(self, tokens):
        """
            This function produces a valid tokenization to be added to a trie from
            a list of tokens depending on their belonging to a vocabulary.

            Tokens is a list of tuples containing a token and a
            boolean indicating whether the token is part of the vocabulary.
        """
        # If the first token is known
        if tokens[0][1]:
            # Apply the first strategy
            return self._get_valid_tokenization_for_start_with_known(tokens)

        # If the first token was unknonw the expression is parsed differently
        return self._get_valid_tokenization_for_start_with_unknown(tokens)

    def _add_with_wiki_tokenization(self, string):
        """
            Helper method for adding a word tokenized with Wordninja
            It returns the concepts that were added
        """
        wiki_tokens = wordninja.split(string)
        marked_wiki_tokens = self._mark_known_words(wiki_tokens)
        wiki_tokens = [word for word, known in marked_wiki_tokens]
        unknown_tokens = [token for token, known in marked_wiki_tokens if not known]

        if not self._acceptable_tokenization(string, wiki_tokens):
            # If the tokenization was not acceptable add the complete string
            self._add_word(string)
            #print("WN_SPLIT_BREAK_HEURISTICS ", string, wiki_tokens)

            # And return it as the single token
            return [string]


        if self._tokens_in_order_form_word(unknown_tokens, string):
            # If all tokens form the word, then add it to the trie
            # It was noticed in experiments that it behaved better this way
            self._add_word(string)
            #print ("WN_SPLIT_UNKNOWN_WORD ", word, wiki_tokens)

            # And return the string as the single token
            return [string]

        if self._known_tokens_single__middle_letters(marked_wiki_tokens):
            self._add_word(string)
            return [string]

        # The tokenization given by wordninja is checked against the trie
        # and some further preprocessing is done to get a valid tokenization
        final_concepts = self._get_valid_tokenization(marked_wiki_tokens)

        for concept in final_concepts:
            if not self.has_word(concept):
                self._add_word(concept)

        return final_concepts

    def _try_add_with_trie_tokenization(self, string):
        """
            It adds a string to trie by checking first if it can be tokenized and if the tokenization is acceptable
            Returns None if it was not a successful tokenization or the list of tokens if successful
        """
        # First tokenize string by using the trie
        trie_tokens = self.tokenize(string)

        # If a tokenization was not possible with the trie
        if len(trie_tokens) == 0:
            return None

        # First check if there are short tokens in the result and
        if self._count_short_tokens_exceeds_threshold(trie_tokens, short_max_length = 2, threshold = 1):
             # the word would exists in wikipedia as a standalone word
            if self._is_wikipedia_word(string):
                # If yes, add the word
                self._add_word(string)
                # And return it as the single token
                return [string]

        # If the tokenization is acceptable
        if self._acceptable_tokenization(string, trie_tokens):
            # Return the obtained tokens
            return trie_tokens

        # the word is really long, give a shot with wikininja
        if len(string) > 15:
            return None

        # If the tokenization was not acceptable add the complete word
        self._add_word(string)
        # And return it as the single token
        return [string]



    def add_string_with_tokenization(self, string):
        """
            Add a word by first trying its tokenization on the current trie content. This is adjusted to deal with genres written as multiple concatenated words without any space

            if the word is tokenized in multiple concepts then a list of concepts is returned, else the word is returned (this is used for adding the edges to the graph)
        """

        # Safety checks on the input
        if string is None or len(string) == 0:
            return []

        # If string is short
        if self._is_short_string(string, upper_limit = 3):
            # Then add string to trie
            self._add_word(string)
            #print("WORD FOR SEEDING ", string)

            # Return list of tokens containing only the string
            return [string]

        # It tries to tokenize a word using the trie and to add it to the trie if the tokenization is not acceptable
        concepts = self._try_add_with_trie_tokenization(string)
        # If the tokenization and adding were successful
        if concepts is not None:
            # Return concepts
            return concepts

        # If the previous strategy to add the string by first trying trie tokenizations failed
        # Then split it instead with wordninja
        # https://github.com/keredson/wordninja
        # It is a probalistic approach to split based on Wikipedia words (mainly English)
        return self._add_with_wiki_tokenization(string)

    # Levenshtein distance from http://stevehanov.ca/blog/index.php?id=114
    def search(self, word, maxCost):
        """
            The search function returns a list of all words that are less than the given
            maximum distance from the target word
        """

        # Build first row
        currentRow = range(len(word) + 1)

        results = []

        # Recursively search each branch of the trie
        root = self._root
        for letter in root._children:
            self.search_recursive(root._children[letter], letter, word, currentRow, results, maxCost)

        return results


    def search_recursive(self, node, letter, word, previousRow, results, maxCost):
        """
            This recursive helper is used by the search function above. It assumes that
            the previousRow has been filled in already.
        """
        columns = len(word) + 1
        currentRow = [previousRow[0] + 1]

        # Build one row for the letter, with a column for each letter in the target
        # word, plus one for the empty string at column 0
        for column in range(1, columns):
            insertCost = currentRow[column - 1] + 1
            deleteCost = previousRow[column] + 1

            if word[column - 1] != letter:
                replaceCost = previousRow[column - 1] + 1
            else:
                replaceCost = previousRow[column - 1]

            currentRow.append(min(insertCost, deleteCost, replaceCost))

        # If the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a word in this trie node, then add it.
        if currentRow[-1] <= maxCost and node._word != None:
            results.append((node._word, currentRow[-1]))

        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if min(currentRow) <= maxCost:
            for letter in node._children:
                self.search_recursive(node._children[letter], letter, word, currentRow, results, maxCost)


