import re


def flatten_list(IN_nested_list):
    """
    Flatten a list of lists recursively.
    """
    flattened_list = []
    for item in IN_nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def get_clean_list_of_words(IN_list_of_items: list):
    """
    Convert a list of items (containing strings as sentences, single words, numbers, ...) into a flattened list while
    cleaning it from upper-case, punctuation, etc.
    
    A simplified version using list comprehension and less cleansing could be
    all_tokens = [[w.lower() for w in c.split()] for c in all_captions]
    all_tokens = [w for sublist in all_tokens for w in sublist]
    
    :param IN_list_of_items: A list of items, which could be single words, sentences, numbers, etc.
    :type IN_list_of_items: list
    :return: Flattened list of strings
    :rtype: list
    """
    
    # split single items of list
    cleaning_list = []
    for item in IN_list_of_items:
        if len(item.split()) > 1:
            cleaning_list.append(item.split())
        else:
            cleaning_list.append(item)
    
    # generate one list of words out of nested lists
    flattened_list = flatten_list(IN_nested_list=cleaning_list)
    
    # convert all words in lower case
    lower_case_list = []
    for word in flattened_list:
        lower_case_list.append(word.lower())
    
    # clean all words from punctuation, trailing or leading spaces, etc.
    for word in range(len(lower_case_list)):
        re_pattern = re.compile(r"""
         \b[aA's]\b        # single a or A enclosed by blank
         |               # or
         [.,;:!]        # any punctuation
         |               # or
         's              # 's attached to word
         |
         /gi             # all single a, A (enclosed by blanks) and punctuations & 's
         with global search ignoring case (lower, upper case)
         """, re.VERBOSE)
        lower_case_list[word] = re_pattern.sub("", lower_case_list[word])
        lower_case_list[word] = re.sub("\s{2,}", " ", lower_case_list[word]).strip()  # delete trailing and leading
        # spaces
        # as well as multiple spaces
    
    # eliminate double words
    out_clean_list = list(set(lower_case_list))
    
    return out_clean_list


"""
# Testing for function get_clean_list_of_words
list1 = ["sdsd", "kjdwd"]
list2 = ["sdsd sEsd LdsD", "kjdwL ", "1913DDD, 13lmd ddll2 e2le.", "LdsD.", "LdsD LDsD."]
clean_list = get_clean_list_of_words(list2)
print(clean_list)
"""