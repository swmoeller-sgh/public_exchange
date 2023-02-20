import pandas as pd


def extract_values_from_column (IN_pandas_dataframe: pd.core.frame.DataFrame,
                                IN_filter_column_name: str,
                                IN_filter_criteria,
                                OUT_column_name: str):
    """
    Selects a subset of values from a specific Pandas DataFrame column based on True and False indicator in the column.

    The shortest way to do it is using the code "subset_imageIds = data_df[data_df['train']].image_id.tolist()"
    The enclosed script is more versatile and allows to select the criteria to look for when selecting the single
    dataframes.
    :param IN_pandas_dataframe: Dataframe containing image name, caption, origin and id
    :type IN_pandas_dataframe: pd.core.frame.DataFrame
    :param IN_filter_column_name: The column name, which should be used to filter in
    :type IN_filter_column_name: str
    :param IN_filter_criteria: The filter criteria to be applied
    :type IN_filter_criteria: str or boolean
    :param OUT_column_name the column name, from which we want to read the values for the list to be generated
    :type OUT_column_name: str

    :return: a list of values from the "value column" after being filtered
    :rtype: list
    """
    data_df = IN_pandas_dataframe
    
    # Filter the dataframe based on the column name and filter criteria provided
    filtered_dataframe = data_df[data_df[IN_filter_column_name]==IN_filter_criteria]
    
    # generate a list of values from on specified column name
    list_of_values_from_one_df_column = filtered_dataframe[OUT_column_name].tolist()
    
    return list_of_values_from_one_df_column

"""
# Test case for "def extract_values_from_column"
data = {
    'Column1': [1, 2, 3, 4],
    'Column2': [10, 20, 30, 40],
    'Column3': ['A', 'B', 'C', 'D'],
    'Column4': [True, False, False, True]
}

# Create the DataFrame
df = pd.DataFrame(data)
list= extract_values_from_column(IN_pandas_dataframe=df,
                           IN_filter_column_name="Column4",
                           IN_filter_criteria="tiger",
                           OUT_column_name="Column3")
print(list)
"""

