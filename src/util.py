import os

class UtilityFunctions:

    @staticmethod
    def get_non_numeric_columns(dataframe):
        non_numeric_columns = dataframe.select_dtypes(exclude=['number']).columns.tolist()
        return non_numeric_columns
    
    @staticmethod
    # print non-numeric columns
    def print_non_numeric_columns(dataframe):
        non_numeric_columns = UtilityFunctions.get_non_numeric_columns(dataframe)
        for coln in non_numeric_columns:
            print(f"{coln}: {dataframe[coln].unique()}")


    @staticmethod
    def save_dataframe(dataframe, filename):
        processed_path = './../data/processed/'
        path = processed_path+filename + '.csv'
        dataframe.to_csv(path, index=False)

        return path
    
    @staticmethod
    def get_csv_files_from_folder(folder_path:str = './../data/raw/'):
        csv_files = [folder_path+f for f in os.listdir(folder_path) if f.endswith('csv') and os.path.isfile(os.path.join(folder_path, f))]
        return csv_files

    @staticmethod
    def filter_features(dataframe, features_to_keep):
        # Filter the DataFrame to only include specified features
        filtered_df = dataframe[features_to_keep]
        missing_features = set(features_to_keep) - set(dataframe.columns)
        if missing_features:
            print("Warning: The following features are missing from the DataFrame:", missing_features)
        return filtered_df

    @staticmethod
    def filter_numerical_values(dataframe):
        columns_to_drop = UtilityFunctions.get_non_numeric_columns(dataframe)   
        df_numeric = UtilityFunctions.drop_columns(dataframe, columns_to_drop)
        df_numeric = dataframe.select_dtypes(include=[float, int])
        
        return df_numeric
    
    @staticmethod
    def filter_negative_values(dataframe):
        df = dataframe.map(lambda x: 0 if x < 0 else x)
        return df
    
    @staticmethod
    def filter_columns_with_less_unique_values_than_threshold(dataframe, threshold: int = 1):
        # Drop columns with fewer than `threshold` unique values
        columns_to_drop = list(UtilityFunctions.few_unique_values_columns(dataframe, threshold).keys())
        df = UtilityFunctions.drop_columns(dataframe, columns_to_drop)
        return df
    
    @staticmethod
    # drop columns if not already dropped
    def drop_columns(dataframe, columns_to_drop):
        # Drop columns only if they exist in the DataFrame
        existing_columns_to_drop = [col for col in columns_to_drop if col in dataframe.columns]

        # Drop the existing columns
        df = dataframe.drop(columns=existing_columns_to_drop)
        return df

    @staticmethod
    # Find columns with fewer than `threshold` unique values
    def few_unique_values_columns(dataframe, threshold):
        few_unique_columns = {}
        for column in dataframe.columns:
            unique_count = dataframe[column].nunique()

            if unique_count <= threshold:
                few_unique_columns[column] = {
                    'unique_count': unique_count,
                    'unique_values': dataframe[column].unique()
                }
        return few_unique_columns
    
    @staticmethod
    def print_few_unique_values_columns(dataframe, threshold):
        few_unique_columns = UtilityFunctions.few_unique_values_columns(dataframe, threshold)
        for column in few_unique_columns:
            print(f"{column}: {few_unique_columns[column]}")



