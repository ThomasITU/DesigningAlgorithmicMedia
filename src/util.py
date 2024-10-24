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
    # Find columns with fewer than `threshold` unique values
    def few_unique_values_columns(df, threshold):
        few_unique_columns = {}
        for column in df.columns:
            unique_count = df[column].nunique()

            if unique_count <= threshold:
                few_unique_columns[column] = {
                    'unique_count': unique_count,
                    'unique_values': df[column].unique()
                }
        return few_unique_columns
    
    @staticmethod
    def print_few_unique_values_columns(df, threshold):
        few_unique_columns = UtilityFunctions.few_unique_values_columns(df, threshold)
        for column in few_unique_columns:
            print(f"{column}: {few_unique_columns[column]}")

    @staticmethod
    def filter_features(df, features_to_keep):
        # Filter the DataFrame to only include specified features
        filtered_df = df[features_to_keep]
        missing_features = set(features_to_keep) - set(df.columns)
        if missing_features:
            print("Warning: The following features are missing from the DataFrame:", missing_features)
        return filtered_df
