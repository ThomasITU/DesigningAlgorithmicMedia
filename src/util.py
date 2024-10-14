class UtilityFunctions:
    
    @staticmethod
    # drop columns if not already dropped
    def drop_columns(df, columns_to_drop):
        # Drop columns only if they exist in the DataFrame
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

        # Drop the existing columns
        df.drop(columns=existing_columns_to_drop, inplace=True)
        return df
    
    @staticmethod
    # print non-numeric columns
    def print_non_numeric_columns(df):
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()

        for coln in non_numeric_columns:
            print(f"{coln}: {df[coln].unique()}")

    @staticmethod
    def save_dataframe(df, filename):
        processed_path = 'data/processed/'
        path = processed_path+filename + '.csv'
        df.to_csv(path, index=False)

        return path
    
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
