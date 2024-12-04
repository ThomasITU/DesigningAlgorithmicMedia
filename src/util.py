import os
import pandas as pd

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
        processed_path = './../data/processed/'  # Path for saving the file
        # Ensure the directory exists
        os.makedirs(processed_path, exist_ok=True)
        # Construct the full file path
        path = os.path.join(processed_path, filename + '.csv')
        # Save the dataframe to the file
        dataframe.to_csv(path, index=False)
        #print(f"Saved {filename} to {path}")
        return path
    
    @staticmethod
    def get_csv_files_from_folder(folder_path:str = './../data/raw/'):
        csv_files = [folder_path+f for f in os.listdir(folder_path) if f.endswith('csv') and os.path.isfile(os.path.join(folder_path, f))]
        return csv_files


    @staticmethod
    def read_data(csv_file, delimiters):
        for sep in delimiters:
            try:
                dataframe = pd.read_csv(csv_file, sep=sep, on_bad_lines='skip', low_memory=False)
                return dataframe
            except Exception as e:
                e
        return None

    @staticmethod
    def print_unique_values_for_columns(dataframe, features):
        # Iterate through each feature in the country_features list
        for feature in features:
            #print(f"Unique values for feature '{feature}':")

            # Check if the feature exists in the dataframe
            if feature in dataframe.columns:
                unique_values = dataframe[feature].unique()
                #print(unique_values)


    @staticmethod
    def find_country_feature_name(dataframe, countries, country_features):
        feature_name = None

        # Loop through each country feature to check for country codes
        for country_code_feature_name in country_features:
            #print(f"Checking feature: {country_code_feature_name}")

            # Check if the country code feature exists in the dataframe
            if country_code_feature_name not in dataframe.columns:
                #print(f"Feature '{country_code_feature_name}' does not exist in the dataframe.")
                continue

            # Loop through each country code to match the values in the column
            for country_code, name in countries:
                #print(f"  Checking if country code '{country_code}' exists in '{country_code_feature_name}' for {name}")

                # Try to find the country code in the feature column
                country_dataframe = dataframe[dataframe[country_code_feature_name] == country_code]

                if not country_dataframe.empty:
                    feature_name = country_code_feature_name
                    print(f"  Found country code '{country_code}' in feature '{country_code_feature_name}' for '{name}'")
                    break  # Found the matching feature, exit the loop early

            if feature_name:
                break  # Exit the outer loop once the feature name is found

        #if feature_name is None:
        #    print("No country code feature name found.")
        #else:
        #    print(f"Country code feature name found: {feature_name}")

        return feature_name

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
    def filter_rows_with_less_unique_values_than_threshold(dataframe, threshold: int = 1):
        # Filter out rows that have fewer than `threshold` unique values
        rows_to_keep = dataframe.apply(lambda row: row.nunique() >= threshold, axis=1)

        # Keep only the rows that meet the condition
        df = dataframe[rows_to_keep]

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



