import pandas as pd
import glob
import os

def merge_csv_files(input_folder='../../data/final', output_file='../../data/final_dataset.csv'):

    file_pattern = os.path.join(input_folder, "*.csv")
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print("Can't find any files from path")
        return

    print(f"Files found for merge {len(all_files)}")

    df_list = []

   
    for file in all_files:
        try:

            df = pd.read_csv(file)
            

            file_name = os.path.basename(file)
            
            df_list.append(df)
            print(f"{file_name} lines loaded {len(df)})")
            
        except Exception as e:
            print(f"probleml with reading{file}: {e}")

    combined_df = pd.concat(df_list, ignore_index=True)

    print(f"\nSummary for merging {len(combined_df)}")

    combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"{output_file}' created succesfully")
    
    return combined_df

raw_df = merge_csv_files()