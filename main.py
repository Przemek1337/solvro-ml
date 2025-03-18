from src.preprocessing import LoadDataset

if __name__ == "__main__":
    file_path = "data/cocktail_dataset.json"

    data = LoadDataset(file_path)
    # print(data.head())