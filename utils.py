import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    data = pd.read_csv(path)

    # Determine text column ('review' or 'Overview')
    if "review" in data.columns:
        text_col = "review"
    elif "Overview" in data.columns:
        text_col = "Overview"
    else:
        object_cols = data.select_dtypes(include=["object"]).columns
        if len(object_cols) > 0:
            text_col = object_cols[0]
        else:
            raise KeyError("Could not find a text column (e.g. 'review' or 'Overview') in dataset.")

    # Determine sentiment target
    if "sentiment" in data.columns:
        if data["sentiment"].dtype == object:
            y = data["sentiment"].str.lower().map({"positive": 1, "negative": 0}).fillna(data["sentiment"])
        else:
            y = data["sentiment"]
    elif "IMDB_Rating" in data.columns:
        # Binary classification based on rating threshold
        y = (data["IMDB_Rating"] >= 8.0).astype(int)
    elif "rating" in data.columns:
        y = (data["rating"] >= data["rating"].median()).astype(int)
    else:
        raise KeyError("Could not find a sentiment or rating column in dataset.")

    X = data[text_col]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )