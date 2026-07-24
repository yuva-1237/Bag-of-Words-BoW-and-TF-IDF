from utils import load_data
# pyrefly: ignore [missing-import]
from models.bow_model import run_bow
# pyrefly: ignore [missing-import]
from models.tfidf_model import run_tfidf

dataset_path = "dataset/IMDB Dataset.csv"

X_train, X_test, y_train, y_test = load_data(dataset_path)

bow_result = run_bow(
    X_train,
    X_test,
    y_train,
    y_test
)

tfidf_result = run_tfidf(
    X_train,
    X_test,
    y_train,
    y_test
)

print("\n==========================")
print("Bag of Words Results")
print("==========================")

for key, value in bow_result.items():
    print(f"{key}: {value:.4f}")

print("\n==========================")
print("TF-IDF Results")
print("==========================")

for key, value in tfidf_result.items():
    print(f"{key}: {value:.4f}")

print("\n==========================")
print("Comparison")
print("==========================")

print("{:<12}{:<12}{:<12}".format("Metric","BoW","TF-IDF"))

for metric in bow_result.keys():
    print("{:<12}{:<12.4f}{:<12.4f}".format(
        metric,
        bow_result[metric],
        tfidf_result[metric]
    ))