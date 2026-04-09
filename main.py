from src.pipeline.run_pipeline import run_pipeline


if __name__=="__main__":
    train_path = "data/sample_data/sample_data.csv"
    test_path = "data/test_data/test_data.csv"
    output_path = "outputs/predictions/test_predictions.csv"

    run_pipeline(train_path, test_path, output_path)