import argparse
import subprocess
from src.pipeline.run_pipeline import run_pipeline

def main(mode):

    if mode == "train":
        run_pipeline(
            "data/sample_data/sample_data.csv",
            "data/test_data/test_data.csv",
            "outputs/predictions/test_predictions.csv"
        )

    elif mode == "app":
        subprocess.run(["streamlit", "run", "app/streamlit_app.py"])

    elif mode == "all":
        run_pipeline(
            "data/sample_data/sample_data.csv",
            "data/test_data/test_data.csv",
            "outputs/predictions/test_predictions.csv"
        )
        subprocess.run(["streamlit", "run", "app/streamlit_app.py"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    args = parser.parse_args()
    main(args.mode)
    
    
    

    
