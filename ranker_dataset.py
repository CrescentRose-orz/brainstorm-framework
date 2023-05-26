from argparse import ArgumentParser

from datasets import load_dataset, disable_progress_bar
disable_progress_bar()




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--problem-path", type=str, required=True,
                        help="Path of the problems.")
    parser.add_argument("--source-paths", type=list, nargs="+", required=True,
                        help="Pathes of data sources.")
    args = parser.parse_args()
    
    
