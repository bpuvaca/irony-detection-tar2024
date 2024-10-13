import pandas as pd
import sys

def check_overlap(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    tweets1 = df1['tweet']
    tweets2 = df2['tweet']

    overlap = tweets1[tweets1.isin(tweets2)]

    if not overlap.empty:
        print("Overlapping tweets found:")
        print(overlap)
    else:
        print("No overlapping tweets found.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    check_overlap(file1, file2)
