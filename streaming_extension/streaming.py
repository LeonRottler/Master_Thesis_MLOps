import pandas as pd
import requests
import argparse


def send_csv_data(csv_file_path: str, url: str):
    df = pd.read_csv(csv_file_path).filter(regex="col_")

    for index, row in df.iterrows():
        data = row.to_dict()

        response = requests.post(url, data=data)

        if response.status_code != 200:
            print(f'Error while sending row {index}: {response.status_code}')
        else:
            print(response)


def main():
    parser = argparse.ArgumentParser(description='Streaming extension for KISS-Data-Generator.')

    parser.add_argument('csv_file', type=str, help='path to the csv file.')
    parser.add_argument('clearml_url', type=str, help='url of the clearml serving instance.')

    args = parser.parse_args()

    send_csv_data(args.csv_file, args.clearml_url)


if __name__ == "__main__":
    main()
