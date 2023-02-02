# script to generate evenly spaced intervals for a list of patients
# run the outputted text file as an argument for get_files.py
import argparse
import getpass
from ieeg.auth import Session
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--user', required=True, help='username')
parser.add_argument('-p', '--password',
                    help='password (will be prompted if omitted)')
parser.add_argument('csv_path', type=str)
parser.add_argument('interval_spacing_usec', type=int)
parser.add_argument('interval_duration_usec', type=int)
parser.add_argument('durs_path', nargs="?", type=str)
# parser.add_argument('dataset', help='dataset name')
# parser.add_argument('start', type=int, help='start offset in usec')
# parser.add_argument('duration', type=int, help='number of usec to request')

args = parser.parse_args()

if not args.password:
    args.password = getpass.getpass()

# iEEG_filename = input('Input iEEG_filename: ')
# rid = input('Input RID: ')
# start_time_usec = int(input('Input start_time_usec: '))
# stop_time_usec = int(input('Input stop_time_usec: '))
# time between each interval
# interval_spacing_usec = int(input('Input interval_spacing_usec: '))
# duration of each interval
# interval_duration_usec = int(input('Input interval_duration_usec: '))
# removed_channels = input('Input removed_channels: ')
# filename = input('Input filename: ')

filename = f"sleep_ints_{int(args.interval_spacing_usec/1000000)}sp_{int(args.interval_duration_usec/1000000)}dur"
durations = []

if not args.durs_path:
    print("Getting recording durations for requested subjects...")

    with Session(args.user, args.password) as session:
        with open(args.csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)
            for row in reader:
                try:
                    print(row)
                    dataset_name = row[3]
                    ds = session.open_dataset(dataset_name)
                    duration = ds.get_time_series_details(ds.ch_labels[0]).duration
                    durations.append(duration)
                    print(f"Duration = {duration}")
                    session.close_dataset(dataset_name)
                except:
                    print("Dataset could not be accessed.")

    with open("durs_of_requested.txt","a+") as f:
        f.write("\n".join(map(str,durations)))

else:
    with open(args.durs_path,"r") as f:
        durations = f.readlines()
    
with open(args.csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)
        k = 0
        for row in reader:
            if all(row):
                stop_time_usec = int(float(durations[k].strip()))
                interval_spacing_usec = args.interval_spacing_usec
                iEEG_filename = row[3]
                rid = row[1]
                interval_duration_usec = args.interval_duration_usec
                removed_channels = []
                with open("{}.txt".format(filename), 'a') as txt:
                    for interval_start in range(0,stop_time_usec,interval_spacing_usec):
                        txt.write('{}\n'.format(iEEG_filename))
                        txt.write('{}\n'.format(rid))
                        txt.write('{}\n'.format(interval_start))
                        txt.write('{}\n'.format(interval_start+interval_duration_usec))
                        txt.write('{}\n'.format(removed_channels))
                k += 1
                print(f"Wrote intervals for patient #{k}")

print('File saved to {}.txt.'.format(filename))
