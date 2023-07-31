import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

def convert_hup_to_rid(rh_table,hup_id):
    # get number from hup_id
    hup_num = int(hup_id[3:6])
    # get rid from rid_hup_table
    rid = rh_table.loc[rh_table["hupsubjno"] == hup_num]["record_id"].values[0]
    return rid

parent_directory = r"/mnt/leif/littlab/users/ianzyong/sleep-atlas/data"

subdirectory_count = 0
sleep_result_count = 0
no_sleep_results = []
no_coherence = []
no_scores = []
empty_scores = []

for root, dirs, files in os.walk(parent_directory):
    # for each patient subdirectory that begins with "sub-"
    if root.split(os.sep)[-1].startswith("sub-"):
        # print dir name
        print(f"{root.split(os.sep)[-1]}:")
        subdirectory_count += 1
        # for each file in the subdirectory
        files = [f"\t{file}" for file in files if file.endswith("sleepstage.csv") and "night2" in file]
        print(files)
        # if no sleep result file was found, add patient to no_sleep_results list
        if len(files) == 0:
            no_sleep_results.append(root.split(os.sep)[-1])
        # check if subdir "func" exists
        if "func" not in dirs:
            # check if files exist in "func" subdir
            no_coherence.append(root.split(os.sep)[-1])
            no_scores.append(root.split(os.sep)[-1])
        # else, check if at least one file ending with "z-scores.csv" exists in subdir "func"
        elif len([file for file in os.listdir(os.path.join(root, "func")) if file.endswith("z-scores.csv")]) == 0:
            no_scores.append(root.split(os.sep)[-1])
        # check if the first file ending with "z-scores.csv" is an empty dataframe, ignore labels
        else:
            # account for index and column labels
            first_score_df = pd.read_csv(os.path.join(root, "func", [file for file in os.listdir(os.path.join(root, "func")) if file.endswith("z-scores.csv")][0]), index_col=0)
            # if all nans
            if first_score_df.isnull().values.all():
                empty_scores.append(root.split(os.sep)[-1])
            
            
                
        
# print out total number of subdirectories
print("Total number of patient subdirectories: ", subdirectory_count)
# print out total number of sleep result files
print("Patients with no sleep result files: ", sorted(no_sleep_results))
print(f"Count = {len(no_sleep_results)}")
# print out total number of patients with no coherence files
print("Patients with no coherence files: ", sorted(no_coherence))
print(f"Count = {len(no_coherence)}")
# print patients that are not in no_sleep_results but in no_coherence
print(f"Patients in no_coherence but not in no_sleep_results: ", sorted(list(set(no_coherence) - set(no_sleep_results))))
# print patients in no_scores
print("Patients with no scores: ", sorted(no_scores))
print(f"Count = {len(no_scores)}")
# print patients in no_scores but not in no_coherence
print("Patients in no_scores but not in no_coherence: ", sorted(list(set(no_scores) - set(no_coherence))))
# print patients in empty_scores
print("Patients with empty scores: ", sorted(empty_scores))
print(f"Count = {len(empty_scores)}")

# plot and save heatmap of atlases
stages = ["W", "N2", "N3", "R"]
bands = ["delta", "theta", "alpha", "beta", "gamma", "broad"]
# for each stage
for stage in stages:
    for band in bands:
        filename = os.path.join(parent_directory, "atlas", f"{stage}_{band}_atlas.csv")
        # read in csv file as dataframe
        df = pd.read_csv(filename, index_col=0, header=0)
        # for each value in dataframe
        for index, row in df.iterrows():
            # for each value in row
            for k in range(len(row.values)):
                value = row.values[k]
                # store value as string
                value = str(value)
                # interpret value as list
                # if value is empty
                if value == "[]":
                    conn_list = []
                else:
                    conn_list = [float(s.strip()) for s in value[1:-1].split(',')]
                #print(conn_list)
                # repalce value in dataframe with average of list
                df.loc[index].iloc[k] = np.nanmean(conn_list)
        # create a heatmap of the dataframe
        plt.figure(figsize=(20,20))
        cmap = mpl.cm.get_cmap("viridis").copy()
        cmap.set_bad(color='black')
        # convert dataframe to type float
        df = df.astype(float)
        plt.imshow(df, cmap=cmap)
        plt.colorbar(label="Average coherence")
        plt.xlabel("Region")
        plt.ylabel("Region")
        plt.title(f"Stage {stage}, {band}")
        # set xticklabels
        plt.xticks(np.arange(len(df.columns)), df.columns, rotation=90)
        plt.yticks(np.arange(len(df.columns)), df.columns)
        plt.savefig(os.path.join(parent_directory, "atlas",f"{stage}_{band}_atlas.png"))
        print(f"Saved heatmap of {stage} {band} atlas")

# for each csv file in atlas directory
for root, dirs, files in os.walk(os.path.join(parent_directory, "atlas")):
    # read in csv file as dataframe
    for file in files:
        if file.endswith(".csv"):
            print(f"===== {file}")
            # read in csv file as dataframe
            df = pd.read_csv(os.path.join(root, file), index_col=0, header=0)
            # for each row
            for index, row in df.iterrows():
                # get list of lists in row
                lol = [x[1:-1].strip().split(",") for x in row.values]
                # get total number of elements in list of lists, disregarding empty strings
                total = sum([len(x) for x in lol if x != ['']])
                # print region label of index and total
                #print(f"{index}: {total}")

    # only do one file
    # apply function to each value of dataframe
    atlas_counts = df.applymap(lambda x: len(x[1:-1].strip().split(",")) if x != "[]" else 0)
    plot_data = atlas_counts.replace(0, np.nan)
    # create a heatmap of the dataframe
    plt.figure(figsize=(20,20))
    cmap = mpl.cm.get_cmap("viridis").copy()
    cmap.set_bad(color='black')
    plt.imshow(plot_data, cmap=cmap)
    # label colorbar with "Count" text
    plt.colorbar().set_label("Count")
    # make the colorbar the same height as the heatmap
    plt.gcf().axes[-1].set_ylim(0, plt.gcf().get_size_inches()[0])
    plt.title("Counts of sampled region pairs")
    plt.xlabel("Region")
    plt.ylabel("Region")
    # add tick labels
    plt.xticks(range(atlas_counts.shape[0]), atlas_counts.columns, rotation=90)
    plt.yticks(range(atlas_counts.shape[1]), atlas_counts.index)
    # annotate heatmap with values, use small font and large heatmap
    for i, j in itertools.product(range(atlas_counts.shape[0]), range(atlas_counts.shape[1])):
        if atlas_counts.iloc[i, j] != 0:
            plt.text(j, i+0.1, atlas_counts.iloc[i, j], horizontalalignment="center", color="black" if atlas_counts.iloc[i, j] > 400 else "white", fontsize=8)
    # save to disk
    plt.savefig(os.path.join(root, "sleep_atlas_counts.png"))
    # print labels of top ten most sampled regions
    print(atlas_counts.sum(axis=1).sort_values(ascending=False).head(10))
    print("Atlas counts saved to disk.")

# load combined_atlas_metadata.csv
combined_atlas_metadata = pd.read_csv("/mnt/leif/littlab/users/ianzyong/sleep-atlas/util/combined_atlas_metadata.csv")

top_num = 20

# get list of labels of top most sampled regions
top = atlas_counts.sum(axis=1).sort_values(ascending=False).head(top_num).index.tolist()
# get counts of each label
top_df = atlas_counts.sum(axis=1).sort_values(ascending=False).head(top_num)
print(top_df)
# for each label, look up the mni coordinates in combined_atlas_metadata.csv
with open(os.path.join(parent_directory, f"most_sampled_coords.node"), "w") as f:
    for label in top:
        print(label)
        coords = combined_atlas_metadata.loc[combined_atlas_metadata["reg"] == label][["mni_x", "mni_y", "mni_z"]].values.tolist()
        # average coords and convert to list
        avg_coords = np.nanmean(coords, axis=0).tolist()
        print(f"avg_coords = {avg_coords}")
        # get value of label in top
        count = top_df[label].astype(int)
        print(f"count = {count}")
        avg_coords.extend([1,count/5000])
        # write to tab-spaced text file
        f.write("\t".join([str(x) for x in avg_coords])+"\n")

print("Most sampled coordinates saved to disk.")

# get list of patient paths
patient_paths = [os.path.join(parent_directory, f"{pt}") for pt in os.listdir(parent_directory) if "sub-" in pt]

stage_dict = {"W": 0, "N2": 1, "N3": 2, "R": 3}
band_dict = {"broad": 0, "gamma": 1, "beta": 2, "alpha": 3, "theta": 4, "delta": 5}

# load rid_hup_table.csv
rid_hup_table = pd.read_csv("/mnt/leif/littlab/users/ianzyong/sleep-atlas/util/rid_hup_table.csv")

for test_patient_path in patient_paths:
    pt = test_patient_path.split(os.sep)[-1].split("-")[1]
    # if z-score files exist
    func_dir = os.path.join(test_patient_path,"func")
    if os.path.exists(func_dir) and len([file for file in os.listdir(func_dir) if file.endswith("z-scores.csv")]) > 0:
        print(test_patient_path)
        # plot all z-score files as heatmaps in a grid
        for root, dirs, files in os.walk(func_dir):
            # 4*6 subplots
            fig, axs = plt.subplots(4, 6, figsize=(35, 25), dpi=300)
            
            # for each file in func directory
            for i, file in enumerate([file for file in files if (file.endswith("z-scores.csv") and "night" in file)]):
                print(file)
                # read in csv file as dataframe
                df = pd.read_csv(os.path.join(root, file), index_col=0)
                # remove all columns and rows with labels that contain "EKG" or "ECG"
                df = df.loc[:, ~df.columns.str.contains("EKG|ECG", regex=True)]
                df = df.loc[~df.index.str.contains("EKG|ECG", regex=True), :]

                # get ID of patient
                hup_id = file.split("_")[0].split("-")[1]
                rid = convert_hup_to_rid(rid_hup_table, hup_id)
                full_rid = f"sub-RID{rid:04d}"

                # get rows of combined_atlas_metadata that match patient ID
                patient_rows = combined_atlas_metadata.loc[combined_atlas_metadata["pt"] == full_rid]

                # get list of values in "name" column that have True in "ch1_resected" column or "ch2_resected" column
                #print(patient_rows)
                resected_bipolar_pairs = patient_rows.loc[(patient_rows["ch1_resected"] == True) | (patient_rows["ch2_resected"] == True)]["name"].values
                print(f"Resected bipolar pairs: {resected_bipolar_pairs}")

                # get value from "Engel" column
                try:
                    engel = patient_rows["engel"].values[0]
                except IndexError:
                    engel = np.nan

                # create a heatmap of the dataframe
                cmap = mpl.cm.get_cmap("viridis").copy()
                cmap.set_bad(color='black')
                cmap.set_over(color='white')

                this_stage = file.split("_")[-3]
                this_band = file.split("_")[-2]
                x = stage_dict[this_stage]
                y = band_dict[this_band]

                vmax = 5
                axs[x, y].imshow(df, cmap=cmap, vmin=0, vmax=vmax)
                # use large font
                axs[x, y].set_title(f"Stage {this_stage}, {this_band}", fontsize=20)
                axs[x, y].set_xlabel("Electrode pair")
                axs[x, y].set_ylabel("Electrode pair")
                # add tick labels
                axs[x, y].set_xticks(range(df.shape[0]))
                axs[x, y].set_yticks(range(df.shape[1]))
                # use small font
                # color tick labels red if they are in resected_bipolar_pairs
                axs[x, y].set_xticklabels(df.columns, fontsize=3, rotation=90)
                axs[x, y].set_yticklabels(df.index, fontsize=3)
                #[i.set_color("green") for i in axs[x, y].get_xticklabels()]
                # set resected pairs to red color
                for lbl in axs[x, y].get_xticklabels():
                    if lbl.get_text() in resected_bipolar_pairs:
                        lbl.set_color("red")
                for lbl in axs[x, y].get_yticklabels():
                    if lbl.get_text() in resected_bipolar_pairs:
                        lbl.set_color("red")
                # adjust padding to the left
                axs[x, y].tick_params(pad=0.98)

        # save to disk
        # use one colorbar for all subplots
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        try:
            fig.colorbar(axs[0, 0].images[0], cax=cbar_ax)
        except IndexError:
            print("No data to plot, skipping...")
            continue
        # colorbar title
        cbar_ax.set_ylabel("|z|", fontsize=30)
        # colorbar tick size
        cbar_ax.tick_params(labelsize=20)
        # set color bar limits to 0 and 5
        cbar_ax.set_ylim(0, 5)
        # make color bar thinner
        cbar_ax.set_aspect(10)

        # add title to subplots
        # if engel is not nan
        if engel == engel:
            fig.suptitle(f"{pt} |z|-scores by electrode pair, sleep stage, and frequency band (Engel = {int(engel)})", fontsize=50)
        else:
            fig.suptitle(f"{pt} |z|-scores by electrode pair, sleep stage, and frequency band (Engel = unknown)", fontsize=50)
        # add subtitle in italic under main title
        fig.text(0.5, 0.935, f"Resected electrode pairs are colored red. Missing scores are colored black, and outliers (|z| > {vmax}) are colored white.", fontsize=26, ha="center", va="center", fontstyle="italic")
        # use tight layout
        #fig.tight_layout()
        plt.savefig(os.path.join(root, f"{pt}_sleep_z-scores.png"))
        print("Z-score figure saved to disk.")

print("Done.")

