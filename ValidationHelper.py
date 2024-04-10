import os
import shutil
from PIL import Image
import json
import csv
from cloudMeta import plot_and_draw


'''
1. Get the base directory for clouds
2. Loop through the directory of clouds
    a. Foreach directory, print
        i. name of file
        ii. # of clouds
    b. Display cloud base plot
    c. Ask user if plot contains a cloud
        i. Store path to directory in confirmed_clouds list if clouds is confirmed
        ii. Store path to directory in confirmed_non_clouds list if clouds is not confirmed
    d. store result in csv
3. Ask user to draw cloud shapes onto confirmed_cloud plots
4. Ask user for permission to delete files that have been used



'''

# Purpose -> Validate cloud/non-cloud files
def main(paths_to_directories, confirmed_clouds_folder_path, confirmed_non_clouds_folder_path, info_csv):
    # Init. variables
    confirmed_clouds = []
    confirmed_non_clouds = []
    confirmed_names = []

    if not os.path.exists(info_csv):
        with open( info_csv, "w" ) as f:
            print(f"{info_csv} created!")
    with open( info_csv, newline='' ) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            confirmed_names.append(row[0])

    # Parent directory of clouds or non_clouds
    for parent_directory in paths_to_directories:

        # The clouds/non_cloud folders
        for child_directory in os.listdir(parent_directory):
            if child_directory in confirmed_names:
                break

            base_path = os.path.join(parent_directory, child_directory)
            print(base_path)

            # Display image
            plot_path = os.path.join(base_path, "plot.png")
            plot = Image.open(plot_path)
            plot.show()

            # Capture user input
            while True:

                user_input = input("Is there a cloud (yes/no): ").lower()
                print(base_path)
                if user_input == "yes":
                    confirmed_clouds.append(base_path)
                    print("Cloud Identified...Continuing to next cloud\n")
                    break
                elif user_input == "no":
                    confirmed_non_clouds.append(base_path)
                    print("Cloud not Identified...Continuing to next cloud\n")
                    break
                else:
                    print("User input not valid.\nIs there a cloud (yes/no): ")

            plot.close()

    # Confirm existence of output directories
    os.makedirs(confirmed_clouds_folder_path, exist_ok=True)
    os.makedirs(confirmed_non_clouds_folder_path, exist_ok=True)

    # Copy confirmed_clouds directories to confirmed_clouds_folder_path
    for confirmed_cloud in confirmed_clouds:
        cloud_name = os.path.basename(confirmed_cloud)
        dest_path = os.path.join(confirmed_clouds_folder_path, cloud_name)
        shutil.copytree(confirmed_cloud, dest_path, dirs_exist_ok=True)

        # Add info to the csv
        #   • <EMUS file name>,<cloud(yes/no)>,<date>,<martian_year>,<L_s(solar_long)>
        csvLine = ""

        jsonData = None
        with open(os.path.join(confirmed_cloud, "metadata.json")) as f:
            jsonData = json.load(f)

        csvLine += f"{jsonData['name']},yes,{jsonData['date']},{jsonData['martian_year']},{jsonData['L_s']}\n"

        with open( info_csv, 'a' ) as f:
            f.write(csvLine)


    # Copy confirmed_non_clouds directories to confirmed_non_clouds_folder_path
    for confirmed_non_cloud in confirmed_non_clouds:
        non_cloud_name = os.path.basename(confirmed_non_cloud)
        dest_path = os.path.join(confirmed_non_clouds_folder_path, non_cloud_name)
        shutil.copytree(confirmed_non_cloud, dest_path, dirs_exist_ok=True)

        # Add info to the csv
        #   • <EMUS file name>,<cloud(yes/no)>,<date>,<martian_year>,<L_s(solar_long)>
        csvLine = ""

        jsonData = None
        with open(os.path.join(confirmed_cloud, "metadata.json")) as f:
            jsonData = json.load(f)

        csvLine += f"{jsonData['name']},no,{jsonData['date']},{jsonData['martian_year']},{jsonData['L_s']}\n"

        with open( info_csv, 'a' ) as f:
            f.write(csvLine)


    print("Confirmed and confirmed non clouds successfully sorted\n\n")

    user_input = input("Draw confirmed clouds? (yes\\no)\n").lower()

    if user_input == "yes":
        draw_clouds(confirmed_clouds,confirmed_clouds_folder_path)


    # Give user option to delete the contents in paths_to_directories
    user_input = input("Delete source cloud/non_cloud files? (yes\\no)\n").lower()
    if user_input == "yes":
        for confirmed_cloud in confirmed_clouds:
            shutil.rmtree(confirmed_cloud)
            for confirmed_non_cloud in confirmed_non_clouds:
                shutil.rmtree(confirmed_non_cloud)

                print("Source files successfully deleted")


# Displays confirmed clouds, allows user to draw shape of clouds,
# stores the indicies of the drawn shape (both indicies and coords)
def draw_clouds(confirmed_clouds, confirmed_clouds_folder_path):
    for confirmed_cloud in confirmed_clouds:
        cloud_name = os.path.basename(confirmed_cloud)
        confirmed_cloud_path = os.path.join(confirmed_clouds_folder_path, cloud_name)

        metadataPath = os.path.join(confirmed_cloud_path, "metadata.json")
        plotPath = os.path.join(confirmed_cloud_path, "plot_draw.png")
        plot_and_draw(cloud_name, "../Data", metadataPath, 176, 182, plotPath)






if __name__ == "__main__":
    main(["cloud"], "confirmed_clouds", "confirmed_non_clouds", "info.csv")
