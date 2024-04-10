# Desired Features
    # Allow users full access to customization (config file?)
    # Have the config file allow for multiple layers of processing files - DONE
    # Allow users to explore one or more files - DONE
        # if directory is given, have the program process all files in the desired directory - DONE
    # Allow users to turn off meta data - DONE
    # Allow users to specify how detailed they want the plots and such
        # Default: one plot with json
        # Verbose: one plot for each wavelength with regular json


# Config file (format: json)
    # file paths (can be directory or list of files)
        # Recurse into directory once
    # Refining
        # wavelength min
        # wavelength max
        # min cloud area
        # section size
        # radiance min
        # radiance max
        # max altitude swap value
    # json output - default = True
        # True  -> Metadata is created
        # False -> Metadata is not created
    # Plot - default - Simple
        # None     -> No plot generated
        # Simple   -> Aggregated radiance plot generated
        # Advanced -> each wavelength's radiance plot is generated


# Add refining parameters into json
# Add all possible parameters
# have a copy of code in json
# Remove magic numbers and replace with defines
# Remove the season/month

# solar long - ls
# longitude - lon
# latitude - lat
# mar_year - martian_year


import json
import os
import threading
import time
from cloudMeta import file_contains_cloud
from cloudMeta import checkValidFile
from cloudMeta import plotCombinedWavelengthsSingleFilter
from cloudMeta import plotCombinedWavelengthsMultipleFilter
from cloudMeta import NumpyEncoder
from cloudMeta import plotCombinedWavelengthsSingle

def main():
    with open("CloudConfig.json", "r") as file:
        timeStart = time.time()

        configData = json.load(file)
        files = []

        # Loop through files
        for possibleFile in configData["Files"]:
            # Check if file is a possible emus data file
            if possibleFile.endswith("fits.gz"):
                files.append(possibleFile)
            # Check if file is actually a directory
            if os.path.isdir(possibleFile):
                subFiles = os.listdir(possibleFile)
                for subFile in subFiles:
                    # add files in said directory if fits.gz file's extension
                    if subFile.endswith("fits.gz"):
                        files.append(subFile)

        # Process files
        cloud_list_copy = files.copy()
        cloud_list = []
        non_cloud_list = []
        cloud_meta_list = []
        non_cloud_meta_list = []

        # Loop through config refining parameters
        for refiningParam in configData["Refining Parameters"]:
            # Reset positives list
            cloud_list = []
            cloud_meta_list = []

            # Process each folder with current refining parameters
            for file in cloud_list_copy:
                if checkValidFile(fileName=file, wavelength_min=refiningParam["Wavelength Min"], wavelength_max=refiningParam["Wavelength Max"]):
                    result, metadata = file_contains_cloud(fileName=file,
                                                            wavelength_min=refiningParam["Wavelength Min"],
                                                            wavelength_max=refiningParam["Wavelength Max"],
                                                            minimum_cloud_area=refiningParam["Min Cloud Area"],
                                                            section_size=refiningParam["Section Size"],
                                                            radiance_min=refiningParam["Radiance Min"],
                                                            radiance_max=refiningParam["Radiance Max"],
                                                            horizontal_radius=configData["Horizontal Radius"],
                                                            vertical_radius=configData["Vertical Radius"])

                    # Check if file contains cloud
                    if result:
                        # Save plot to cloud folder
                        cloud_list.append(file)
                        if configData["JSON Output"]:
                            cloud_meta_list.append(metadata)
                        print("Cloud")
                    # Otherwise
                    else:
                        # Save plot to not cloud folder
                        print("Not cloud")
                        non_cloud_list.append(file)
                        if configData["JSON Output"]:
                            non_cloud_meta_list.append(metadata)

            # Prepare new list of files to process
            cloud_list_copy = cloud_list.copy()




        # Plot
        for file in cloud_list:
            plotAndMeta(file, f"cloud/{file}", cloud_meta_list,
                        configData["Plot Wavelength Min"],
                        configData["Plot Wavelength Max"],
                        configData["JSON Output"],
                        configData["Plot - Default - Verbose - Both"],
                        configData["Refining Parameters"][-1],
                        configData["Refining Parameters"],
                        configData["Horizontal Radius"],
                        configData["Vertical Radius"])
        for file in non_cloud_list:
            plotAndMeta(file, f"non_cloud/{file}", non_cloud_meta_list,
                        configData["Plot Wavelength Min"],
                        configData["Plot Wavelength Max"],
                        configData["JSON Output"],
                        configData["Plot - Default - Verbose - Both"],
                        configData["Refining Parameters"][-1],
                        configData["Refining Parameters"],
                        configData["Horizontal Radius"],
                        configData["Vertical Radius"])

        print(f"Time to complete {time.time() - timeStart}")


def plotAndMeta(filename, path, metadata_list,  wavelength_min, wavelength_max,
                metadataFlag, plotSetting, refiningParam, refiningParamMeta,
                horizontal_radius, vertical_radius):

    # Create directory for new file
    os.makedirs(path, exist_ok=True)

    # Plot file and store
    print(f"Indicies |{getIndicies(filename, metadata_list)}|")
    if plotSetting == "Default" or plotSetting == "Both":
        plotCombinedWavelengthsSingleFilter(filename, wavelength_min, wavelength_max,
                                            f"{path}/plot")
        plotCombinedWavelengthsSingle(filename, wavelength_min, wavelength_max, refiningParam,
                                            f"{path}/plot_filtered", horizontal_radius, vertical_radius,
                                            getIndicies(filename, metadata_list) )
    if plotSetting == "Verbose" or plotSetting == "Both":
        plotCombinedWavelengthsMultipleFilter(filename, wavelength_min, wavelength_max, refiningParam,
        f"{path}/plot")


    # Write meta data and store
    if metadataFlag:
        for metadata in metadata_list:
            if metadata["name"] == filename:
                metadata["refining_parameters"] = refiningParamMeta
                with open(f"{path}/metadata.json", "w") as file:
                    json.dump(metadata, file, cls=NumpyEncoder, indent=4)

def getIndicies(filename, metadata_list):
    for metadata in metadata_list:
        if metadata["name"] == filename:
            return metadata["indices"]
    return []

if __name__ == "__main__":
    main()
