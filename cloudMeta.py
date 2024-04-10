import numpy as np
import os
import matplotlib.pyplot as plt
import time
from read_emus_data import read_emus_l2a
from read_emus_data import read_emus_l2a_fov
import math
import json
from scipy.ndimage import label, center_of_mass
from skimage.restoration import denoise_tv_chambolle
from skimage import img_as_float
from skimage.morphology import binary_erosion, disk
from datetime import datetime


# ===================== TODO ========================
# Clean code up to remove unnecessary computations
# fix bug with csv
# start getting dataset on server
# email Justin if any specific info required
# ===================== END TODO =====================


# ===================== DONE =====================
# =================== END DONE ===================


# ~176 nm (highest altitude clouds)
# ~182 nm (lower altitude clouds)
# The dimensions of the radiance arrays are (X, Y, λ),
# and the wavelength array is (Y, λ).

def get_lat_lon(x, y, lats, lons):

    return lats[int(y)][0][int(x)], lons[int(y)][0][int(x)]

def onclick(event, metadata, indices, indices_coords, lats, lons, ax, savePath):
    ix, iy = event.xdata, event.ydata
    lat, lon = get_lat_lon(ix, iy, lats, lons)
    indices.append((lat, lon))
    indices_coords.append((ix, iy))

    # Plot the clicked point
    ax.plot(ix, iy, 'ro')  # Plot the point as a red circle

    # If there's at least one previous point, draw a line to the last point
    if len(indices_coords) > 1:
        x_vals = [indices_coords[-2][0], indices_coords[-1][0]]
        y_vals = [indices_coords[-2][1], indices_coords[-1][1]]
        ax.plot(x_vals, y_vals, 'r-')  # Draw a red line between the points

    plt.draw()  # Update the plot with the new line and point
    if savePath != "":
        plt.savefig(savePath)

    print(f"Clicked coordinates: {lat}, {lon}")

def plot_and_draw(fileName, directoryPath, metadataPath,
                  wavelength_min, wavelength_max, savePath=""):
    timeStart = time.time()
    out_dict = read_emus_l2a_fov(fileName, directoryPath)
    wavelength = np.array(out_dict["wavelength"])
    radiance_norm = np.array(out_dict["radiance_norm"])
    latitude = np.array(out_dict["lat"])
    longitude = np.array(out_dict["lon"])
    timeEnd = time.time()
    print(f"Time to get data is {timeEnd - timeStart} seconds")

    fig, ax = plt.subplots()
    indices = []
    indices_coords = []

    # Connect the onclick event handler
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, out_dict, indices, indices_coords, latitude, longitude, ax, savePath))

    x_values = wavelength[:, 0]
    y_values = wavelength[:, 1]
    # For each wavelength in the predefined range
    wavelength_range = range(wavelength_min, wavelength_max + 1)
    for lambda_index in wavelength_range:
        try:
            x_values = np.arange(radiance_norm.shape[1])
            y_values = np.arange(radiance_norm.shape[0])
            radiance_data = radiance_norm[:, :, lambda_index]
        except:
            print("error")
            plt.close()
            return

        plt.pcolormesh(x_values, y_values, radiance_data, shading='auto', alpha=0.5, cmap='inferno')

    # Shared elements for the plot
    plt.colorbar(label='Radiance')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Radiance Heatmap for Cloud Detection ({wavelength_range.start}-{wavelength_range.stop-1})')

    if savePath != "":
        plt.savefig(savePath)

    plt.show(block=True)


    # read in json at metadataPath
    metadata = None
    with open( metadataPath, "r" ) as metadata_file:
        metadata = json.load(metadata_file)
        metadata["indices"] = indices
    with open(metadataPath, 'w') as f:
        json.dump(metadata, f, indent=4)

def plotCombinedWavelengthsSingleFilter(fileName, wavelength_min, wavelength_max, savePath=""):
    timeStart = time.time()
    out_dict = read_emus_l2a_fov(fileName, "../Data")
    wavelength = np.array(out_dict["wavelength"])
    radiance_norm = np.array(out_dict["radiance_norm"])
    latitude = np.array(out_dict["lat"])
    longitude = np.array(out_dict["lon"])
    timeEnd = time.time()
    print(f"Time to get data is {timeEnd - timeStart} seconds")



    x_values = wavelength[:, 0]
    y_values = wavelength[:, 1]
    # For each wavelength in the predefined range
    wavelength_range = range(wavelength_min, wavelength_max + 1)
    for lambda_index in wavelength_range:
        try:
            x_values = np.arange(radiance_norm.shape[1])
            y_values = np.arange(radiance_norm.shape[0])
            radiance_data = radiance_norm[:, :, lambda_index]
        except:
            print("error")
            plt.close()
            return

        plt.pcolormesh(x_values, y_values, radiance_data, shading='auto', alpha=0.5, cmap='inferno')

    # Shared elements for the plot
    plt.colorbar(label='Radiance')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Radiance Heatmap for Cloud Detection ({wavelength_range.start}-{wavelength_range.stop-1})')

    if savePath != "":
        plt.savefig(savePath)
    else:
        plt.show()

    plt.close()

def plotCombinedWavelengthsSingle(fileName, wavelength_min, wavelength_max,
                                  refiningParam, savePath="",
                                  horizontal_radius=0, vertical_radius=0, indicesXY=[]):
    timeStart = time.time()
    out_dict = read_emus_l2a_fov(fileName, "../Data")
    wavelength = np.array(out_dict["wavelength"])
    radiance_norm = np.array(out_dict["radiance_norm"])
    latitude = np.array(out_dict["lat"])
    longitude = np.array(out_dict["lon"])
    alt = np.array(out_dict["mrh_alt"])
    timeEnd = time.time()
    print(f"Time to get data is {timeEnd - timeStart} seconds")

    x_values = wavelength[:, 0]
    y_values = wavelength[:, 1]

    # Sum the radiance values over all wavelengths
    aggregated_radiance = np.sum(radiance_norm[:, :, wavelength_min:wavelength_max + 1], axis=-1)

    # Determine the center of the image
    center_y, center_x = aggregated_radiance.shape[0] // 2, aggregated_radiance.shape[1] // 2

    radius = min(center_x, center_y)
    semi_major_axis = int(horizontal_radius * radius)
    semi_minor_axis = int(vertical_radius * radius)

    # Create a grid of distances from the center
    Y, X = np.ogrid[:aggregated_radiance.shape[0], :aggregated_radiance.shape[1]]
    dist_from_center = np.sqrt(((X - center_x) / semi_major_axis)**2 + ((Y - center_y) / semi_minor_axis)**2)

    # Create an oval mask where distances satisfy the elliptical condition
    oval_mask = dist_from_center <= 1.0
    aggregated_radiance_cropped = np.where(oval_mask, aggregated_radiance, 0)


    # Split the image into squares and get their original indices
    section_size=refiningParam["Section Size"]
    squares, indices = split_into_squares_with_indices(aggregated_radiance_cropped, section_size)
    processed_squares = []

    for square, (idx, idy) in zip(squares, indices):
        local_radiance_threshold_min = adaptive_threshold(square, refiningParam["Radiance Min"])
        local_radiance_threshold_max = adaptive_threshold(square, refiningParam["Radiance Max"])
        # local_radiance_threshold_min = max(local_radiance_threshold_min, aggregated_radiance.mean())

        #  Use clustering to remove noise (less than 10 pixels-ish)
        #  Convert circuluar mask to square mask

        binary_mask = square > local_radiance_threshold_min
        selem = disk(erosion_value)
        eroded_mask = binary_erosion(binary_mask, selem)
        eroded_square = np.where(eroded_mask, square, 0)
        processed_squares.append((eroded_square, (idx, idy)))

    # Initialize an empty array to recombine the processed squares
    recombined_image = np.zeros_like(aggregated_radiance_cropped)

    # Recombine the processed squares
    for processed_square, (idx, idy) in processed_squares:
        recombined_image[idx:idx+section_size, idy:idy+section_size] = processed_square

    x_values = np.arange(recombined_image.shape[1])
    y_values = np.arange(recombined_image.shape[0])

    plt.pcolormesh(x_values, y_values, recombined_image, shading='auto', alpha=0.5, cmap='inferno')

    # Shared elements for the plot
    plt.colorbar(label='Radiance')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Radiance Heatmap for Cloud Detection ({wavelength_max}-{wavelength_min})')

    # Plotting the provided indices on top of the heatmap
    for pointList in indicesXY:
        for point in pointList:
            print(f"Point: |{point}|")
            print(f"X: |{ int(point['X']) }|")
            print(f"X: |{int(point['y'])}|")
            plt.scatter(int(point['X']), int(point['y']), color='blue', s=10, label='Point')

    if savePath != "":
        plt.savefig(savePath)
    else:
        plt.show()

    plt.close()

def plotCombinedWavelengthsMultipleFilter(fileName, wavelength_min, wavelength_max, savePath=""):
    timeStart = time.time()
    out_dict = read_emus_l2a_fov(fileName, "../Data")
    wavelength = np.array(out_dict["wavelength"])
    radiance_norm = np.array(out_dict["radiance_norm"])
    latitude = np.array(out_dict["lat"])
    longitude = np.array(out_dict["lon"])
    timeEnd = time.time()
    print(f"Time to get data is {timeEnd - timeStart} seconds")


    x_values = wavelength[:, 0]
    y_values = wavelength[:, 1]
    # For each wavelength in the predefined range
    wavelength_range = range(wavelength_min, wavelength_max + 1)
    for lambda_index in wavelength_range:
        try:
            x_values = np.arange(radiance_norm.shape[1])
            y_values = np.arange(radiance_norm.shape[0])
            radiance_data = radiance_norm[:, :, lambda_index]
        except:
            print("Error in fetching data for wavelength index:", lambda_index)
            continue

        plt.figure()  # Create a new figure for each wavelength
        plt.pcolormesh(x_values, y_values, radiance_data, shading='auto', alpha=0.5, cmap='inferno')
        plt.colorbar(label='Radiance')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Radiance Heatmap for Cloud Detection at {lambda_index} nm')

        if savePath != "":
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(f"{savePath}Radiance_{lambda_index}.png")
        else:
            plt.show()

        plt.close()

def plot_radiance_with_thresholds(fileName, wavelength_min, wavelength_max, threshold_min, threshold_max, savePath=""):
    timeStart = time.time()
    out_dict = read_emus_l2a_fov(fileName, "../Data")
    radiance_norm = np.array(out_dict["radiance_norm"])
    alt = np.array(out_dict["mrh_alt"])
    timeEnd = time.time()
    print(f"Time to get data is {timeEnd - timeStart} seconds")

    accumulated_radiance = np.zeros(radiance_norm[:,:,0].shape)

    for lambda_index in range(wavelength_min, wavelength_max + 1):
        try:
            radiance_data = radiance_norm[:, :, lambda_index]
            accumulated_radiance += radiance_data
        except:
            print("error")
            plt.close()
            return

    # Calculate average radiance across wavelengths
    average_radiance = accumulated_radiance / (wavelength_max - wavelength_min + 1)
    above_threshold = average_radiance > threshold_max

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # Creating 2 side-by-side subplots

    # Plot unfiltered average radiance data on the first subplot
    pcm1 = ax1.pcolormesh(average_radiance, shading='auto', cmap='inferno')
    fig.colorbar(pcm1, ax=ax1, label='Radiance')
    ax1.set_title('Average Radiance Data')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Plot regions above the threshold on the second subplot
    pcm2 = ax2.pcolormesh(np.ma.masked_where(~above_threshold, average_radiance), shading='auto', cmap='Reds')
    fig.colorbar(pcm2, ax=ax2, label='Radiance Above Threshold')
    ax2.set_title(f'Radiance Data Above {threshold_max}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    plt.tight_layout()  # Adjust layout so plots don't overlap

    if savePath != "":
        plt.savefig(savePath)
    else:
        plt.show()

    plt.close()

def file_contains_cloud_helper(radiance_data, min_cloud_area):
    cloud_found_flag = False
    cloud_areas = []
    cloud_centers = []
    cloud_indices = []
    # Count the number of contiguous regions with high radiance values
    labeled_array, num_features = label(radiance_data)

    # Check if any of the cloud regions have an area within the specified range
    for i in range(num_features):
        area = np.sum(labeled_array == i+1)

        if min_cloud_area <= area:
            center = center_of_mass(labeled_array, labels=labeled_array, index=i)
            cloud_found_flag = True
            cloud_areas.append(area)
            cloud_centers.append(center)
            cloud_indices.append( np.where(labeled_array == i+1) )

    return cloud_found_flag, cloud_areas, cloud_centers, cloud_indices

def adaptive_threshold(radiance_data, num_std_dev=2):
    """Determine threshold based on mean and standard deviation."""
    mean_val = np.mean(radiance_data)
    std_val = np.std(radiance_data)
    return mean_val + num_std_dev * std_val

def split_into_squares_with_indices(aggregated_radiance, square_size):
    squares = []
    indices = []
    height, width = aggregated_radiance.shape

    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            square = aggregated_radiance[i:i+square_size, j:j+square_size]
            squares.append(square)
            indices.append((i, j))

    return squares, indices

def file_contains_cloud(fileName, wavelength_min, wavelength_max,
                        minimum_cloud_area, section_size, radiance_min,
                        radiance_max,
                        horizontal_radius, vertical_radius):
    radiance_swap_value = 5000
    max_cloud_area = 50000
    cloud_detected = False

    out_dict = read_emus_l2a_fov(fileName, "../Data")
    radiance_norm = np.array(out_dict["radiance_norm"])
    alt = np.array(out_dict["mrh_alt"])
    lat = np.array(out_dict["lat"])
    long = np.array(out_dict["lon"])

    # Sum the radiance values over all wavelengths
    aggregated_radiance = np.sum(radiance_norm[:, :, wavelength_min:wavelength_max + 1], axis=-1)

    # Determine the center of the image
    center_y, center_x = aggregated_radiance.shape[0] // 2, aggregated_radiance.shape[1] // 2

    radius = min(center_x, center_y)
    semi_major_axis = int(horizontal_radius * radius)
    semi_minor_axis = int(vertical_radius * radius)

    # Create a grid of distances from the center
    Y, X = np.ogrid[:aggregated_radiance.shape[0], :aggregated_radiance.shape[1]]
    dist_from_center = np.sqrt(((X - center_x) / semi_major_axis)**2 + ((Y - center_y) / semi_minor_axis)**2)

    # Create an oval mask
    oval_mask = dist_from_center <= 1.0
    aggregated_radiance_cropped = np.where(oval_mask, aggregated_radiance, 0)

    # Split the image into squares and get their indicies
    squares, indices = split_into_squares_with_indices(aggregated_radiance_cropped, section_size)
    processed_squares = []

    # Loop through squares
    for square, (idx, idy) in zip(squares, indices):
        local_radiance_threshold_min = adaptive_threshold(square, radiance_min)
        local_radiance_threshold_min = max(local_radiance_threshold_min, aggregated_radiance.mean())
        local_radiance_threshold_max = adaptive_threshold(square, radiance_max)

        # Apply binary mask with local thresholds
        binary_mask = square > local_radiance_threshold_min
        selem = disk(erosion_value)
        eroded_mask = binary_erosion(binary_mask, selem)
        eroded_square = np.where(eroded_mask, square, 0)
        processed_squares.append((eroded_square, (idx, idy)))


    recombined_image = np.zeros_like(aggregated_radiance)
    # Recombine the processed squares
    for processed_square, (idx, idy) in processed_squares:
        recombined_image[idx:idx+section_size, idy:idy+section_size] = processed_square

    cloud_areas = []
    cloud_centers = []
    coord_centers = []
    indices = []
    indices_coords = []
    indices_xy = []
    cloud_found_flag, areas, centers, indices = file_contains_cloud_helper(recombined_image, minimum_cloud_area)

    if cloud_found_flag:
        cloud_detected = True

        for cloud_index in range(len(areas)):
            area = areas[cloud_index]
            center = centers[cloud_index]
            cloud_areas.append(area)
            cloud_x, cloud_y = center[0], center[1]

            # Get cloud center
            if not (np.isnan(cloud_x) or np.isnan(cloud_y)):
                cloud_x, cloud_y = int(cloud_x), int(cloud_y)
                cloud_centers.append(center)
                coord_centers.append( {"lat": lat[cloud_x][0][cloud_y], "lon": long[cloud_x][0][cloud_y]}  )

                # Get cloud shape
                # for cloud_shape_index in indices:
                #     indices_local_xy = []
                #     indices_local_coords = []
                #     for cloud_shape_index_inner in range(len(cloud_shape_index)):
                #         x,y = cloud_shape_index[0][cloud_shape_index_inner], cloud_shape_index[1][cloud_shape_index_inner]
                #         indices_local_xy.append( {"X": x, "y": y} )
                #         indices_local_coords.append( {"lat": lat[x][0][y], "lon": long[x][0][y]}  )
                #     indices_xy.append( indices_local_xy )
                #     indices_coords.append( indices_local_coords )


    # Get elapsed time and convert to minutes
    elapsed_time = (datetime.fromisoformat(out_dict["observation_end_time"])
                    - datetime.fromisoformat(out_dict["observation_start_time"])
                    ).total_seconds()
    elapsed_time /= 60
    elapsed_time = round(elapsed_time, 4)

    metadata = {
        "name": fileName,
        "radiance_stats": f"Min: {aggregated_radiance.min()}\nMean: {aggregated_radiance.mean()}\nMax: {aggregated_radiance.max()}",
        "cloud_area": cloud_areas,
        "wavelength_range": (float(wavelength_min), float(wavelength_max)),
        "indexes": cloud_centers,
        "coordinates": coord_centers,
        "date": out_dict["date"],
        "observation_start_time": out_dict["observation_start_time"],
        "observation_end_time": out_dict["observation_end_time"],
        "time_elapsed_minutes": elapsed_time,
        "L_s": out_dict["solar_long"],
        "quality_flag": out_dict["quality_flag"],
        "images_recieved": out_dict["images_recieved"],
        "martian_year": int(out_dict["mar_year"]),
        "horizontal_radius": horizontal_radius,
        "vertical_radius": vertical_radius,
        "indices_coords": indices_coords,
        "indices": indices_xy
    }

    return cloud_detected, metadata

def checkValidFile(fileName, wavelength_min, wavelength_max):
    out_dict = read_emus_l2a(fileName, "../Data")
    radiance_norm = np.array(out_dict["radiance_norm"])

    for lambda_index in range(wavelength_min, wavelength_max + 1):
        try:

            radiance_data = radiance_norm[:, :, lambda_index]
        except:
            return False

    return True

def process_files(file_list, count, clouds_list, non_cloud_list, cloud_meta_list, non_cloud_meta_list,
                    wavelength_min, wavelength_max, min_cloud_area, section_size,
                    radiance_min, radiance_max, radiance_swap_value=5000):

    for file in file_list:
        if checkValidFile(file, wavelength_min, wavelength_max):
            print(f"Exploring {file}")
            result, metadata = file_contains_cloud(file, wavelength_min, wavelength_max, min_cloud_area, section_size, radiance_min, radiance_max, radiance_swap_value=radiance_swap_value) # Gets all clouds and some false positives

            # Check if file contains cloud
            if result:
                # Save plot to cloud folder
                clouds_list.append(file)
                cloud_meta_list.append(metadata)
                print("Cloud")
            # Otherwise
            else:
                # Save plot to not cloud folder
                print("Not cloud")
                non_cloud_list.append(file)
                non_cloud_meta_list.append(metadata)


    print(f"Thread {count} completed")

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

if __name__ == "__main__":
    main()
