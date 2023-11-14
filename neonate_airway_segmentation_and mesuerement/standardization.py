import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt




# read from execl file the measurments data.
# return the CSA of the two nasal cavities. (connected and disconnected area)

def read(exelpath, inferior=False):

    df_list = []
    sheet_name = ["cs r data", "cs l data", "notcc cs r data", "notcc cs l data"]

    for name in sheet_name:
        if inferior is True:
            name = "inferior " + name
        df_temp = pd.read_excel(exelpath, sheet_name=name)
        df_temp = df_temp.loc[(df_temp["percentage of nasal airway"] >= 0) & (df_temp["percentage of nasal airway"] <= 100),["percentage of nasal airway","area"]]
        df_temp = df_temp.reset_index(drop=True)
        if df_temp.empty or df_temp[df_temp["area"] > 0].empty:
            continue
        df_list.append(df_temp)


        sum_df = df_list[0]
    for i in range(1,len(df_list)):
        sum_df["area"] += df_list[i]["area"].reset_index(drop=True)

    return sum_df



# calculating the average and STD of CSA in a specific region
# between min_percent and max_percent
# for comparing the CSA in severe, moderate CNPAS and normal cases
# percent="percentage of pa-ch" the midnasal region
# Theoretically possible also "percentage of airway" for the whole airway

def average(exel_path, inferior=False , percent="percentage of pa-ch",min_percent=0, max_percent=10):
    df_list = []
    sheet_name = ["cs r data", "cs l data", "notcc cs r data", "notcc cs l data"]
    for name in sheet_name:
        if inferior is True:
            name = "inferior " + name
        df_temp = pd.read_excel(exel_path, sheet_name=name)
        df_temp = df_temp.loc[(df_temp[percent] >= min_percent) &
                              (df_temp[percent] <= max_percent),["area"]]
        if df_temp.empty or df_temp[df_temp["area"] > 0].empty:
            continue

        df_list.append(df_temp.reset_index(drop=True))

    sum_df = df_list[0]["area"]

    for i in range(1,len(df_list)):
        sum_df += df_list[i]["area"]

    vol = round(sum_df.sum(), 2)
    avg = round(sum_df.mean(), 2)
    std = round(sum_df.std(), 2)

    return vol, avg, std





def normalize_dataframe(df, perctange = "percentage of pa-ch"):
    """
    Normalize the dataframe by selecting specific columns and rounding the "percentage of pa-ch" values.
    Create a new dataframe with normalized data in steps of 0.5%.
    """
    # Select specific columns
    columns = [perctange, "area", "avg_width", "max_width"]

    if perctange == "percentage of pa-ch":
        # Take only the internal airway region
        df_reduce = df[df['region'] == "airway"][columns]


    else:
        pa_percent = df[df["percentage of pa-ch"] == 0][perctange]
        ch_percent  = df[df["percentage of pa-ch"] == 100][perctange]
        df_reduce = df[columns]


    # Create a new DataFrame for normalized data
    normalize_df = pd.DataFrame(columns=df_reduce.columns)

    # Round the "percentage of nasal airway" column values to 0.5 steps
    rounded_df = df_reduce[perctange] * 2
    rounded_df = rounded_df.round(0) / 2
    df_reduce[perctange] = rounded_df

    # Create new rows with normalized data in steps of 0.5%
    for percent_index in np.arange(0, 100.5, 0.5):
        for factor in np.arange(0, 4, 0.5):
            # Find rows within the specified range
            temp_df = df_reduce[
                (df_reduce[perctange] >= percent_index - factor) &
                (df_reduce[perctange] <= percent_index + factor)
            ].copy()

            if not temp_df.empty:
                break

        if temp_df.empty:
            new_row = [percent_index, 0, 0, 0]
        else:
            new_row = [
                percent_index,
                round(temp_df["area"].mean(), 1),
                round(temp_df["avg_width"].mean(), 1),
                round(temp_df["max_width"].mean(), 1),
            ]
        normalize_df.loc[len(normalize_df.index)] = new_row
    if perctange == "percentage of pa-ch":
        return normalize_df
    else:
        return normalize_df, pa_percent, ch_percent



def read(excel_path, mode="1connected", inferior=False, percent_method="percentage of pa-ch" ):
    """
    Read data from an Excel file and return normalized data frames based on the specified mode.

    :param excel_path: Path to the Excel file.
    :param mode: Mode of operation. Valid values are "1connected", "2connected", and "all". Defaults to "1connected".
    :param inferior: Flag indicating whether inferior data should be considered. Defaults to False.
    :return: List of normalized data frames.

    The function reads data from an Excel file and performs normalization based on the specified mode. It supports different modes of operation:
    - "1connected": Returns normalized data frames for the "cs r data" and "cs l data" sheets.
    - "2connected": Returns a single normalized data frame by combining the "cs r data" and "cs l data" sheets.
    - "all": Returns a single normalized data frame by combining all available sheets.

    If the `inferior` flag is True, it considers the inferior data by prepending "inferior " to the sheet names.

    The normalization process involves reading the data frames from the Excel file, applying normalization operations using the `normalize_dataframe` function,
    and storing the normalized data frames in a list. The resulting list of data frames is then returned as the output.
    """
    sheet_names = []

    if mode == "2connected" or mode == "1connected":
        sheet_names = ["cs r data", "cs l data"]
    else:
        sheet_names = ["cs r data", "cs l data", "notcc cs r data", "notcc cs l data"]

    data_frames = []

    for sheet_name in sheet_names:

        if inferior:
            sheet_name = "inferior " + sheet_name

        df_temp = pd.read_excel(excel_path, sheet_name=sheet_name)
        if percent_method == "percentage of pa-ch":
            df_temp = normalize_dataframe(df_temp, percent_method)
        else:
            df_temp, pa_percent, ch_percent = normalize_dataframe(df_temp, percent_method)
        data_frames.append(df_temp)

    if mode == "2connected":
        data_frames[0]["area"] += data_frames[1]["area"]
        # data_frames[0]["area"] /= 2
        data_frames = [data_frames[0]]

    elif mode == "all":
        data_frames[0]["area"] += data_frames[1]["area"] + data_frames[2]["area"] + data_frames[3]["area"]
        # data_frames[0]["area"] /= 2
        data_frames = [data_frames[0]]

    elif mode == "1":
        # If cc is not in the same length
        data_frames[0]["area"] += data_frames[2]["area"]
        data_frames[1]["area"] += data_frames[3]["area"]
        data_frames = [data_frames[0], data_frames[1]]
    if percent_method == "percentage of pa-ch":
        return data_frames
    else:
        return data_frames, pa_percent, ch_percent



def standardize_cases(df_list, percent_col_name="percentage of pa-ch"):
    """
    Standardize a list of data frames of the nasal airway cross-sectional area
    by calculating the average and standard deviation.

    :param df_list: List of data frames to be standardized.
    :param percent_col_name: Name of the percentage column in the data frames. Default is "percentage of pa-ch".
    :return: List containing the average and standard deviation data frames.

    The function takes a list of data frames and performs standardization by calculating the average and standard deviation.
    It returns a list containing the average and standard deviation data frames.

    The standardization process involves calculating the average data frame by summing up the data frames in the list
    and dividing by the number of cases. Then, the standard deviation data frame is calculated by subtracting each data frame
    from the average, squaring the differences, summing them up, dividing by the number of cases, and taking the square root.

    Finally, the average and standard deviation data frames are rounded to one decimal place, and the percentage column
    from the average data frame is copied to the standard deviation data frame for consistency.

    Note: If the input list of data frames is empty, the function returns 0 for the average and standard deviation.

    Example usage:
    avg_df, std_df = standardize_cases([df1, df2, df3])
    """
    if len(df_list) == 0:
        return 0, 0

    avg_df = df_list[0].copy()
    number_of_cases = len(df_list)

    for index in range(1, number_of_cases):
        avg_df += df_list[index]

    avg_df /= number_of_cases

    std_df = (avg_df - df_list[0]) ** 2

    for index in range(1, number_of_cases):
        std_df += (avg_df - df_list[index]) ** 2

    std_df /= number_of_cases
    std_df **= 0.5

    avg_df = avg_df.round(1)
    std_df = std_df.round(1)
    std_df[percent_col_name] = avg_df[percent_col_name]

    return avg_df, std_df

def plot_graph(x, y, std, leg_label, avg_color, std_color):
    """
    Plot a graph of CSA along the cavity with STD region.

    :param x: Values for the x-axis.
    :param y: Values for the y-axis.
    :param std: Standard deviation values for error regions.
    :param leg_label: Label for the legend.
    :param avg_color: Color for the line plot.
    :param std_color: Color for the shaded error regions.
    :return: None
    """
    # Plot a line graph with shaded error regions.
    plt.plot(x, y, color=avg_color, label=leg_label)
    plt.fill_between(x, y - std, y + std, color=std_color, alpha=0.2)

def save_plot(y_name, mode, plot_path = "C:\\Users\\owner\\Desktop\\cases\\plots\\compare\\standardization"):
    """
    Save the current plot as a JPEG image.

    :param y_name: Name of the y-axis or plot.
    :param mode: Mode or variant of the plot.
    :return: None
    """
    if plot_path == "":
        plot_path = input("Enter the path for save the plot")
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    os.chdir(plot_path)
    plt.savefig(f"{y_name}({mode}).jpg")

#compare graph of two case
def plot_compare(df_list, data=None, mode="all"):
    num_of_graph = len(df_list)
    x = np.arange(0, 100.5, 0.5)
    for y_name in ["area", "avg_width", "max_width"]:
        if y_name == "area":
            suptitle = "Compare Normal Via Obstructs Cases Average Cross-Sectional Area"
            ylabel = "Area" + r"$(mm^2)$"
        elif y_name == "avg_width":
            suptitle = "Compare Normal Via Obstructs Cases Average Cross-Sectional Avg Width"
            ylabel = "Avg Width (mm)"
        elif y_name == "max_width":
            suptitle = "Compare Normal Via Obstructs Cases Average Cross-Sectional Max Width"
            ylabel = "Max Width (mm)"

        for i in range(num_of_graph):
            if i == 0:
                leg_label = "all meatus - normal cases"
                std_color = "springgreen"
                avg_color = "mediumaquamarine"
            elif i == 1:
                leg_label = "all meatus - obstructed cases"
                std_color = "pink"
                avg_color = "sandybrown"
            elif i == 2:
                leg_label = "inferior meatus - normal cases"
                std_color = "green"
                avg_color = "darkgreen"
            elif i == 3:
                leg_label = "inferior meatus - obstructed cases"
                std_color = "red"
                avg_color = "darkred"

            y = df_list[i][0][y_name]
            std = df_list[i][1][y_name]

            # Plot a line graph with shaded error regions.
            plt.plot(x, y, color=avg_color, label=leg_label)
            plt.fill_between(x, y - std, y + std, color=std_color, alpha=0.2)

        plt.xlim(left=0, right=100)
        plt.xlabel("Percent(%)")
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        plt.suptitle(suptitle)

        save_plot(y_name, mode)


def plot_compare_inferior(df_list, data=None, mode="all"):

    num_of_graph = len(df_list)

    x = np.arange(0, 100.5, 0.5)
    for y_name in ["area", "avg_width", "max_width"]:
        if y_name == "area":
            suptitle = "Compare Normal Via Obstructs Cases Average Cross-Sectional Area"
            ylabel = "Area" + r"$(mm^2)$"
        elif y_name == "avg_width":
            suptitle = "Compare Normal Via Obstructs Cases Average Cross-Sectional Avg Width"
            ylabel = "Avg Width (mm)"
        elif y_name == "max_width":
            suptitle = "Compare Normal Via Obstructs Cases Average Cross-Sectional Max Width"
            ylabel = "Max Width (mm)"

        for i in range(num_of_graph):
            if i == 0:
                leg_label = "inferior meatus - normal cases"
                std_color = "green"
                avg_color = "darkgreen"
            elif i == 1:
                leg_label = "inferior meatus - obstructed cases"
                std_color = "red"
                avg_color = "darkred"

            y = df_list[i][0][y_name]
            std = df_list[i][1][y_name]

            # Plot a line graph with shaded error regions.
            plt.plot(x, y, color=avg_color, label=leg_label)
            plt.fill_between(x, y - std, y + std, color=std_color, alpha=0.2)

        plt.xlim(left=0, right=100)
        plt.xlabel("Percent(%)")
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        plt.suptitle(suptitle)

        save_plot("inferior_" + y_name, mode)



def plot_compare_obstruct(df_list, data = None, mode = "all"):
    num_of_graph = len(df_list)
    x = np.arange(0,100.5, 0.5)
    for y_name in ["area"]:
        if y_name == "area":
            suptitle = "Compare Average Cross-Sectional Area\n" \
                       " Surgical Intervention cases Via \n  Obstruct cases without Surgical Intervention\n\n"
            ylabel = "Area" + r"$(mm^2)$"


        for i in range(num_of_graph):
            if i == 0:
                leg_label = "all meatus - without surgery"
                std_color = "springgreen"
                avg_color = "mediumaquamarine"
            elif i == 1:
                leg_label = "all meatus - surgical intervention"
                std_color = "pink"
                avg_color = "sandybrown"
            elif i == 2:
                leg_label = "inferior meatus - without surgery"
                std_color = "green"
                avg_color = "darkgreen"
            elif i == 3:
                leg_label = "inferior meatus - surgical intervention"
                std_color = "red"
                avg_color = "darkred"




            y = df_list[i][0][y_name]
            std = df_list[i][1][y_name]

            plt.plot(x, y, color= avg_color , label=leg_label)
            plt.fill_between(x, y - std, y + std, alpha=0.2)
            # plt.plot(x, y + std, linestyle='loosely dashdotted', color=std_color)
            # plt.plot(x, y - std, linestyle='loosely dashdotted', color=std_color)

        plt.xlim(left=0, right=100)
        plt.xlabel("Percent(%)")
        plt.ylabel(ylabel)
        plt.legend(loc='best',)
        plt.suptitle(suptitle)

        save_plot("obstruct_" + y_name, mode)

def plot_compare_inferior_obstruct(df_list, data=None, mode="all"):
    num_of_graph = len(df_list)
    x = np.arange(0, 100.5, 0.5)
    for y_name in ["area"]:
        if y_name == "area":
            suptitle = "Compare Average Cross-Sectional Area\n" \
                       " Surgical Intervention cases Via \n  Obstruct cases without Surgical Intervention\n\n"
            ylabel = "Area" + r"$(mm^2)$"


        for i in range(num_of_graph):
            if i == 0:
                leg_label = "inferior meatus - without surgery"
                std_color = "green"
                avg_color = "darkgreen"
            elif i == 1:
                leg_label = "inferior meatus - surgical intervention"
                std_color = "red"
                avg_color = "darkred"

            y = df_list[i][0][y_name]
            std = df_list[i][1][y_name]

            plt.plot(x, y, color=avg_color, label=leg_label)
            plt.fill_between(x, y - std, y + std, color=std_color, alpha=0.2)
                # plt.plot(x, y + std, linestyle='loosely dashdotted', color=std_color)
                # plt.plot(x, y - std, linestyle='loosely dashdotted', color=std_color)

        plt.xlim(left=0, right=100)
        plt.xlabel("Percent(%)")
        plt.ylabel(ylabel)
        plt.legend(loc='best',)
        plt.suptitle(suptitle)

        save_plot("obstruct_inferior_" + y_name, mode)


# plot a compare graph for the 3 groups.
# can be one data case from each group, or an averaging of some cases


def plot_compare_3(df_list, data = None, std_flag = True ,mode = "all", percent_mode="pa-ch",pa_percent=0,ch_percent=0, units = 'mm'):
    num_of_graph = len(df_list)
    x = np.arange(0,100.5, 0.5)/100
    f, ax = plt.subplots()
    for y_name in ["area"]:
        if y_name == "area":
            suptitle = "Compare Average Cross-Sectional Area\n"\
                       "Normal cases &\n"\
                       "Obstruct cases without Surgical Intervention &\n Surgical Intervention cases\n"
            if units == "mm":
                ylabel = "Average Cross-Sectional Area" + r"$(mm^2)$"
            else:
               ylabel = "Average Cross-Sectional Area" + r"$(cm^2)$"
        max_val = 0
        for i in range(num_of_graph):
            if i == 1:
                leg_label = "Moderate (CNPAS)"
                std_color = "orange"
                avg_color = "orange"
                marker = '+'
            elif i == 2:
                leg_label = "Severe (CNPAS)"
                std_color = "red"
                avg_color = "red"
                marker = 'x'
            elif i == 0:
                leg_label = "Normal"
                std_color = "green"
                avg_color = "darkgreen"
                marker = 'o'

            y = df_list[i][0][y_name]
            if units == "cm":
                y /= 100
            if np.max(y) > max_val:
                max_val = np.max(y)
            plt.plot(x, y, color= avg_color , label=leg_label)

            if std_flag is True:
                std = df_list[i][1][y_name]
                if units == "cm":
                    std /= 100
                plt.fill_between(x, y - std, y + std, color=std_color, alpha=0.2)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        plt.xlim(left=0, right=1)
        if units == "mm":
            plt.ylim(bottom=0, top=max_val + 20)
        else:
            plt.ylim(bottom=0, top=max_val + 0.2)
        if percent_mode =="pa-ch":
            plt.axvline(x=10, color="blue", linestyle="--") # for poster
            plt.annotate("",(0,0.7), (0.1,0.7), xycoords ='axes fraction',  arrowprops=dict(arrowstyle="<->", color='blue'))
            plt.text( x=0.05,y=0.7, s="PA\n region",  transform=ax.transAxes, color='blue', ha='center', va='center' )
            plt.xlabel("Normalized Distance Between The PA-Choanae(%)", fontsize=12)
        else:
            plt.axvline(x=pa_percent / 100, color="blue", linestyle="--")
            plt.axvline(x=ch_percent / 100, color="blue", linestyle="--")
            plt.text(x=pa_percent / 100, y=0.6, s="mean PA", transform=ax.transAxes, color='blue', ha='right',rotation='vertical', va='center')
            plt.text(x=ch_percent / 100, y=0.6, s="mean choanae", transform=ax.transAxes, color='blue', ha='right', va='center',rotation='vertical')
            plt.xlabel("Normalized Distance of nasal airway", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(loc='upper right',)

        save_plot("3_" + y_name + "_" + percent_mode , mode)


def plot_compare_inferior_3(df_list, data=None, std_flag = True, mode="all"):
    num_of_graph = len(df_list)
    f, ax = plt.subplots()
    print(num_of_graph)
    x = np.arange(0, 100.5, 0.5)
    for y_name in ["area"]:
        if y_name == "area":
            suptitle = "Compare Average Cross-Sectional Area in The Inferior Meatus\n" \
                       "Normal cases &\n"\
                       "Obstruct cases without Surgical Intervention &\n Surgical Intervention cases\n"
            ylabel = "Area" + r"$(mm^2)$"

        max_val = 0
        for i in range(num_of_graph):
            if i == 1:
                leg_label = "PAS w/o surgery"
                std_color = "yellow"
                avg_color = "yellow"
            elif i == 2:
                leg_label = "PAS + surgery"
                std_color = "red"
                avg_color = "darkred"
            elif i == 0:
                leg_label = "normal airway"
                std_color = "green"
                avg_color = "darkgreen"

            y = df_list[i][0][y_name]
            if np.max(y) > max_val:
                max_val = np.max(y)
            plt.plot(x, y, color=avg_color, label=leg_label)
            if std_flag is True:
                std = df_list[i][1][y_name]
                plt.fill_between(x, y - std, y + std, color=std_color, alpha=0.2)


        plt.xlim(left=0, right=100)
        plt.ylim(bottom=0, top=max_val + 20)
        plt.xlabel("Percent(%)")
        plt.ylabel(ylabel)
        plt.axvline(x=10, color="blue", linestyle="--")  # for poster
        plt.annotate("", (0, 0.7), (0.1, 0.7), xycoords='axes fraction',
                     arrowprops=dict(arrowstyle="<->", color='blue'))
        plt.text(x=0.05, y=0.7, s="PA\n region", transform=ax.transAxes, color='blue', ha='center', va='center')
        #plt.legend(loc='best',)
        plt.suptitle(suptitle)
        plt.tight_layout()
        save_plot("inferior3" + y_name, mode)

