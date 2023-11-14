import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import usefull_function as uf
import os



# visulaze the cross-sectional area with important data along each nasal cavity


def plot_cs_area(df):
    df.plot(x='coronal index', y='area', title = 'cs_area', xlabel= "distance(coronol index)", ylabel = "area(mm^2)" )

def plot_avg_width(df):
    df.plot(x='coronal index', y='avg_width', title = 'cs_avg_width', xlabel= "distance(coronol index)", ylabel = "width(mm)")

def plot_max_width(df):
    df.plot(x='coronal index', y='max_width',  title = 'cs_max_width', xlabel= "distance(coronol index)", ylabel = "width(mm)")


def rearrange_data_to_plot(df, stat ='all airway'):
    # arange volume data for writing on the plot
    cc_volume_data = df[0]
    not_cc_volume_data = df[1]


    if stat == 'all airway':
        gdata = [[cc_volume_data["cc total volume"].to_numpy(), not_cc_volume_data["not_cc total volume"].to_numpy(),
              cc_volume_data["nasopharynx volume"].to_numpy()]]
        dataex = [[cc_volume_data["right nostril cc volume"].to_numpy(),
               not_cc_volume_data["not_cc right nostril volume"].to_numpy()],
              [cc_volume_data["left nostril cc volume"].to_numpy(),
               not_cc_volume_data["not_cc left nostril volume"].to_numpy()]]
        datain = [[cc_volume_data["right cc volume"].to_numpy(),
               not_cc_volume_data["not_cc right airway volume"].to_numpy()],
              [cc_volume_data["left cc volume"].to_numpy(),
               not_cc_volume_data["not_cc left airway volume"].to_numpy()]]
        data = [gdata, dataex, datain]
        data_names = [["Volume", "Disconnected\nVolume", "Nasopharynx\nVolume"], ["Volume", "Disconnected\nVolume"], ["Volume", "Disconnected\nVolume"]]

    elif stat == 'inferior':
        r_cc_vol = cc_volume_data["vol r"]
        r_not_cc_vol = not_cc_volume_data["vol r"]
        l_cc_vol = cc_volume_data["vol l"]
        l_not_cc_vol = not_cc_volume_data["vol l"]
        data = [[r_cc_vol.iloc[0], r_not_cc_vol.iloc[0]],[l_cc_vol.iloc[0], l_not_cc_vol.iloc[0]]]
        data_names = [["Volume", "Disconnected Volume"]]

    return data, data_names

def index_to_distance(df, index):
    row = df[df["coronal index"] == index]
    distance = row["x"].iloc[0]
    return distance

def plot(df, pa_index, ch_index, save_path, stat="x"):
    plotpath = save_path + "\\CSA(all)"
    if not os.path.exists(plotpath):
        os.mkdir(plotpath)
    os.chdir(plotpath)
    if stat == "x":
        pa_x = index_to_distance(df[2], pa_index)
        ch_x = index_to_distance(df[2], ch_index)
        x_name = "x"
        x_title = "Distance From Nostril(mm)"
    else:
        pa_x = pa_index
        ch_x = ch_index
        x_name = "coronal index"
        x_title = "Distance(Coronal Slice Index)"

    x_r = df[2][x_name].to_numpy()
    x_l = df[3][x_name].to_numpy()
    x_r_notcc = df[4][x_name].to_numpy()
    x_l_notcc = df[5][x_name].to_numpy()


    for y_name in ["area","avg_width","max_width"]:
        if y_name == "area":
            suptitle = "Automated Measuring of The Cross-Sectional Area\n Along Nasal Cavity"
            ylabel = "Area"+ r"$(mm^2)$"
            data, data_names = rearrange_data_to_plot(df[:2])
            # change the genral data to text format
            g_data = data[0][0]
            g_names = data_names[0]
            g_txt = "Genral Information:\n"
            for i in range(len(g_data)):
                g_txt += g_names[i]+ ": " + str(g_data[i][0])+ r"$(mm^3)$"+ "\n\n"
        elif y_name == "avg_width":
            suptitle = "CC Cross Sectional Average Width"
            ylabel = "Avg Width (mm)"
        elif y_name == "max_width":
            suptitle = "CC Cross Sectional Max Width"
            ylabel = "Max Width (mm)"

    # create x,y from the data frame. for two plot. left side, and right side

        y_r = df[2][y_name]
        y_l = df[3][y_name]
        y_r_notcc = df[4][y_name]
        y_l_notcc = df[5][y_name]

        max_y = int(max(np.amax(y_r),np.amax(y_l))) + 1
        index = 1

        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,7)) # 2 horizontal plot
        fig.suptitle(suptitle) # add super title
        ax1.plot(x_l, y_l, marker = '+',label= "Connect")
        ax1.plot(x_l_notcc,y_l_notcc, marker = 'x', color = "red", label="Disconnect")
        ax1.legend(loc='center right')
        ax1.set_title("Left Cavity")

        ax2.plot(x_r, y_r, marker = '+',label="Connect")
        ax2.plot(x_r_notcc, y_r_notcc, marker = 'x', color="red", label="Disconnect")
        ax2.set_title("Right Cavity")
        ax2.legend(loc='center right')


        # set labels, vertical lines + annotation + y_lim +and ticks + information
        for ax in [ax1, ax2]:
            ax.set_xlabel(x_title, weight='bold')
            ax.set_ylabel(ylabel, weight='bold')
            ax.set_xlim(left=0)
            ax.vlines(x=pa_x, ymin=0, ymax=max_y-25, color="orange" , linestyles= "dashed")
            ax.annotate("PA", (pa_x,max_y-25), verticalalignment="center", horizontalalignment= "center" )
            ax.vlines(x=ch_x, ymin=0, ymax=max_y-25, color="orange", linestyles="dashed") #, label="choana"
            ax.annotate("Choana", (ch_x, max_y-25), verticalalignment="center", horizontalalignment="center" )
            ax.set_ylim(bottom=0)


            # change information to txt.
            txt1 = "External\n Nose\n\n"
            txt2 = "Internal\n Nasal Airway\n\n"

        # data1 is external nose, data2 is internal airway
            add = max_y // 10
            y_txt = max_y + add / 4
            if y_name == "area":
                ax.set_yticks(np.arange(0, max_y + add, 25))

                data1_names = data_names[1]
                data1 = data[1][index]
                data2 = data[2][index]
                data2_names = data_names[2]
                index -= 1
                for i in range(len(data1)):
                    txt1 += data1_names[i]+ ":\n" + str(data1[i][0])+ r"$(mm^3)$" + "\n\n"
                for i in range(len(data2)):
                    txt2 += data2_names[i] + ":\n" + str(data2[i][0]) + r"$(mm^3)$" + "\n\n"
            else:
                ax.set_yticks(np.arange(0, max_y + add, 2))

            ax.text(pa_x//2, y_txt, txt1, color="darkblue", size='small', verticalalignment="top", horizontalalignment="center" )
            ax.text((pa_x+ch_x)//2  , y_txt , txt2, color="darkblue", size= 'small',verticalalignment="top", horizontalalignment="center" )
            ax.text((ch_x) + 10 , y_txt, "Nasopharynx", color="darkblue", size= 'small',verticalalignment="top")

        if y_name == "area":
            plt.tight_layout(pad = 1, w_pad=10)  #  add space between plots
            plt.figtext(0.45,0.65,g_txt, size='small')
        else:
            plt.tight_layout()



        plt.savefig(y_name + ".jpg")
        plt.close("all")



def plot_inferior(df, pa_index, ch_index,not_cc=True, stat ="x", save_path=""):
    plotpath = save_path + "\\CSA(inferior)"
    if not os.path.exists(plotpath):
        os.mkdir(plotpath)
    os.chdir(plotpath)
    if stat == "x":
        pa_x = index_to_distance(df[2], pa_index)
        ch_x = index_to_distance(df[2], ch_index)
        x_name = "x"
        x_title = "Distance From Nostril(mm)"
    else:
        x_name = "coronal index"
        pa_x = pa_index
        ch_x = ch_index
        x_title = "Distance(Coronal Slice Index)"

    x_r = df[2][x_name]
    x_l = df[3][x_name]
    x_r_notcc = df[4][x_name]
    x_l_notcc = df[5][x_name]

    for y_name in ["area", "avg_width", "max_width"]:
        if y_name == "area":
            suptitle = " Automated Measuring of The Cross-Sectional Area\n of The Inferior Meatus Between The Pyriform Aperture And The Choana"
            ylabel = "Area" + r"$(mm^2)$"
            data, data_names = rearrange_data_to_plot(df[:2], stat='inferior')


            # change the genral data to text format
        elif y_name == "avg_width":
            suptitle = "Inferior Meatus Cross Sectional Average Width"
            ylabel = "Avg Width (mm)"
        elif y_name == "max_width":
            suptitle = "Inferior Meatus Cross Sectional Max Width"
            ylabel = "Max Width (mm)"

        # create x,y from the data frame. for two plot. left side, and right side

        y_r = df[2][y_name]
        y_l = df[3][y_name]
        y_r_notcc = df[4][y_name]
        y_l_notcc = df[5][y_name]


        max_y = int(max(np.amax(y_r), np.amax(y_l))) + 1




        index = 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))  # 2 horizontal plot
        fig.suptitle(suptitle, size=16)  # add super title
        # left = ax1
        ax1.plot(x_l, y_l, marker='+', label="connect")
        y_min = np.amin(y_l)
        # add to title and anonate norrwed region
        l_title = ["Left Cavity\n"]
        colors = ["black"]
        if y_min < 5:
            l_title.append("Narrowed Region Was Found")
            colors.append("red")
            x_min = x_l[np.argmin(y_l)]
            ax1.annotate('Narrowed region', xy=(x_min, y_min), xycoords='data',
                         xytext=(x_min, y_min + 10), textcoords='data', color='red',
                         arrowprops=dict(arrowstyle="->", shrinkA=1, shrinkB=1, color='r'),
                         horizontalalignment='center', verticalalignment='top', )
        else:
            l_title.append('Normal Cavity')
            colors.append("green")

        rainbow_title(l_title, colors,ax=ax1)



        #ax1.set_title("Left Cavity" + l_title, color=color)

        # right = ax2"
        ax2.plot(x_r, y_r, marker='+', label="Connect")
        y_min = np.amin(y_r)
        r_title = ["Right Cavity\n"]
        colors = ["black"]
        if  y_min < 5:
            r_title.append("Narrowed Region Was Found")
            colors.append("red")
            x_min =  x_r[np.argmin(y_r)]
            ax2.annotate('Narrowed region', xy=(x_min, y_min), xycoords='data',
                        xytext=(x_min, y_min + 10), textcoords='data', color='red',
                        arrowprops=dict(arrowstyle="->", shrinkA=0.5, shrinkB=0.5,  color='r'),
                        horizontalalignment='center', verticalalignment='top',)
        else:
            r_title.append("Normal Cavity")
            colors.append("green")
        rainbow_title(r_title, colors, ax=ax2)

        # draw graph of the disconnected part of the airway
        if not_cc is True:
            ax1.plot(x_l_notcc, y_l_notcc, color="red",marker='x', label="Disconnect")
            ax1.legend(loc='center left')
            ax2.plot(x_r_notcc, y_r_notcc, color="red",marker='x', label="Disconnect")
            ax2.legend(loc='center left')

        # set labels, vertical lines + annotation + y_lim +and ticks + information
        for ax in [ax1, ax2]:
            ax.set_xlabel(x_title, weight='bold')
            ax.set_ylabel(ylabel, weight='bold')
            # data1 is external nose, data2 is internal airway
            if y_name == "area":
                add =  max_y / 10
                if add < 10:
                    add = 10
                ax.set_yticks(np.arange(0, max_y + add, 5))
                y_txt = max_y + add//3

                data_name = data_names[0]
                data_side = data[index]

                index -= 1
                txt = ''
                for i in range(len(data_side)):
                    txt += data_name[i] + ": " + str(data_side[i]) + r"$(mm^3)$" + "\n"
                ax.text((pa_x + ch_x) // 2, y_txt, txt, color="darkblue", size='medium',
                        verticalalignment='top', horizontalalignment="center")


            else:
                ax.set_yticks(np.arange(0, max_y + max_y/10, 0.5))
            plt.tight_layout()



        plt.savefig(y_name + ".jpg")
        plt.close("all")


def rainbow_title(strings, colors, ax=None,sizes = [12,10],  **kw):
    x_pos = [0.5, 0.5]
    y_pos = [1.02,1.01]
    if ax is None:
        ax = plt.gca()

    for x,y, s, c,size in zip(x_pos,y_pos, strings, colors, sizes):
        text = ax.text(x, y, s + " ", color=c, horizontalalignment='center',transform = ax.transAxes,size=size, **kw)



def plot_compare3(cases_df, cases_type, pa_indexs, ch_indexs, x_name, case_label, cases_path):

    y_name = "area"
    fig, (ax) = plt.subplots(1, 1, figsize=(12, 7))  # 2 horizontal plot
    suptitle = "Cross Sectional Area of Nasal Airway\n" +"Compare of 3 Cases"
    ylabel = "Area" + r"$(mm^2)$"
                #data, data_names = rearrange_data_to_plot(df[:2])
                # change the genral data to text format
                #g_data = data[0][0]
                #g_names = data_names[0]
                #g_txt = "Case" +  "Information\n\n"
                #for i in range(len(g_data)):
                #    g_txt += g_names[i] + ": " + str(g_data[i][0]) + r"$(mm^3)$" + "\n"

    fig.suptitle(suptitle)  # add super title

    # create x,y from the data frame. for two plot. left side, and right side
    case_index = 0
    for df,case_type in zip(cases_df,cases_type):
        x = df[2][x_name]
        y = df[2][y_name] + df[3][y_name] + df[4][y_name] + df[5][y_name]
        nasopharynx_index = df[2][df[2]["coronal index"] == ch_indexs[case_index]+1].index[0]
        y[nasopharynx_index:] /= 2


        max_y = np.amax(y) + 1
        if case_type == 1: # normal, airway
            plotcolor = "green"
            plot_label = "normal airway (" + case_label[case_index] + ")"

        elif case_type == 2: # CNPAS without surgery
            plotcolor = "yellow"
            plot_label = "CNPAS without surgery (" + case_label[case_index] + ")"

        elif case_type == 3: #CNPAS with surgery
            plotcolor = "red"
            plot_label = "CNPAS with surgery (" + case_label[case_index] + ")"

        ax.plot(x, y, color=plotcolor, label=plot_label)

        if x_name == "x":
            pa_index = index_to_distance(df[2],pa_indexs[case_index])
            ch_index = index_to_distance(df[2],ch_indexs[case_index])
        else:
            pa_index = pa_indexs[case_index]
            ch_index = ch_indexs[case_index]


        # set labels, vertical lines + annotation + y_lim +and ticks + information
        ax.vlines(x=pa_index, ymin=0, ymax=max_y, color=plotcolor, linestyles="dashed")
        ax.annotate("PA", (pa_index, max_y), verticalalignment="center", horizontalalignment="left",
                           rotation=90)
        ax.vlines(x=ch_index, ymin=0, ymax=max_y, color=plotcolor, linestyles="dashed")
        ax.annotate("Choana", (ch_index, max_y), verticalalignment="center", horizontalalignment="left",
                            rotation=90)
        ax.set(xlabel="Distance Fron Nostril(mm)", ylabel=ylabel)


                # change information to txt.
        if case_index == 0:
            txt1 = "External Nose"
            txt2 = "Airway"
            txt3 = "Nasopharynx"
            ax.set_yticks(np.arange(0, max_y + 35, 25))
            y_txt = max_y + 20
            ax.text(pa_index//2, y_txt, txt1, color="darkblue", size='small', verticalalignment="top",ha="center")
            ax.text((pa_index + ch_index) // 2 , y_txt, txt2, color="darkblue", size='small',
                            verticalalignment="top", ha="center")
            ax.text(((ch_index) + np.amax(x))//2 , y_txt, txt3, color="darkblue", size='small', verticalalignment="top",
                    ha="center")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)


        elif case_index == len(cases_df) - 1:
            ax.legend(loc='center right')
            plt.tight_layout()

        case_index += 1

    plotpath = "C:\\Users\\owner\\Desktop\\cases\\plots\\compare\\"
    if not os.path.exists(plotpath):
        os.mkdir(plotpath)
    os.chdir(plotpath)
    plt.savefig(cases_path + ".jpg")


def plot_compare3_inferior(cases_df, cases_type, x_name, case_label, cases_path):
    y_name = "area"
    fig, (ax) = plt.subplots(1, 1, figsize=(12, 7))  # 2 horizontal plot
    suptitle = "Cross Sectional Area of The Inferior Meatus\n" + "Compare of 3 Cases"
    ylabel = "Area" + r"$(mm^2)$"
    # data, data_names = rearrange_data_to_plot(df[:2])
    # change the genral data to text format
    # g_data = data[0][0]
    # g_names = data_names[0]
    # g_txt = "Case" +  "Information\n\n"
    # for i in range(len(g_data)):
    #    g_txt += g_names[i] + ": " + str(g_data[i][0]) + r"$(mm^3)$" + "\n"

    fig.suptitle(suptitle)  # add super title

    # create x,y from the data frame. for two plot. left side, and right side
    case_index = 0
    for df, case_type in zip(cases_df, cases_type):

        x = df[2][x_name]
        y = df[2][y_name] + df[3][y_name] + df[4][y_name] + df[5][y_name]


        max_y = np.amax(y) + 1
        if case_type == 1:  # normal, airway
            plotcolor = "green"
            plot_label = "normal airway (" + case_label[case_index] + ")"

        elif case_type == 2:  # CNPAS without surgery
            plotcolor = "yellow"
            plot_label = "CNPAS without surgery (" + case_label[case_index] + ")"

        elif case_type == 3:  # CNPAS with surgery
            plotcolor = "red"
            plot_label = "CNPAS with surgery (" + case_label[case_index] + ")"

        ax.plot(x, y, color=plotcolor, label=plot_label)



        # set labels, vertical lines + annotation + y_lim +and ticks + information

        ax.set(xlabel="Distance Fron Nostril(mm)", ylabel=ylabel)


        # change information to txt.
        if case_index == 0:
            ax.set_yticks(np.arange(0, max_y + 25, 25))
        elif case_index == len(cases_df) - 1:
            ax.legend(loc='center right')

            ax.set_ylim(bottom=0)
            plt.tight_layout()
        case_index += 1

    plotpath = "C:\\Users\\owner\\Desktop\\cases\\plots\\compare\\"
    if not os.path.exists(plotpath):
        os.mkdir(plotpath)
    os.chdir(plotpath)
    plt.savefig("inferior-" + cases_path + ".jpg")



# need to change the indexes to distance
def plot_compare(cases_df, pa_indexs, ch_indexs, case_label, cases_path):
    x_name = "coronal index"
    for y_name in ["area", "avg_width", "max_width"]:
        case_index = 0
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))  # 2 horizontal plot
        if y_name == "area":
            suptitle = "CC Cross Sectional Area\n" + "compare " + cases_path
            ylabel = "Area" + r"$(mm^2)$"
                        # data, data_names = rearrange_data_to_plot(df[:2])
                        # change the genral data to text format
                        # g_data = data[0][0]
                        # g_names = data_names[0]
                        # g_txt = "Case" +  "Information\n\n"
                        # for i in range(len(g_data)):
                        #    g_txt += g_names[i] + ": " + str(g_data[i][0]) + r"$(mm^3)$" + "\n"
        elif y_name == "avg_width":
            suptitle = "CC Cross Sectional Average Width\n" + "compare " + cases_path
            ylabel = "Avg Width (mm)"
        elif y_name == "max_width":
            suptitle = "CC Cross Sectional Max Width\n" + "compare " + cases_path
            ylabel = "Max Width (mm)"
        fig.suptitle(suptitle)  # add super title
        ax1.set_title("Left")
        ax2.set_title("Right")

                    # create x,y from the data frame. for two plot. left side, and right side

        for df in cases_df:
            x_r = df[2][x_name]
            y_r = df[2][y_name]

            x_l = df[3][x_name]
            y_l = df[3][y_name]

            max_y = int(max(np.amax(y_r), np.amax(y_l))) + 1
            if case_index == 1:
                plotcolor = "green"
                plotmarker = "o"
            else:
                plotcolor = "red"
                plotmarker = "x"

            ax1.plot(x_l, y_l, marker=plotmarker, color=plotcolor, label=case_label[case_index])
            ax2.plot(x_r, y_r, marker=plotmarker, color=plotcolor, label=case_label[case_index])
            pa_index = pa_indexs[case_index]
            ch_index = ch_indexs[case_index]
            if case_index == 1:
                ax1.legend(loc='center left')
                ax2.legend(loc='center left')

            # set labels, vertical lines + annotation + y_lim +and ticks + information
            for ax in [ax1, ax2]:
                ax.vlines(x=pa_index, ymin=0, ymax=max_y, color=plotcolor, linestyles="dashed")
                ax.annotate("PA", (pa_index, max_y), verticalalignment="center", horizontalalignment="left", rotation =90)
                ax.vlines(x=ch_index, ymin=0, ymax=max_y, color=plotcolor, linestyles="dashed")
                ax.annotate("Choana", (ch_index, max_y), verticalalignment="center", horizontalalignment="left", rotation =90)
                ax.set(xlabel="Distance(coronal slice ind)", ylabel=ylabel)
                ax.set_ylim(bottom=0)

            # change information to txt.
                if case_index == 1:
                    txt1 = "External Nose"
                    txt2 = "Airway"
                    txt3 = "Nasopharynx"
            # region_index = 0
            # data1 is external nose, data2 is internal airway
                    if y_name == "area":
                        ax.set_yticks(np.arange(0, max_y + 75, 25))
                        y_txt = max_y + 45
                    else:
                        ax.set_yticks(np.arange(0, max_y + 5, 2))
                        y_txt = max_y + 2.5

                    ax.text(5, y_txt, txt1, color="darkblue", size='x-small', verticalalignment="top")
                    ax.text((pa_index + ch_index) // 2 - 10, y_txt, txt2, color="darkblue", size='x-small',
                    verticalalignment="top")
                    ax.text((ch_index) + 10, y_txt, txt3 , color="darkblue", size='x-small', verticalalignment="top")

                plt.tight_layout()

            case_index += 1

        plotpath = "C:\\Users\\owner\\Desktop\\cases\\plots\\compare\\"
        if not os.path.exists(plotpath):
            os.mkdir(plotpath)
        os.chdir(plotpath)
        plt.savefig(cases_path + " " + y_name + ".jpg")



