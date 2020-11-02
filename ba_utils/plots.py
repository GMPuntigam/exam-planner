'''
Plotting utilities
'''
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralCoclustering
from ba_utils import general
# define colors for plots
PRIMARY_COLOR = (0.382, 0.613, 0.789)
SECONDARY_COLOR = 'tab:red'
THIRD_COLOR = 'tab:green'

def student_exam_count(exam_matrix_binary):
    '''
    create a histogram plot, showing how many students chose how many exams

    arguments:
        exam_matrix_binary: a binary matrix
    '''
    nr_exams = exam_matrix_binary.sum(axis=1)
    fig, ax = plt.subplots(dpi=100)
    n, bins, patches = ax.hist(x = nr_exams, bins=np.arange(17)+0.5, color=PRIMARY_COLOR, rwidth=0.85)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.hist(x = nr_exams, bins=np.arange(18)+0.5, color = 'black', density=True, histtype='step', cumulative=1,
            label='Kumulierte Verteilung')
    ax2.set_ylabel('Kumulierte Verteilung')

    plt.title('Histogramm der Prüfungswahlsumme')
    ax.set_xlabel('Summe gewählte Prüfungen')
    ax.set_ylabel('Anzahl Studenten')
    plt.xticks(np.arange(1, 18, 1))

def exam_sizes(exam_matrix_binary):
    '''
    create a histogram plot, showing exam sizes

    arguments:
        exam_matrix_binary: a binary matrix
    '''
    fig, ax = plt.subplots(dpi=100)
    nr_attending_students = exam_matrix_binary.sum(axis=0)
    ax.hist(x=nr_attending_students, bins=20, color=PRIMARY_COLOR, rwidth=0.85)
    plt.title('Histogramm über Prüfungsgröße')
    ax.set_xlabel('Anzahl Studenten')
    ax.set_ylabel('Anzahl Prüfungen')

def binary_matrix(exam_matrix_binary):
    '''
    plot the binary matrix of students

    arguments:
        exam_matrix_binary: a binary matrix
    '''
    labels = exam_matrix_binary.columns
    binary = exam_matrix_binary.T > 0
    fig, ax =plt.subplots(figsize=(18, 16))
    im = ax.imshow(binary, aspect=15, cmap=plt.get_cmap('Blues'))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylabel('Prüfungsgruppen')
    ax.set_xlabel('Studenten')
    plt.title('Binäre Prüfungswahlmatrix')

def binary_matrix_clustered(exam_matrix_binary, n_clusters):
    '''
    plot the clustered binary matrix of students and automatically mark clusters with numbers

    arguments:
        exam_matrix_binary: a binary matrix
        n_clusters: number of desired clusters
    '''
    labels = exam_matrix_binary.columns
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
    model.fit(exam_matrix_binary+1)
    fit_data = exam_matrix_binary.values[np.argsort(model.row_labels_)]
    cluster_matrix = fit_data[:, np.argsort(model.column_labels_)]

    binary = cluster_matrix.T > 0
    fig, ax =plt.subplots(figsize=(18, 16))
    im = ax.imshow(binary, aspect=15, cmap=plt.get_cmap('Blues'))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels[np.argsort(model.column_labels_)])
    cluster_nr, rowsizes = np.unique(model.column_labels_, return_counts=True)
    cluster_nr, colsizes = np.unique(model.row_labels_, return_counts=True)
    for i in cluster_nr:
        if i > 0:
            #left marker
            ax.axvline(x=sum(colsizes[:i-1])-0.5, ymin=1-sum(rowsizes[:i-1])/sum(rowsizes), ymax=1-sum(rowsizes[:i])/sum(rowsizes),color='red')
            #right marker
            ax.axvline(x=sum(colsizes[:i])-0.5, ymin=1-sum(rowsizes[:i-1])/sum(rowsizes), ymax=1-sum(rowsizes[:i])/sum(rowsizes),color='red')
            #top marker
            ax.axhline(y=sum(rowsizes[:i-1])-0.5, xmin=sum(colsizes[:i-1])/sum(colsizes), xmax=sum(colsizes[:i])/sum(colsizes),color='red')
            #bottom marker
            ax.axhline(y=sum(rowsizes[:i])-0.5, xmin=sum(colsizes[:i-1])/sum(colsizes), xmax=sum(colsizes[:i])/sum(colsizes),color='red')
            ax.text(x=(sum(colsizes[:i-1]) + sum(colsizes[:i]))/2, 
            y = (sum(rowsizes[:i-1]) + sum(rowsizes[:i]))/2, s=i , fontsize=18,
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(boxstyle = 'circle', facecolor='white', alpha=1),
            color = 'red')
    #last left marker
    ax.axvline(x=sum(colsizes[:i])-0.5, ymin=1-sum(rowsizes[:i-1])/sum(rowsizes), ymax=0,color='red')
    #last top marker
    ax.axhline(y=sum(rowsizes[:i])-0.5, xmin=sum(colsizes[:i-1])/sum(colsizes), xmax=1,color='red')
    ax.text(x=(sum(colsizes[:i]) + sum(colsizes))/2, 
            y = (sum(rowsizes[:i]) + sum(rowsizes))/2, s=i+1 , fontsize=18,
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(boxstyle = 'circle', facecolor='white', alpha=1),
            color = 'red')
    ax.set_ylabel('Prüfungsgruppen')
    ax.set_xlabel('Studenten')
    plt.title('Geclusterte binäre Prüfungswahlmatrix')

def simulated_annealing_time_series(scores, temp_log, times_log):
    '''
    plot a simulated annealing time series chart

    arguments:
        scores: list of logged scores
        temp_log: list of logged temperatures
        times_log: list of measuered times
    '''
    scores = scores[scores != 0]
    temp_log = temp_log[temp_log != 0]
    times_log = times_log[times_log != 0]
    fig, ax1 = plt.subplots()
    color = PRIMARY_COLOR
    ax1.plot(times_log, scores, color=color)
    ax1.set_xlabel('Zeit (s)')
    ax1.set_ylabel('Score', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    color = 'tab:red'
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(times_log, temp_log, color=color)
    ax2.set_ylabel('Temp', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.set_title('Simulated Annealing zeitlicher Verlauf')

def histogram(df_score):
    '''
    plots a score histogram

    arguments:
        df_score: dataframe containing column ['Total'] with scores for each individual student 
    '''
    fig=plt.figure(figsize=(10,5.89))
    n, bins, patches = plt.hist(x=df_score['Total'], bins=np.arange(max(df_score['Total'].values)+1)-0.5, color=PRIMARY_COLOR, rwidth=0.85)
    plt.title('Bewertungshistogramm')
    plt.xlabel('Bewertung')
    plt.ylabel('Anzahl Studenten')
    plt.xticks(np.arange(0, max(df_score['Total'])+1, 1))
    #print(sum(df_score['Total']))
    mean = np.average(df_score['Total'].values)
    var = np.average((df_score['Total'].values - mean)**2)
    std_dev = np.sqrt(var)
    #plt.errorbar(x=mean, y=2, xerr=std_dev, linestyle='None', marker='o', color = SECONDARY_COLOR)
    plt.fill_between([mean-std_dev, mean+std_dev], 0, 1000, color=(1, 0, 0, 0.5), linewidth=0.5)
    plt.axvline(x=mean, ymin=0, ymax=1,color='red')
    plt.ylim((0, 1000))
    # Generate text to write.
    text1 = r"$\varnothing = {:.{p}f}$".format(mean, p=2)
    text2 = "$\sigma = {:.{p}f}$".format(std_dev, p=2)
    text = text1 + '\n' + text2
    plt.annotate(text, xy=(1, 1), xytext=(-15, -15), fontsize=12,
        xycoords='axes fraction', textcoords='offset points',
        bbox=dict(facecolor='white', alpha=0.8),
        horizontalalignment='right', verticalalignment='top')

def pointplot(df_score_full, df_score):
    '''
    Plots a score dotplot. Scores on X-Axis, Number of Chosen Exams on Y-Axis, dot sizes represent number of students with combination 

    arguments:
        df_score: dataframe containing column ['Total'] with scores for each individual student
        df_score_full: dataframe containing the complete student timetable
    '''
    fig=plt.figure(figsize=(10,5.89))
    nr_exams = [len(df_score_full[df_score_full['Student_ID'] == i]) for i in range(max(df_score_full['Student_ID'])+1)]
    df_plot = pd.DataFrame([[score, nr]  for score, nr in zip(df_score['Total'].values, nr_exams)])
    vals , counts = np.unique(df_plot, axis=0, return_counts=True)
    df_plot = pd.DataFrame([[i[0],i[1], counts]  for i, counts in zip(vals, counts)], columns=['Score', 'Nr_exams','Counts'] )
    scatter = plt.scatter(x = df_plot['Score'], y= df_plot['Nr_exams'], s=df_plot['Counts'], c = PRIMARY_COLOR)
    plt.xlabel('Bewertung')
    plt.ylabel('Anzahl gewählte Prüfungen')
    plt.xticks(np.arange(0, max(df_score['Total'])+1, 1))

    meanx = np.average(df_score['Total'].values)
    meany = np.average(nr_exams)
    plt.plot(meanx, meany, marker = "x", c = SECONDARY_COLOR)

    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    plt.legend([handles[i] for i in [0, 2, 4, 7]], [labels[i] for i in [0,2,4,7]], title="Anz. Stud.", labelspacing  = 1)

def written_over_time(df_exam, exam_matrix_binary):
    '''
    Plots a time series showing how many students write an exams on each day

    arguments:
        df_exam: exam timetable
    '''
    df_exam = df_exam.sort_values(['Week','Day','Slot'])
    written_exams = general.get_written_per_day(df_exam, exam_matrix_binary)
    fig, ax1 = plt.subplots()
    ax1.bar([i[0]+1 for i in written_exams], [i[1] for i in written_exams], color = PRIMARY_COLOR)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot([i[0]+1 for i in written_exams],[i[2] for i in written_exams]/written_exams[14][2],color = 'black')
    ax2.set_ylabel('Kumuliert [%]',color = 'black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax1.set_title('Studenten pro Tag')
    ax1.set_xlabel('Tag')
    ax1.set_ylabel('Anzahl Studenten')
    ax1.set_xticks(np.arange(1, 16, 1))

def createplot(examplan, folderpath=""):
    '''
    Creates Plots for an examplan for every day.

    arguments:
        examplan: exam timetable
        folderpath: path for plots to be saved

    adapted from http://masudakoji.github.io/2015/05/23/generate-timetable-using-matplotlib/en/
    '''


    rooms=['Room A','Room B', 'Room C', 'Room D', 'Room E']
    colors=['pink', 'lightgreen', 'lightblue', 'wheat', 'salmon']

    #get necessary rooms: 

    res = [str(i) + str(j) + str(k) for i, j, k in zip(examplan['Week'], examplan['Day'], examplan['Slot'])] 
    slotcombinations, counts = np.unique(res, return_counts = True)
   

    #create dict to remap room IDs
    roomdict = {}
    for slotcombination in slotcombinations:
        roomdict[slotcombination] = 0

    for row in range(len(res)):
        slotcombination = res[row]
        res[row] = roomdict[slotcombination]
        roomdict[slotcombination] = roomdict[slotcombination] + 1

    examplan['Room_ID'] = res
    rooms=["Raum "+str(i) for i in range(max(counts))]
    slot_to_time = {0:'8',1:'10',2:'12',3:'14',4:'16', 5:'18'}
    for week in range(max(examplan['Week'].values) +1):
        weekplan = examplan[examplan['Week'] == week]
        week_label = week + 1
        for day in range(max(weekplan['Day'].values) +1):
            dayplan = weekplan[weekplan['Day'] == day]
            fig=plt.figure(figsize=(10,5.89))
            day_label = day + 1 
            for exam in dayplan.values:
                slot_label = exam[3]
                exam_label = exam[0]
                Room_ID = exam[4]
                event=exam_label
                #data=map(float, data[:-1])
                room=Room_ID-0.48+1
                start=int(slot_to_time[slot_label])
                end=int(slot_to_time[slot_label+1])
                # plot event
                plt.fill_between([room, room+0.96], [start, start], [end,end], color=colors[int(Room_ID)], edgecolor='k', linewidth=0.5)
                # plot beginning time
                plt.text(room+0.02, start+0.05 ,'{0}:00'.format(int(start)), va='top', fontsize=7)
                # plot event name
                plt.text(room+0.48, (start+end)*0.5, event, ha='center', va='center', fontsize=11)
            # Set Axis
            ax=fig.add_subplot(111)
            ax.yaxis.grid()
            ax.set_xlim(0.5,len(rooms)+0.5)
            ax.set_ylim(18.1, 7.9)
            ax.set_xticks(range(1,len(rooms)+1))
            ax.set_xticklabels(rooms)
            ax.set_ylabel('Uhrzeit')

            # Set Second Axis
            ax2=ax.twiny().twinx()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_ylim(ax.get_ylim())
            ax2.set_xticks(ax.get_xticks())
            ax2.set_xticklabels(rooms)
            ax2.set_ylabel('Uhrzeit')

            plt.title('Woche ' + str(week_label) + ', Tag ' + str(day_label),y=1.07)
            if folderpath != "":
                plt.savefig(folderpath +'\\' + str(week_label) + '_' + str(day_label) + '.png', dpi=200)

def createplot_unique(examplan):
    '''
    Creates a plot for a singe student schedule

    arguments:
        examplan: exam timetable of a singe student, dataframe containing ['Student_ID', 'Week', 'Day', 'Slot', 'Exam']

    adapted from http://masudakoji.github.io/2015/05/23/generate-timetable-using-matplotlib/en/
    '''

    days=['Montag','Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag']
    colors=['lightgreen', 'lightblue', 'lightgreen', 'lightblue', 'lightgreen']

    #get necessary rooms: 

    slot_to_time = {0:'8',1:'10',2:'12',3:'14',4:'16', 5:'18'}
    fig, axs = plt.subplots(3, figsize=(10,5.89*3.5))
    for week in range(max(examplan['Week'].values) +1):
        weekplan = examplan[examplan['Week'] == week]
        week_label = week + 1
        #fig=plt.figure(figsize=(10,5.89))
        for day in range(max(weekplan['Day'].values) +1):
            dayplan = weekplan[weekplan['Day'] == day]
            day_label = day + 1 
            for exam in dayplan.values:
                slot_label = exam[3]
                exam_label = exam[4]
                Room_ID = exam[2]
                event=exam_label
                #data=map(float, data[:-1])
                room=Room_ID-0.48+1
                start=int(slot_to_time[slot_label])
                end=int(slot_to_time[slot_label+1])
                # plot event
                axs[week].fill_between([room, room+0.96], [start, start], [end,end], color=colors[int(Room_ID)], edgecolor='k', linewidth=0.5)
                # plot beginning time
                axs[week].text(room+0.02, start+0.05 ,'{0}:00'.format(int(start)), va='top', fontsize=7)
                # plot event name
                axs[week].text(room+0.48, (start+end)*0.5, event, ha='center', va='center', fontsize=11)
            # Set Axis
        #ax=fig.add_subplot(111)
        axs[week].yaxis.grid()
        axs[week].set_xlim(0.5,len(days)+0.5)
        axs[week].set_ylim(18.1, 7.9)
        axs[week].set_xticks(range(1,len(days)+1))
        axs[week].set_xticklabels(days)
        axs[week].set_ylabel('Uhrzeit')

        # Set Second Axis
        ax2=axs[week].twiny().twinx()
        ax2.set_xlim(axs[week].get_xlim())
        ax2.set_ylim(axs[week].get_ylim())
        ax2.set_xticks(axs[week].get_xticks())
        ax2.set_xticklabels(days)
        ax2.set_ylabel('Uhrzeit')

        plt.title('Woche ' + str(week_label),y=1.07)
    fig.subplots_adjust(hspace=0.3)
