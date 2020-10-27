'''
General utilities and data preperation
'''
import math
import pandas as pd
import numpy as np

def prepare_data(path_student_data, path_exam_data):
    '''
    arguments:
        path_student_data: Contains path to file with student data
        path_exam_data: Contains path to file with exam group data

    returns:
        exam_matrix_binary: binary matrix, containing the exam choices of all students
        exam_groups: list of lists, containing Exam Codes for the first four semesters
    '''
    student_data = pd.read_excel(path_student_data)
    exam_data = pd.read_excel(path_exam_data)

    #exam dictionaries
    exam_data_relevant = exam_data[exam_data['Gruppen_ID']!=0].sort_values(by=['Gruppen_ID'])
    exam_data_relevant = exam_data[exam_data['planen']==1].sort_values(by=['Gruppen_ID'])
    exam_groups = []
    for i in range(4):
        exam_groups.append(np.unique(exam_data_relevant[exam_data_relevant['Semester'] == i+1]['Gruppen_ID']))
    group_list = np.unique(exam_data_relevant['Gruppen_ID'])

    n_students = max(student_data["Stud_ID"])
    n_exams = len(group_list)
    exam_matrix_binary = np.zeros((n_students, n_exams),np.dtype(int))
    exam_matrix_binary = pd.DataFrame(exam_matrix_binary)
    exam_matrix_binary.columns = group_list

    exam_to_group = {}
    for nrow in range(exam_data_relevant["Gruppen_ID"].size):
        # for several exams with same PP_Code
        if exam_data_relevant["PP_Code"].values[nrow] in exam_to_group:
            exam_to_group[exam_data_relevant["PP_Code"].values[nrow]] = [exam_to_group[exam_data_relevant["PP_Code"].values[nrow]],exam_data_relevant["Gruppen_ID"].values[nrow]]
        else:
            exam_to_group[exam_data_relevant["PP_Code"].values[nrow]] = exam_data_relevant["Gruppen_ID"].values[nrow]
    #generate Student Rows
    student_data = student_data.sort_values(by=['Stud_ID'])
    exam_choice_list = [[z,[y,x]] for x, y, z in zip(student_data['ANCODE'], student_data['STG'], student_data['Stud_ID'])]
    for choices in exam_choice_list:
        if choices[1][1] < 10:
            numberdnummy = "00" + str(choices[1][1])
        elif choices[1][1] < 100:
            numberdnummy = "0" + str(choices[1][1])
        else:
            numberdnummy = str(choices[1][1])
        bchoice = choices[1][0]+numberdnummy
        if bchoice in exam_to_group:
            exam_matrix_binary.at[choices[0]-1, exam_to_group[bchoice]] = 1
    # delete empty cols:
    exam_matrix_binary = exam_matrix_binary.loc[:, (exam_matrix_binary != 0).any(axis=0)]
    # delete empty rows
    nr_exams = exam_matrix_binary.sum(axis=1)
    row_sums_0 = exam_matrix_binary[nr_exams == 0].index.values
    exam_matrix_binary = exam_matrix_binary.drop(row_sums_0).reset_index(drop=True)

    return exam_matrix_binary, exam_groups

def get_remaining_over(exam_matrix_binary, planned_exams, x):
    '''
    arguments:
        exam_matrix_binary: the binary exam matrix to evaluate
        planned_exams: list of already planned exams
        x: number of attending students
    returns:
        a list of exams, that are in exam_matrix_binary, but not in planned_exams and have more students attending than x
    '''
    rest_matrix = exam_matrix_binary.drop(planned_exams[:,0], axis = 1)
    order = pd.DataFrame(rest_matrix.sum())
    order.columns = ["n_students"]
    order = order.sort_values(by=['n_students'])
    remaining_over_x = order[order["n_students"]>x].index.values
    return remaining_over_x

def get_examtime_dict(weekplan):
    '''
    creates a dictionary that maps exams to their times

    arguments:
        weekplan: dataframe containing rows ['Exam_ID', 'Week', 'Day', 'Slot']
    returns:
        dictionary
    '''
    examtime_dict = {}
    examtimes = [[week, day, slot]  for week, day, slot in zip(weekplan['Week'], weekplan['Day'], weekplan['Slot'])] 
    for exam_nr in range(len(weekplan['Exam_ID'])):
        examtime_dict[weekplan['Exam_ID'].values[exam_nr]] = examtimes[exam_nr]
    return examtime_dict

def get_full_student_table(exam_matrix_binary, examtime_dict):
    '''
    gets the full student timetable

    arguments:
        exam_matrix_binary: the binary exam matrix 
        examtime_dict: a dictionary containing exam times
    returns:
        a dataframe with columns 'Student_ID', 'Week','Day','Slot', 'Exam'
    '''
    student_table = []
    for exam in exam_matrix_binary.columns:
        for student in range(len(exam_matrix_binary[exam])):
            if exam_matrix_binary[exam].values[student] == 1:
                student_table.append([student, examtime_dict[exam][0], examtime_dict[exam][1], examtime_dict[exam][2], exam])
    df_student = pd.DataFrame(student_table, columns=['Student_ID', 'Week','Day','Slot', 'Exam'])
    df_student = df_student.sort_values(by=['Student_ID'])
    return df_student

def get_time_dict(FACTOR):
    '''
    creates a dictionary that enables fast score calculations
    FACTOR: Factor by which exams on the same day are weightened more than exams on consecutive days
    returns:
        dictionary
    '''
    timedict = {}
    for week1 in range(3):
        for week2 in range(3):
            if week1 <= week2:
                for day1 in range(5):
                    for day2 in range(5):
                        for slot1 in range(5):
                            for slot2 in range(5):
                                if week1 == week2 and day1 == day2 and slot1 == slot2:
                                    timedict[(week1, week2, day1, day2, slot1, slot2)] = 1000
                                elif week1 == week2 and day1 == day2 and slot1 < slot2:
                                    timedict[(week1, week2, day1, day2, slot1, slot2)] = 5
                                elif week1 == week2 and day1 + 1 == day2:
                                    timedict[(week1, week2, day1, day2, slot1, slot2)] = 1
                                else:
                                    timedict[(week1, week2, day1, day2, slot1, slot2)] = 0
    return timedict

def get_exam_student_count_dict(exam_matrix_binary):
    '''
    creates a dictionary that maps exams to number of attending students

    arguments:
        exam_matrix_binary: binary matrix
    returns:
        dictionary
    '''
    exam_student_count = exam_matrix_binary.sum(axis=0)
    exam_student_count_dict = {}
    for exam in range(len(exam_matrix_binary.columns)):
        exam_student_count_dict[exam_matrix_binary.columns[exam]] = exam_student_count[exam]
    return exam_student_count_dict

def calc_score_array(df_student, FACTOR):
    '''
    calculates scores for each individual student in a student timetable

    arguments:
        df_student: student timetable (dataframe) containing columns ['Student_ID','Week','Day','Slot']
        FACTOR: Factor by which exams on the same day are weightened more than exams on consecutive days
    returns:
        score array with scores for each individual student
    '''
    score = 0
    scoredict = get_time_dict(FACTOR)   
    df_student = df_student.sort_values(['Student_ID', 'Week', 'Day', 'Slot'], ascending = (True, True, True, True))
    #print(df_student)
    student_values = df_student.values
    score_arr = []
    consecutive_score = 0
    double_score = 0
    total_score = 0
    for row in range(1, len(student_values)):
        if student_values[row][0] == student_values[row-1][0]:
            score = scoredict.get((student_values[row-1][1], student_values[row][1], student_values[row-1][2],
                                        student_values[row][2], student_values[row-1][3], student_values[row][3]))
            total_score = total_score + score
            if score == FACTOR:
                double_score = double_score + FACTOR
            elif score == 1:
                consecutive_score = consecutive_score + 1
        else:
            score_arr.append([double_score, consecutive_score, total_score])
            consecutive_score = 0
            double_score = 0
            total_score = 0
    df_Score = pd.DataFrame(np.array(score_arr), columns = ['Double', 'Consecutive', 'Total'])
    return df_Score

def checkplan(exam_plan, exam_matrix_binary, arr_capacity_preset, exam_planned_timeonly, FACTOR):
    '''
    checks if a exam plan (generated by simulated annealing) fullfills all requirements

    prints problems in the exam plan to the screen

    arguments:
        exam_plan: a full exam plan
        exam_matrix_binary: binary matrix
        MaxCapacity: List of slot capacities
        exam_planned_timeonly: a list [['G013', 1],['G160', 1]...] containing exams that have fixed slots
        FACTOR: Factor by which exams on the same day are weightened more than exams on consecutive days
    '''
    examtime_dict = get_examtime_dict(exam_plan)
    exam_student_count_dict = get_exam_student_count_dict(exam_matrix_binary)
    #exam_planned_timeonly = [['G013',1],['G160',1]]
    for ex  in exam_planned_timeonly:
        if examtime_dict[ex[0]][2] != ex[1]:
            print("Exam " + ex + " not at right time")
    # capacity is enough?
    exam_plan = exam_plan.sort_values(['Week', 'Day', 'Slot'], ascending = (True, True, True))
    #capacity_arr = []
    exam_plan_vals = exam_plan.values
    occupation = 0
    prev_week = -1
    prev_day = -1
    prev_slow = -1
    parallel = 0
    for row in range(len(exam_plan_vals)):
        if prev_week == exam_plan_vals[row][1] and prev_day == exam_plan_vals[row][2] and prev_slow == exam_plan_vals[row][3]:
            occupation = occupation + exam_student_count_dict[exam_plan_vals[row][0]]
            #also count parallel excercises
            parallel = parallel + 1
        else:
            # maximum of 5 Exams parallel?
            if parallel > 5:
                print("Too many parallel excercises on week "+ str(exam_plan_vals[row-1][1]) +", day "+ str(exam_plan_vals[row-1][2]) +", slot " + str(exam_plan_vals[row-1][3]))
            if arr_capacity_preset[exam_plan_vals[row-1][3]] - occupation < 0:
                print(str(occupation - arr_capacity_preset[exam_plan_vals[row-1][3]]) + " too many students on week "+ str(exam_plan_vals[row-1][1]) 
                    +", day "+ str(exam_plan_vals[row-1][2]) +", slot " + str(exam_plan_vals[row-1][3]))
            #set new compare values
            occupation = exam_student_count_dict[exam_plan_vals[row][0]]
            prev_week = exam_plan_vals[row][1]
            prev_day = exam_plan_vals[row][2]
            prev_slow = exam_plan_vals[row][3]
            parallel = 1
    # no hard conflicts?
    examtime_dict = get_examtime_dict(exam_plan)
    df_score_full = get_full_student_table(exam_matrix_binary, examtime_dict)
    score_arr = calc_score_array(df_score_full, FACTOR)
    if max(score_arr['Total']) >= 1000: 
        print("Hard Collision")


def get_written_per_day(exam_plan, exam_matrix_binary):
    '''
    arguments:
        exam_plan: a full exam plan
    return: a list containing how many exams are written in total per day
    '''
    exam_student_count_dict = get_exam_student_count_dict(exam_matrix_binary)
    exams_over_days = []
    cumulative = 0
    for day in range(15):
        weekplan = exam_plan[exam_plan['Week'] == math.floor(day/5)]
        dayplan = weekplan[weekplan['Day'] == day%5]
        daycount = 0
        for exam in dayplan['Exam_ID']:
            daycount = daycount + exam_student_count_dict[exam]    
        cumulative = cumulative + daycount
        exams_over_days.append([day, daycount, cumulative])
    return exams_over_days