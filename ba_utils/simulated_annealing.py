'''
Utilities for simulated annealing
'''

import pandas as pd
import numpy as np
from ba_utils import linear_solver
from ba_utils import general

def generate_start_solution(Exam_IDs, N_WEEKS, N_DAYS, N_SLOTS):
    '''
    generates a random starting solution for the simulated annealing process

    arguments:
        Exam_IDs: List of Exam names ['G001','G002']
        N_WEEKS, N_DAYS, N_SLOTS: avialable time for exams in weeks, days and slots
    returns:
        a dataframe with columns 'Exam_ID', 'Week', 'Day', 'Slot'
    '''
    N_EXAMS = len(Exam_IDs)
    # generate completely random initial solution
    weeks = np.random.randint(0, N_WEEKS, size=(N_EXAMS, 1))
    days = np.random.randint(0, N_DAYS, size=(N_EXAMS, 1))
    slots = np.random.randint(0, N_SLOTS, size=(N_EXAMS, 1))

    mat_exam = np.concatenate([weeks, days, slots], axis=1)
    df_exam = pd.DataFrame(mat_exam, columns=['Week','Day','Slot'])
    df_exam['Exam_ID'] = Exam_IDs
    df_exam = df_exam[['Exam_ID', 'Week', 'Day', 'Slot']]
    return df_exam

def get_student_table(exam_matrix_binary, examtime_dict, exam_ID):
    '''
    gets a partial student timetable for all students that write one specific exam

    arguments:
        exam_matrix_binary: the binary exam matrix
        examtime_dict: a dictionary containing exam times
        exam_ID: exam id to be looked for
    returns:
        a dataframe with columns 'Student_ID', 'Week','Day','Slot', 'Exam'
    '''
    student_table = []
    exam_matrix_binary = exam_matrix_binary[exam_matrix_binary[exam_ID] == 1]
    for exam in exam_matrix_binary.columns:
        for student in range(len(exam_matrix_binary[exam])):
            if exam_matrix_binary[exam].values[student] == 1:
                student_table.append([student, examtime_dict[exam][0], examtime_dict[exam][1], examtime_dict[exam][2], exam])
    df_student = pd.DataFrame(student_table, columns=['Student_ID', 'Week','Day','Slot', 'Exam'])
    return df_student

def calc_score(df_Student, FACTOR):
    '''
    fast method to calculate the full score of a student timetable

    arguments:
        df_Student: full student timetable
    returns:
        the score of the timetable
    '''
    scoredict = general.get_time_dict(FACTOR)
    score = 0
    df_Student = df_Student.sort_values(['Student_ID', 'Week', 'Day', 'Slot'],
                                        ascending=(True, True, True, True))
    student_values = df_Student.values
    for row in range(1, len(student_values)):
        if student_values[row][0] == student_values[row-1][0]:
            score = score + scoredict.get((student_values[row-1][1], student_values[row][1],
                                           student_values[row-1][2], student_values[row][2],
                                           student_values[row-1][3], student_values[row][3]))
    return score

def run_simulation(exam_matrix_binary, config_object, END_SCORE = 1850, 
                   N_TEMP = 50, N_ITER = 200, p1 = 0.7, p50 = 0.0001, MAX_ITER = 3, start_solution=[], planned_exams=[]):
    '''
    adapted from from http://apmonitor.com/me575/index.php/Main/SimulatedAnnealing

    arguments: 
        exam_matrix_binary: the binary exam matrix
        config_object: tuple of (N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED)
        END_SCORE: Desired Score to break simulation loop early
        N_TEMP: number of Temperatur steps
        N_ITER: number of iterations per temperature step
        p1 : Probability of accepting worse solution at the start
        p50: Probability of accepting worse solution at the end
        MAX_ITER: maximum number of simulation loops 
        start_solution: start solution. Will be generated randomly if not given
        planned_exams: a list of preplanned exams
    return: 
        a pandas dataframe with weeks, days and slots of the exams    
    '''
    N_WEEKS, N_DAYS, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR = config_object
    Exam_IDs = list(exam_matrix_binary.columns.values)
    # exclude preplanned exams
    planned_exams = np.array(planned_exams)
    if len(planned_exams) > 0:
        Exam_IDs.remove(planned_exams[:,0])
    N_EXAMS = len(Exam_IDs)
    if start_solution == []:
        df_exam = generate_start_solution(Exam_IDs, N_WEEKS, N_DAYS, N_SLOTS)
    else:
        df_exam = start_solution
    score_full = 10000

    # apppend preplanned exams:
    if len(planned_exams) > 0:
        planned_exams = pd.DataFrame([[i[0], int(i[1]), int(i[2]), int(i[3])] for i in planned_exams], columns=['Exam_ID', 'Week', 'Day', 'Slot'])
        df_exam = df_exam.append(planned_exams, ignore_index=True)
    #Repeat model until a good score is achieved
    loopcounter = 0
    # Number of accepted solutions
    na = 0
    # Initial temperature
    t1 = -1.0/np.log(p1)
    # Final temperature
    t50 = -1.0/np.log(p50)
    # Fractional reduction every cycle
    frac = (t50/t1)**(1.0/(N_TEMP-1.0))
    # Initialize score_log
    while score_full > END_SCORE and loopcounter < MAX_ITER:
        # start temperature
        t = t1
        # DeltaE Average
        DeltaE_avg = 0.0
        for temp in range(N_TEMP): 
            print('Cycle: ' + str(temp) + ' with Temperature: ' + str(t))
            for Iter in range(N_ITER):     
                #select random neighbours and only allow feasible movement of exams
                row = np.random.randint(0,N_EXAMS)
                exam = df_exam['Exam_ID'].values[row]
                #print("Switching exam " + str(exam))
                oldvals = df_exam.iloc[row].values
                # Generate random new position
                NewWeek = np.random.randint(0, 3)
                NewDay = np.random.randint(0, 5)
                NewSlot = np.random.randint(0, 5)
                #compute change in cost function
                # old score
                examtime_dict = general.get_examtime_dict(df_exam)
                df_score_old = get_student_table(exam_matrix_binary, examtime_dict, exam)
                score_old= calc_score(df_score_old, FACTOR)
                # new score
                df_exam.iloc[row] = [exam, NewWeek, NewDay, NewSlot]
                examtime_dict = general.get_examtime_dict(df_exam)
                df_score_new = get_student_table(exam_matrix_binary, examtime_dict, exam)
                score_new = calc_score(df_score_new, FACTOR)
                DeltaE = abs(score_new - score_old)
                #generate random number if solution is worse and accept it with a propability
                #print("Score difference: " +str(score_new - score_old))
                if score_new - score_old > 0:
                    # Initialize DeltaE_avg if a worse solution was found
                    #   on the first iteration
                    if (temp==0 and Iter==0): 
                        DeltaE_avg = DeltaE
                    p = np.exp(-DeltaE/(DeltaE_avg * t))
                    if (np.random.random()<p):
                        # accept the worse solution
                        accept = True
                    else:
                        accept = False
                        #restore old solution
                        df_exam.iloc[row] = oldvals
                #if solution is better always accept
                else:
                    accept = True
                if (accept):
                    # increment number of accepted solutions
                    na = na + 1
                    # update DeltaE_avg
                    DeltaE_avg = (DeltaE_avg * (na-1) +  DeltaE) / na
            # Record the score at the end of each cycle
            examtime_dict = general.get_examtime_dict(df_exam)
            df_score_full = general.get_full_student_table(exam_matrix_binary, examtime_dict)
            score_full = calc_score(df_score_full, FACTOR)
            print('Score after Cycle ' + str(temp) + ': '+ str(score_full))
            # Lower the temperature for next cycle
            t = frac * t
        loopcounter = loopcounter +1
    return df_exam

def fixplan(config_object, exam_matrix_binary, exam_plan, exam_planned_timeonly, MaxCapacity):
    '''
    tries to fix an exam plan for one day, generated by simulated annealing

    arguments:
        config_object: tuple of (N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED)
        exam_matrix_binary: the binary exam matrix
        exam_plan: dataframe containing exam times
        exam_planned_timeonly: a list [['G013', 1],['G160', 1]...] containing exams that have fixed slots
        MaxCapacity: List of slot capacities
    returns:
        a reordered partial exam plan for one day
    '''
    N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR = config_object

    #exam_planned_timeonly = [['G013',1],['G160',1]]
    exam_plan_vals = exam_plan.sort_values(['Week', 'Day', 'Slot'], ascending=(True, True, True)).values
    for week in range(N_WEEKS):
        for day in range(N_DAYS_PER_WEEK):
            reduced_df = exam_plan[exam_plan['Week'] == week]
            reduced_df = reduced_df[reduced_df['Day'] == day]
            exams_to_plan = reduced_df['Exam_ID']
            preplanned = []
            for ex_timed in exam_planned_timeonly:
                if ex_timed[0] in exams_to_plan.values:
                    preplanned.append([ex_timed[0], ex_timed[1]])
            # linear solver for day rearranging
            optimized_day = linear_solver.run_optimise_day(exam_matrix_binary, exams_to_plan, MaxCapacity, preplanned)
            #put results into data field
            for exam in optimized_day.values:
                for row2 in range(len(exam_plan_vals)):
                    if exam_plan.iat[row2, 0] == exam[0]:
                        exam_plan.iat[row2, 3] = exam[1]
    return exam_plan