'''
Linear Solver utilities, CBC Solver
'''

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

def solveExamSchedule_weeks(config_object, exam_choice, N_EXAMS, N_STUDENTS, rowweights, planned_exams =[]):
    '''
    Designed for internal usage within the module. Runs the actual solver for week planning.

    arguments:
        config_object: tuple of (N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR)
        exam_choice: List containing different exam choices of students [[1,3,7],[2,5,6]...]
        N_EXAMS: Number of exams to be planned
        N_STUDENTS: Number of Students with different exam combinations
        rowweights: weigthing each exam_choice by the amount of students with the combination
        planned_exams: optional dataframe for preplanned exams
    '''
    # constants:
    N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR = config_object
    S = pywraplp.Solver('SolveAssignmentProblem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    #exam_choice look like [[1,3,7],[2,5,6]...]
    #new decision variable: placement of exams
    exam_placing = [[S.IntVar(0, 1, '') for exam in range(N_EXAMS)] for week in range(N_WEEKS)]

    #evaluation variable (pseudo decision): Timetable for students
    student_schedule = [[[S.IntVar(0,1,'') for exam in range(N_EXAMS)] for week in range(N_WEEKS)] for student in range(N_STUDENTS)]

    #fill timetable accoring to exams
    for student in range(N_STUDENTS):
        for week in range(N_WEEKS):
            for exam in exam_choice[student]:
                student_schedule[student][week][exam] = exam_placing[week][exam]

    '''helper variables'''
    double_booked_weeks = [[S.IntVar(0,8-WEEKLY_EXAM_STRESS_ALLOWED,'') for week in range(N_WEEKS)] for student in range(N_STUDENTS)]
    #double_booked_weeks_trigger = [[S.IntVar(0,1,'') for week in range(N_WEEKS)] for student in range(N_STUDENTS)]
    normal_booked_weeks = [[S.IntVar(0,WEEKLY_EXAM_STRESS_ALLOWED,'') for week in range(N_WEEKS)] for student in range(N_STUDENTS)]

    '''Constraints'''

    '''OPTIMIZE WEEKS'''
    # differentiate between normal exam stress and extra exam stress
    for week in range(N_WEEKS):
        for student in range(N_STUDENTS):
            limit=S.Sum([student_schedule[student][week][exam] for exam in range(N_EXAMS)])
            S.Add(double_booked_weeks[student][week] + normal_booked_weeks[student][week] == limit)

    # normal weeks are filled first
    #allow only three regular exams per week
    for week in range(N_WEEKS):
        for student in range(N_STUDENTS):
            #S.Add(normal_booked_weeks[student][week]/WEEKLY_EXAM_STRESS_ALLOWED <= WEEKLY_EXAM_STRESS_ALLOWED)
            S.Add(normal_booked_weeks[student][week] <= WEEKLY_EXAM_STRESS_ALLOWED)

    #activate trigger
    # for week in range(N_WEEKS):
    #     for student in range(N_STUDENTS):
    #         S.Add(double_booked_weeks_trigger[student][week] <= normal_booked_weeks[student][week]/WEEKLY_EXAM_STRESS_ALLOWED)

    # fill excess after trigger has been activated
    for week in range(N_WEEKS):
        for student in range(N_STUDENTS):
            S.Add(double_booked_weeks[student][week] <= 6)
    '''Other constraints'''
    #limit number of students to capacity

    # every exam gets exactly one slot
    for exam in range(N_EXAMS):
        S.Add(S.Sum([exam_placing[week][exam] for week in range(N_WEEKS)]) == 1)
    
    #every student writes as much exams as chosen
    for student in range(N_STUDENTS):
        S.Add(len(exam_choice[student]) == S.Sum([student_schedule[student][week][exam] 
            for exam in range(N_EXAMS) 
            for week in range(N_WEEKS)]))

    # fix preplanned exams
    for planned in planned_exams:
        exam = int(planned[0])
        week = int(planned[1])
        S.Add(exam_placing[week][exam] == 1)

    '''objective/scoring'''
    exam_cost = S.Sum([double_booked_weeks[student][week]*rowweights[student] for week in range(N_WEEKS) for student in range(N_STUDENTS)])
    S.Minimize(exam_cost)

    '''set time limit in ms'''
    S.set_time_limit(60*60*1000)

    S.EnableOutput()
    '''start solver'''
    res = S.Solve()

    resdict = {0:'OPTIMAL', 1:'FEASIBLE', 2:'INFEASIBLE', 3:'UNBOUNDED', 
            4:'ABNORMAL', 5:'MODEL_INVALID', 6:'NOT_SOLVED'}

    print('LP solver result:', resdict[res])

    l_exam = [(exam, week) 
        for week in range(N_WEEKS) 
        for exam in range(N_EXAMS) if exam_placing[week][exam].solution_value()>0]

    df_exam = pd.DataFrame(l_exam, columns=['Exam_ID','Week'])
    Week_Count = S.Objective().Value()
    return df_exam, Week_Count

def solveExamSchedule_days(config_object, exam_choice, N_EXAMS, N_STUDENTS, rowweights, MaxCapacity, planned_exams =[], planned_exams_timeonly=[]):
    '''
    Designed for internal usage within the module. Runs the actual solver for day planning.

    arguments:
        config_object: tuple of (N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR)
        exam_choice: List containing different exam choices of students [[1,3,7],[2,5,6]...]
        N_EXAMS: Number of exams to be planned
        N_STUDENTS: Number of Students with different exam combinations
        rowweights: weigthing each exam_choice by the amount of students with the combination
        MaxCapacity: List of slot capacities
        planned_exams: optional dataframe for preplanned exams
        planned_exams_timeonly: optional dataframe for preplanned exams with only slot constraints
    '''
    # constants:
    N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR = config_object
    S = pywraplp.Solver('SolveAssignmentProblem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    #exam_choice look like [[1,3,7],[2,5,6]...]
    #new decision variable: placement of exams
    exam_placing = [[[S.IntVar(0,1,'') for exam in range(N_EXAMS)] for slot in range(N_SLOTS)] for day in range(N_DAYS_PER_WEEK)]

    #evaluation variable (pseudo decision): Timetable for students
    student_schedule = [[[[S.IntVar(0,1,'') for exam in range(N_EXAMS)] for slot in range(N_SLOTS)] for day in range(N_DAYS_PER_WEEK)] 
                        for student in range(N_STUDENTS)]

    #fill timetable accoring to exams
    for student in range(N_STUDENTS):
        for day in range(N_DAYS_PER_WEEK):
            for slot in range(N_SLOTS):
                for exam in exam_choice[student]:
                    student_schedule[student][day][slot][exam] = exam_placing[day][slot][exam]

    slot_occupancy = [[S.Sum([student_schedule[student][day][slot][exam]*rowweights[student] 
                       for exam in range(N_EXAMS) 
                       for student in range(N_STUDENTS)])
                       for slot in range(N_SLOTS)]
                       for day in range(N_DAYS_PER_WEEK)]

    '''helper variables'''
    double_booked_days = [[S.IntVar(0, 1, '') for day in range(N_DAYS_PER_WEEK)] for student in range(N_STUDENTS)]
    normal_booked_days = [[S.IntVar(0, 1, '') for day in range(N_DAYS_PER_WEEK)] for student in range(N_STUDENTS)]
    consecutive_days_a = [[S.IntVar(0, 1, '') for day in range(N_DAYS_PER_WEEK-1)] for student in range(N_STUDENTS)]
    consecutive_days_b = [[S.IntVar(0, 1, '') for day in range(N_DAYS_PER_WEEK-1)] for student in range(N_STUDENTS)]
    '''Constraints'''

    '''OPTIMIZE DAYS'''
    # differentiate between normal exam stress and extra exam stress
    for day in range(N_DAYS_PER_WEEK):
        for student in range(N_STUDENTS):
            limit = S.Sum([student_schedule[student][day][slot][exam] 
                           for exam in range(N_EXAMS) for slot in range(N_SLOTS)])
            S.Add(double_booked_days[student][day] + normal_booked_days[student][day] == limit)
            # normal days are filled first
            # S.Add(double_booked_days[student][day] <= normal_booked_days[student][day])

    '''OPTIMIZE CONSECUTIVE DAYS'''
    for day in range(N_DAYS_PER_WEEK-1):
        for student in range(N_STUDENTS):
            S.Add(consecutive_days_a[student][day] + consecutive_days_b[student][day] ==
                  S.Sum([normal_booked_days[student][day]])+
                  S.Sum([normal_booked_days[student][day+1]]))
    '''Other constraints'''
    #limit number of students to capacity
    for day in range(N_DAYS_PER_WEEK):
        for slot in range(N_SLOTS):
            S.Add(slot_occupancy[day][slot] <= MaxCapacity[slot])

    # every exam gets exactly one slot
    for exam in range(N_EXAMS):
        S.Add(S.Sum([exam_placing[day][slot][exam] 
                    for slot in range(N_SLOTS) 
                    for day in range(N_DAYS_PER_WEEK)]) == 1)

    # every Student a maximum of one exam per slot
    for student in range(N_STUDENTS):
        for day in range(N_DAYS_PER_WEEK):
            for slot in range(N_SLOTS):
                S.Add(S.Sum([student_schedule[student][day][slot][exam] for exam in range(N_EXAMS)]) <= 1)

    #every student writes as much exams as chosen
    for student in range(N_STUDENTS):
        S.Add(len(exam_choice[student]) == S.Sum([student_schedule[student][day][slot][exam]
            for exam in range(N_EXAMS) 
            for slot in range(N_SLOTS)
            for day in range(N_DAYS_PER_WEEK)]))
    # fix preplanned exams
    for planned in planned_exams:
        exam = int(planned[0])
        day = int(planned[1])
        slot = int(planned[2])
        S.Add(exam_placing[day][slot][exam] == 1)

    # block slots for exams with preplanned time
    for planned in planned_exams_timeonly:
        exam = int(planned[0])
        required_slot = int(planned[1])
        for day in range(N_DAYS_PER_WEEK):
            for slot in range(N_SLOTS):
                if not slot == required_slot:
                    S.Add(exam_placing[day][slot][exam] == 0)

    '''objective/scoring'''

    daily_double_count = S.Sum([double_booked_days[student][day]*rowweights[student]
                                for day in range(N_DAYS_PER_WEEK)
                                for student in range(N_STUDENTS)])
    consecutive_count = S.Sum([consecutive_days_b[student][day]*rowweights[student]
                               for day in range(N_DAYS_PER_WEEK-1)
                               for student in range(N_STUDENTS)])

    exam_cost = daily_double_count * FACTOR + consecutive_count
    S.Minimize(exam_cost)

    '''set time limit in ms'''
    S.set_time_limit(60*60*1000)

    S.EnableOutput()
    '''start solver'''
    res = S.Solve()

    resdict = {0:'OPTIMAL', 1:'FEASIBLE', 2:'INFEASIBLE', 3:'UNBOUNDED',
            4:'ABNORMAL', 5:'MODEL_INVALID', 6:'NOT_SOLVED'}

    print('LP solver result:', resdict[res])

    l_exam = [(exam, day, slot) 
        for day in range(N_DAYS_PER_WEEK) 
        for exam in range(N_EXAMS) 
        for slot in range(N_SLOTS) if exam_placing[day][slot][exam].solution_value() > 0]

    df_exam = pd.DataFrame(l_exam, columns=['Exam_ID','Day','Slot'])
    Double_Count = S.Objective().Value()
    #consecutives = [(student, day, rowweights[student]) 
    #    for day in range(N_DAYS_PER_WEEK-1)
    #    for student in range(N_STUDENTS) if consecutive_days_b[student][day].solution_value() > 0]
    return df_exam ,Double_Count

def optimise_day(exam_choice, N_EXAMS, N_STUDENTS, rowweights, MaxCapacity, planned_exams =[]):
    '''
    Designed for internal usage within the module. Runs the actual solver for single day planning.

    arguments:
        exam_choice: List containing different exam choices of students [[1,3,7],[2,5,6]...]
        N_EXAMS: Number of exams to be planned
        N_STUDENTS: Number of Students with different exam combinations
        rowweights: weigthing each exam_choice by the amount of students with the combination
        MaxCapacity: List of slot capacities
        planned_exams: optional dataframe for preplanned exams with only slot constraints
    '''
    # constants:
    N_SLOTS = len(MaxCapacity)
    S = pywraplp.Solver('SolveAssignmentProblem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)


    #new decision variable: placement of exams
    exam_placing = [[S.IntVar(0,1,'') for exam in range(N_EXAMS)] for slot in range(N_SLOTS)]

    #evaluation variable (pseudo decision): Timetable for students
    student_schedule = [[[S.IntVar(0,1,'') for exam in range(N_EXAMS)] for slot in range(N_SLOTS)]
                        for student in range(N_STUDENTS)]

    #fill timetable accoring to exams
    for student in range(N_STUDENTS):
        for slot in range(N_SLOTS):
            for exam in exam_choice[student]:
                student_schedule[student][slot][exam] = exam_placing[slot][exam]

    slot_occupancy = [S.Sum([student_schedule[student][slot][exam]*rowweights[student] for exam in range(N_EXAMS) 
                    for student in range(N_STUDENTS)])
                    for slot in range(N_SLOTS)]

    '''Other constraints'''
    #limit number of students to capacity
    for slot in range(N_SLOTS):
        S.Add(slot_occupancy[slot] <= MaxCapacity[slot])

    # every exam gets exactly one slot
    for exam in range(N_EXAMS):
        S.Add(S.Sum([exam_placing[slot][exam] for slot in range(N_SLOTS)]) == 1)

    # every Student a maximum of one exam per slot
    for student in range(N_STUDENTS):
        for slot in range(N_SLOTS):
            S.Add(S.Sum([student_schedule[student][slot][exam] for exam in range(N_EXAMS)]) <=1)

    #every student writes as much exams as chosen
    for student in range(N_STUDENTS):
        S.Add(len(exam_choice[student]) == S.Sum([student_schedule[student][slot][exam] 
            for exam in range(N_EXAMS) 
            for slot in range(N_SLOTS)]))

    for planned in planned_exams:
        exam = planned[0]
        slot = planned[1]
        S.Add(exam_placing[slot][exam] == 1)

    '''objective/scoring'''
    capacity_dummy = [S.IntVar(0,1,'') for slot in range(N_SLOTS)]

    capa_score = S.Sum(capacity_dummy)
    S.Maximize(capa_score)

    '''set time limit in ms'''
    S.set_time_limit(10*60*1000)

    S.EnableOutput()
    '''start solver'''
    res = S.Solve()

    resdict = {0:'OPTIMAL', 1:'FEASIBLE', 2:'INFEASIBLE', 3:'UNBOUNDED', 
            4:'ABNORMAL', 5:'MODEL_INVALID', 6:'NOT_SOLVED'}

    print('LP solver result:', resdict[res])

    l_exam = [(exam, slot)  
        for exam in range(N_EXAMS) 
        for slot in range(N_SLOTS) if exam_placing[slot][exam].solution_value()>0]

    df_exam = pd.DataFrame(l_exam, columns=['Exam_ID','Slot'])
    return df_exam 

def solveExamSchedule(config_object, exam_choice, N_EXAMS, N_STUDENTS, rowweights, MaxCapacity, planned_exams =[]):
    '''
    Designed for internal usage within the module. Runs the actual solver for planning the entire exam schedule in one step.

    arguments:
        config_object: tuple of (N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR)
        exam_choice: List containing different exam choices of students [[1,3,7],[2,5,6]...]
        N_EXAMS: Number of exams to be planned
        N_STUDENTS: Number of Students with different exam combinations
        rowweights: weigthing each exam_choice by the amount of students with the combination
        MaxCapacity: List of slot capacities
        planned_exams: optional dataframe for preplanned exams
    '''

    # constants:
    N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR = config_object
    S = pywraplp.Solver('SolveAssignmentProblem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    #exam_choice look like [[1,3,7],[2,5,6]...]
    #new decision variable: placement of exams
    exam_placing = [[[[S.IntVar(0,1,'') for exam in range(N_EXAMS)] for slot in range(N_SLOTS)] for day in range(N_DAYS_PER_WEEK)]
        for week in range(N_WEEKS)]

    #evaluation variable (pseudo decision): Timetable for students
    student_schedule = [[[[[S.IntVar(0,1,'') for exam in range(N_EXAMS)] for slot in range(N_SLOTS)] for day in range(N_DAYS_PER_WEEK)]
        for week in range(N_WEEKS)] for student in range(N_STUDENTS)]

    #fill timetable accoring to exams
    for student in range(N_STUDENTS):
        for week in range(N_WEEKS):
            for day in range(N_DAYS_PER_WEEK):
                for slot in range(N_SLOTS):
                    for exam in exam_choice[student]:
                        student_schedule[student][week][day][slot][exam] = exam_placing[week][day][slot][exam]

    slot_occupancy = [[[S.Sum([student_schedule[student][week][day][slot][exam]*rowweights[student] for exam in range(N_EXAMS) 
                    for student in range(N_STUDENTS)])
                    for slot in range(N_SLOTS)]
                    for day in range(N_DAYS_PER_WEEK)]
                    for week in range(N_WEEKS)]

    '''helper variables'''
    double_booked_days = [[[S.IntVar(0,DAILY_EXAM_STRESS_ALLOWED,'') for day in range(N_DAYS_PER_WEEK)]for week in range(N_WEEKS)] for student in range(N_STUDENTS)]
    normal_booked_days = [[[S.IntVar(0,1,'') for day in range(N_DAYS_PER_WEEK)]for week in range(N_WEEKS)] for student in range(N_STUDENTS)]
    consecutive_days_a = [[[S.IntVar(0,1,'') for day in range(N_DAYS_PER_WEEK-1)]for week in range(N_WEEKS)] for student in range(N_STUDENTS)]
    consecutive_days_b = [[[S.IntVar(0,4,'') for day in range(N_DAYS_PER_WEEK-1)]for week in range(N_WEEKS)] for student in range(N_STUDENTS)]
    '''Constraints'''

    '''OPTIMIZE DAYS'''
    #differentiate between normal exam stress and extra exam stress
    for week in range(N_WEEKS):
        for day in range(N_DAYS_PER_WEEK):
            for student in range(N_STUDENTS):
                S.Add(double_booked_days[student][week][day] + normal_booked_days[student][week][day] == S.Sum([student_schedule[student][week][day][slot][exam] for exam in range(N_EXAMS) for slot in range(N_SLOTS)]))
                # normal days are filled first
                # allow only one regular exam per day
                S.Add(normal_booked_days[student][week][day] <= 1)
                #allow double booking only if trigger has been reached
                #S.Add(double_booked_days[student][week][day] <= DAILY_EXAM_STRESS_ALLOWED)
    
    '''OPTIMIZE CONSECUTIVE DAYS'''
    for week in range(N_WEEKS):
        for day in range(N_DAYS_PER_WEEK-1):
            for student in range(N_STUDENTS):
                S.Add(consecutive_days_a[student][week][day] + consecutive_days_b[student][week][day] == 
                      normal_booked_days[student][week][day] + normal_booked_days[student][week][day+1])

    '''Other constraints'''
    #limit number of students to capacity
    for week in range(N_WEEKS):
        for day in range(N_DAYS_PER_WEEK):
            for slot in range(N_SLOTS):
                S.Add(slot_occupancy[week][day][slot] <= MaxCapacity[slot])

    #every exam gets exactly one slot
    for exam in range(N_EXAMS):
        S.Add(S.Sum([exam_placing[week][day][slot][exam] for slot in range(N_SLOTS) for day in range(N_DAYS_PER_WEEK) for week in range(N_WEEKS)]) == 1)

    #every Student a maximum of one exam per slot
    for student in range(N_STUDENTS):
        for week in range(N_WEEKS):
            for day in range(N_DAYS_PER_WEEK):
                for slot in range(N_SLOTS):
                    S.Add(S.Sum([student_schedule[student][week][day][slot][exam] for exam in range(N_EXAMS)]) <=1)
    
    # fix preplanned exams
    for planned in planned_exams:
        exam = planned[0]
        week = planned[1]
        day = planned[2]
        slot = planned[3]
        S.Add(exam_placing[week][day][slot][exam] == 1)

    #every student writes as much exams as chosen
    for student in range(N_STUDENTS):
        S.Add(len(exam_choice[student]) == S.Sum([student_schedule[student][week][day][slot][exam] 
            for exam in range(N_EXAMS) 
            for slot in range(N_SLOTS)
            for day in range(N_DAYS_PER_WEEK) 
            for week in range(N_WEEKS)]))
    '''objective/scoring'''
    daily_double_count = S.Sum([double_booked_days[student][week][day]*rowweights[student] 
        for day in range(N_DAYS_PER_WEEK)
        for week in range(N_WEEKS) 
        for student in range(N_STUDENTS)])
    consecutive_count = S.Sum([consecutive_days_b[student][week][day]*rowweights[student] 
        for day in range(N_DAYS_PER_WEEK-1)
        for week in range(N_WEEKS) 
        for student in range(N_STUDENTS)])

    exam_cost = daily_double_count*FACTOR + consecutive_count
    S.Minimize(exam_cost)

    '''set time limit in ms'''
    S.set_time_limit(60*60*1000)

    S.EnableOutput()
    '''start solver'''
    res = S.Solve()

    resdict = {0:'OPTIMAL', 1:'FEASIBLE', 2:'INFEASIBLE', 3:'UNBOUNDED', 
            4:'ABNORMAL', 5:'MODEL_INVALID', 6:'NOT_SOLVED'}

    print('LP solver result:', resdict[res])

    l_exam = [(exam, week, day, slot) 
        for day in range(N_DAYS_PER_WEEK) 
        for week in range(N_WEEKS) 
        for exam in range(N_EXAMS) 
        for slot in range(N_SLOTS) if exam_placing[week][day][slot][exam].solution_value()>0]

    df_exam = pd.DataFrame(l_exam, columns=['Exam_ID','Week','Day','Slot'])
    Double_Count = S.Objective().Value()
    return df_exam, Double_Count

# Helper functions
def get_choices_from_matrix(exam_matrix_binary):
    '''
    gets a list containing the numerical exam choices of each student 

    arguments:
        exam_matrix_binary: the binary matrix 
    returns:
        list containing choices [[1,3,7],[2,5,6]...]
    '''
    N_STUDENTS = len(exam_matrix_binary)
    N_EXAMS = len(exam_matrix_binary[0])
    exam_choice = []
    for student in range(N_STUDENTS):
        choice =[]
        for exam in range(N_EXAMS):
            if exam_matrix_binary[student][exam] ==1:
                choice.append(exam)
        exam_choice.append(choice)
    return exam_choice

def get_exams_semester(semester, Exam_groups):
    '''
    gets a list containing all exams until a specifies semester ( up to semester 4)

    arguments:
        semester: last semester to get exams from 
        Exam_groups: list of lists containing the exams in each semester
    returns:
        list containing exam IDs
    '''
    newcols = []
    for i in range(semester):
        newcols = np.append(newcols,Exam_groups[i])
    return newcols

def unique_matrix(exam_matrix_binary):
    '''
    gets a reduced matrix with unique rows as well as their counts

    arguments:
        exam_matrix_binary: the binary matrix 
    returns:
        reduced binary matrix, rowweights
    '''
    exam_matrix_binary, rowweights = np.unique(exam_matrix_binary, axis=0, return_counts=True)
    return exam_matrix_binary, rowweights

def get_group_to_row_dict(exam_matrix_binary):
    '''
    maps exams columns to their respective rows in the matrix

    arguments:
        exam_matrix_binary: binary matrix as dataframe
    returns:
        dictionary containing the row in the matrix for each indivicual exam
    '''
    group_to_row={}
    n=0
    for row in exam_matrix_binary.columns:
        group_to_row[row] = n
        n=n+1
    return group_to_row

def get_necessary_rows(matrix):
    '''
    gets all rowindexes from a matrix that have a sum bigger than 0 

    arguments:
        matrix: matrix to evaluate
    returns:
        list, containing all rows bigger than 0
    '''
    row_sums = matrix[matrix.sum(axis=1)>0]
    return row_sums.index.values

def run_optimise_day(exam_matrix_binary, exams_to_plan, MaxCapacity, planned_exams=[]):
    '''
    runs the optimisation process for a single day

    arguments:
        exam_matrix_binary: binary exam matrix
        exams_to_plan: exams to plan on the day
        MaxCapacity: List of slot capacities
        planned_exams list of: exams with fixed slots [['G013', 1], ['G160', 1], ...]
    returns:
        dataframe with exam times
    '''

    if len(planned_exams) > 0:
        allcols = np.sort(np.unique(np.append([i[0] for i in planned_exams], exams_to_plan)))
    else:
        allcols=np.sort(exams_to_plan)
    full_matrix = exam_matrix_binary[allcols]
    # exam_names = full_matrix.columns
    # get a group to row dict to remap the planned exams
    group_to_row =  get_group_to_row_dict(full_matrix)
    for exam in range(len(planned_exams)):
        planned_exams[exam][0] = group_to_row[planned_exams[exam][0]]
    reduced_matrix, rowweights = unique_matrix(full_matrix)
    reduced_choice_1 = get_choices_from_matrix(reduced_matrix)
    N_EXAMS = len(reduced_matrix[0])
    N_STUDENTS = len(reduced_matrix)
    df_exam = optimise_day(reduced_choice_1, N_EXAMS, N_STUDENTS, rowweights, MaxCapacity, planned_exams)
    df_exam = df_exam.sort_values(by=['Exam_ID'])
    df_exam['Exam_ID']= allcols
    return df_exam

def run_solver_days(config_object, exam_matrix_binary, exams_to_plan, iweek, MaxCapacity, planned_exams, planned_exams_timeonly):
    '''
    runs the day optimisation process for a single week

    arguments:
        config_object: tuple of (N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR)
        exam_matrix_binary: binary exam matrix
        exams_to_plan: exams to plan on in the week
        iweek: the week to be planned (for return values)
        planned_exams: list of exams with fixed times [['G001', 1 , 1],  ...]
        planned_exams_timeonly: list of exams with fixed slots [['G013', 1], ['G160', 1], ...]
        MaxCapacity: List of slot capacities
    returns:
        [week score], [dataframe with exam times]
    '''
    allcols=np.sort(exams_to_plan) 
    if len(planned_exams) > 0:
        allcols = np.sort(np.unique(np.append(allcols, planned_exams[:,0])))
    if len(planned_exams_timeonly) > 0:
        allcols = np.sort(np.unique(np.append(allcols, planned_exams_timeonly[:,0])))
    full_matrix = exam_matrix_binary[allcols]
    exam_names = full_matrix.columns
    # get a group to row dict to remap the planned exams
    group_to_row =  get_group_to_row_dict(full_matrix)
    for exam in range(len(planned_exams)):
        planned_exams[exam][0] = group_to_row[planned_exams[exam][0]]
    for exam in range(len(planned_exams_timeonly)):
        planned_exams_timeonly[exam][0] = group_to_row[planned_exams_timeonly[exam][0]]
    reduced_matrix, rowweights = unique_matrix(full_matrix)
    reduced_choice = get_choices_from_matrix(reduced_matrix)
    N_EXAMS = len(reduced_matrix[0])
    N_STUDENTS = len(reduced_matrix)
    df_exam, Week_Count = solveExamSchedule_days(config_object, reduced_choice, N_EXAMS, N_STUDENTS, rowweights, MaxCapacity, planned_exams.tolist(), planned_exams_timeonly.tolist())
    df_exam = df_exam.sort_values(by=['Exam_ID'])
    df_exam['Exam_ID']= exam_names
    #print(df_exam)
    #print("Double Bookings and weekly excess booking score")
    #print(Week_Count)
    return Week_Count, df_exam

def run_solver_weeks(config_object, exam_matrix_binary, exams_to_plan, planned_exams=[]):
    '''
    runs the day optimisation process to distribute exams evenly on weeks

    arguments:
        config_object: tuple of (N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR)
        exam_matrix_binary: binary exam matrix
        exams_to_plan: exams to plan
        planned_exams: list of exams with fixed times, in this case weeks [['G001', 1],  ...]
    returns:
        dataframe with exam times
    '''

    # reduce matrix to contain only semester relevant exams (and lower)
    #newcols = np.append(planned_exams[:,0], exams_to_plan)
    if len(planned_exams) > 0:
        allcols = np.sort(np.unique(np.append(planned_exams[:,0], exams_to_plan)))
    else:
        allcols=np.sort(exams_to_plan)
    full_matrix = exam_matrix_binary[allcols]
    exam_names = full_matrix.columns
    # get a group to row dict to remap the planned exams
    group_to_row =  get_group_to_row_dict(full_matrix)
    for exam in range(len(planned_exams)):
        planned_exams[exam][0] = group_to_row[planned_exams[exam][0]]
    reduced_matrix, rowweights = unique_matrix(full_matrix)
    reduced_choice_1 = get_choices_from_matrix(reduced_matrix)
    N_EXAMS = len(reduced_matrix[0])
    N_STUDENTS = len(reduced_matrix)
    df_exam, Week_Count = solveExamSchedule_weeks(config_object, reduced_choice_1, N_EXAMS, N_STUDENTS, rowweights, planned_exams)
    df_exam = df_exam.sort_values(by=['Exam_ID'])
    #exam_names= np.sort(exam_names)
    df_exam['Exam_ID']= exam_names
    #print(df_exam)
    print("Weekly excess score: " + str(Week_Count))
    return df_exam

def run_solver_exams(config_object, exam_matrix_binary, exams_to_plan, MaxCapacity, planned_exams=[]):
    '''
    runs the optimisation process to find the best exam schedule for the compelte period of time

    arguments:
        config_object: tuple of (N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR)
        exam_matrix_binary: binary exam matrix
        exams_to_plan: exams to plan
        MaxCapacity: List of slot capacities
        planned_exams: list of exams with fixed times [['G001', 1, 2 ,3],  ...]
    returns:
        dataframe with exam times
    '''

    # reduce matrix to contain only semester relevant exams (and lower)
    if len(planned_exams) > 0:
        allcols = np.sort(np.unique(np.append(planned_exams[:,0], exams_to_plan)))
    else:
        allcols=np.sort(exams_to_plan)
    full_matrix = exam_matrix_binary[allcols]
    #reduced_matrix = exam_matrix_binary[exams_to_plan]
    #row_ids = get_necessary_rows(reduced_matrix)
    exam_names = full_matrix.columns
    # get a group to row dict to remap the planned exams
    group_to_row =  get_group_to_row_dict(full_matrix)
    for exam in range(len(planned_exams)):
        planned_exams[exam][0] = group_to_row[planned_exams[exam][0]]
    #full_matrix = full_matrix.iloc[row_ids]
    reduced_matrix, rowweights = unique_matrix(full_matrix)
    reduced_choice_1 = get_choices_from_matrix(reduced_matrix)
    N_EXAMS = len(reduced_matrix[0])
    N_STUDENTS = len(reduced_matrix)
    df_exam ,doubleBookingcount = solveExamSchedule(config_object, reduced_choice_1, N_EXAMS, N_STUDENTS, rowweights, MaxCapacity, planned_exams)
    #print(df_student)    
    #print(df_exam)

    df_exam = df_exam.sort_values(by=['Exam_ID'])
    #exam_names= np.sort(exam_names)
    df_exam['Exam_ID']= exam_names
    print(df_exam)
    print("Double Bookings and weekly excess booking score")
    print(doubleBookingcount)
    df_exam = df_exam.values
    return doubleBookingcount, df_exam

def plan_weeks(config_object, exam_matrix_binary, MaxCapacity, preplanned_exams=[], planned_exams=[], exam_planned_timeonly=[]):
    '''
    runs the optimisation process to find the best exam schedule for the compelte period of time

    arguments:
        config_object: tuple of (N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR)
        exam_matrix_binary: binary exam matrix
        preplanned_exams: list of exams with fixed times [['G001', 1, 2 ,3],  ...]
        planned_exams: list of exams with planned weeks [['G001', 1],  ...]
        planned_exams_timeonly: list of exams with fixed slots [['G013', 1], ['G160', 1], ...]
        MaxCapacity: List of slot capacities
    returns:
        dataframe with exam times
    '''
    N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR = config_object
    weekplan = None
    score_arr = []
    for iweek in range(N_WEEKS):
        exams_to_plan = planned_exams[planned_exams['Week'] == iweek]['Exam_ID']
        rest_matrix = exam_matrix_binary[exams_to_plan]
        order = pd.DataFrame(rest_matrix.sum())
        order.columns=["N_students"]
        order=order.sort_values(by=['N_students'])
        exams_to_plan=order.index.values
        timed_and_big = []
        timed_and_small = []
        for timed_exam in exam_planned_timeonly:
            if timed_exam[0] in exams_to_plan[-9:]:
                timed_and_big.append(timed_exam)
            elif timed_exam[0] in exams_to_plan[:-9]: 
                timed_and_small.append(timed_exam)
        planned_exams_days=[]
        for exam in preplanned_exams:
            if iweek == exam[1]:
                planned_exams_days.append([exam[0], exam[2], exam[3]])
        score1, weekplan_1 = run_solver_days(config_object, exam_matrix_binary, exams_to_plan[-9:], iweek, MaxCapacity, np.array(planned_exams_days),np.array(timed_and_big))
        score1, weekplan_1 = run_solver_days(config_object, exam_matrix_binary, exams_to_plan[:-9], iweek, MaxCapacity, weekplan_1.values, np.array(timed_and_small))
        weekplan_1.insert(1, 'Week', iweek, True)
        try:
            weekplan = weekplan.append(weekplan_1)
        except:
            weekplan = weekplan_1
        score_arr.append(score1)
        print("Score for week", iweek, ":" ,score1)
    print("Total Score: ",sum(score_arr))
    df_exam = weekplan.sort_values(by=['Exam_ID'])
    return df_exam

def solve_clusterrecursive(exam_matrix_binary, planned_exams, ShowPlots):
    '''
    Do not use without checking code. This was used in a previous version of the jupyter notebook and probably won't work without adjustment.

    Runs the day optimisation process to find the best exam schedule for the compelte period of time, while using clustering.
    Exams-Matrix is clustered into two groups. The smaller cluster gets planned. Process is repeated with bigger cluster until clusters have less than 3 exams.

    arguments:
        exam_matrix_binary: binary exam matrix
        exams_to_plan: exams to plan
        ShowPlots [boolean]: if true matrix clusters for each step are plottet.
    returns:
        dataframe with exam times
    '''
    clustersize = 3
    #cluster data if more than 2 exams in smallest cluster
    while clustersize > 2:
        rest_matrix = exam_matrix_binary.drop(planned_exams[:,0], axis = 1)
        model = SpectralCoclustering(n_clusters=2, random_state=0)
        model.fit(rest_matrix+1)
        fit_data = rest_matrix.values[np.argsort(model.row_labels_)]
        fit_data = fit_data[:, np.argsort(model.column_labels_)]
        Group_names_remaining = rest_matrix.columns
        if ShowPlots:
            binary = fit_data.T > 0
            fig, ax =plt.subplots(figsize=(18, 16))
            im = ax.imshow(binary, aspect=15)

            ax.set_yticks(np.arange(len(Group_names_remaining)))
            ax.set_yticklabels(Group_names_remaining[np.argsort(model.column_labels_)])

            plt.show()
        clusters, sizes = np.unique(model.column_labels_, return_counts=True)
        exam_clusters = [[clusters[i], sizes[i]] for i in range(len(clusters))]
        clusters_to_plan = []
        for i in exam_clusters:
            if i[1] != max(sizes):
                clusters_to_plan.append(i[0])
        clustersize = min(sizes)
        exams_to_plan = []
        for i in clusters_to_plan:
            exams_to_plan.append(Group_names_remaining[model.column_labels_==i])
        #print(exams_to_plan)
        for group in exams_to_plan:
            planned_exams = run_solver_exams(group, planned_exams)
        print(clustersize)
    else:
        return planned_exams