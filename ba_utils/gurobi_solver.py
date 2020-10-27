'''
Gurobi utilities
'''

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

def solveExamSchedule(config_object, MaxCapaticy, exam_choice, N_EXAMS, N_STUDENTS, rowweights, exam_sizes, planned_exams =[], start =[]):
    '''
    Designed for internal usage within the module. Runs the actual solver for planning the entire exam schedule in one step.
    
    Gurobi Version with warm start and added weights to exclude similar solutions.

    arguments:
        config_object: tuple of (N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR)
        exam_choice: List containing different exam choices of students [[1,3,7],[2,5,6]...]
        N_EXAMS: Number of exams to be planned
        N_STUDENTS: Number of Students with different exam combinations
        rowweights: weigthing each exam_choice by the amount of students with the combination
        planned_exams: optional dataframe for preplanned exams
        start: Exam times for warm start
    '''
    # constants:
    N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR = config_object
    #DAILY_EXAM_STRESS_ALLOWED = 2
    n_days = N_WEEKS*N_DAYS_PER_WEEK
    m = gp.Model("Exam Scheduling")

    #exam_choice look like [[1,3,7],[2,5,6]...]
    #new decision variable: placement of exams
    exams = range(N_EXAMS)
    slots = range(N_SLOTS)
    days = range(N_DAYS_PER_WEEK)
    weeks = range(N_WEEKS)
    students = range(N_STUDENTS)

    exam_placing = m.addVars(weeks,days,slots,exams,
                 vtype=GRB.BINARY,
                 name="exam_placing")


    #evaluation variable (pseudo decision): Timetable for students
    student_schedule = m.addVars(students,weeks,days,slots,exams,
                 vtype=GRB.BINARY,
                 name="student_schedule")

    #fill timetable accoring to exams
    for student in range(N_STUDENTS):
        for week in range(N_WEEKS):
            for day in range(N_DAYS_PER_WEEK):
                for slot in range(N_SLOTS):
                    for exam in exam_choice[student]:
                        student_schedule[student,week,day,slot,exam] = exam_placing[week,day,slot,exam]

    slot_occupancy = [[[gp.quicksum([student_schedule[student,week,day,slot,exam]*rowweights[student] for exam in range(N_EXAMS) for student in range(N_STUDENTS)])
                    for slot in range(N_SLOTS)]
                    for day in range(N_DAYS_PER_WEEK)]
                    for week in range(N_WEEKS)]
    
    #alternative model
    combination = m.addVars(students,weeks,days[:-1],range(3),range(3),
                 vtype=GRB.BINARY,
                 name="normal_booked_days")

    '''helper variables'''
    double_booked_days = m.addVars(students,weeks,days,
                 vtype=GRB.INTEGER,
                 name="double_booked_days")
    normal_booked_days = m.addVars(students,weeks,days,
                 vtype=GRB.BINARY,
                 name="normal_booked_days")
    consecutive_days_a = m.addVars(students,weeks,days[:-1],
                 vtype=GRB.BINARY,
                 name="consecutive_days_a")
    consecutive_days_b = m.addVars(students,weeks,days[:-1],
                 vtype=GRB.INTEGER,
                 name="consecutive_days_b")
    '''Constraints'''

    '''OPTIMIZE DAYS'''
    # differentiate between normal exam stress and extra exam stress
    for week in range(N_WEEKS):
        for day in range(N_DAYS_PER_WEEK):
            for student in range(N_STUDENTS):
                m.addConstr((double_booked_days[student,week,day] + normal_booked_days[student,week,day] == gp.quicksum([student_schedule[student,week,day,slot,exam] for exam in range(N_EXAMS) for slot in range(N_SLOTS)])))
                m.addConstr((normal_booked_days[student,week,day] <= 1))

    '''OPTIMIZE CONSECUTIVE DAYS'''
    for week in range(N_WEEKS):
        for day in range(N_DAYS_PER_WEEK-1):
            for student in range(N_STUDENTS):
                m.addConstr((consecutive_days_a[student,week,day] + consecutive_days_b[student,week,day] == 
                             normal_booked_days[student,week,day] + normal_booked_days[student,week,day+1]))

    '''Other constraints'''
    #limit number of students to capacity
    for week in range(N_WEEKS):
        for day in range(N_DAYS_PER_WEEK):
            for slot in range(N_SLOTS):
                m.addConstr((slot_occupancy[week][day][slot] <= MaxCapaticy[slot]))

    # every exam gets exactly one slot
    for exam in range(N_EXAMS):
        m.addConstr((gp.quicksum([exam_placing[week,day,slot,exam] for slot in range(N_SLOTS) for day in range(N_DAYS_PER_WEEK) for week in range(N_WEEKS)]) == 1))

    # every Student a maximum of one exam per slot
    for student in range(N_STUDENTS):
        for week in range(N_WEEKS):
            for day in range(N_DAYS_PER_WEEK):
                for slot in range(N_SLOTS):
                    m.addConstr((gp.quicksum([student_schedule[student,week,day,slot,exam] for exam in range(N_EXAMS)]) <=1))
    
    #every student writes as much exams as chosen
    for student in range(N_STUDENTS):
        m.addConstr((len(exam_choice[student]) == gp.quicksum([student_schedule[student,week,day,slot,exam]
            for exam in range(N_EXAMS) 
            for slot in range(N_SLOTS)
            for day in range(N_DAYS_PER_WEEK) 
            for week in range(N_WEEKS)]) 
            ))
    '''objective/scoring'''
    daily_double_count = gp.quicksum([double_booked_days[student,week,day]*rowweights[student] 
        for day in range(N_DAYS_PER_WEEK)
        for week in range(N_WEEKS) 
        for student in range(N_STUDENTS)])
    consecutive_count = gp.quicksum([consecutive_days_b[student,week,day]*rowweights[student] 
        for day in range(N_DAYS_PER_WEEK-1)
        for week in range(N_WEEKS) 
        for student in range(N_STUDENTS)])
    # weights for improved runtime
    weight_day = [(n_days - d)/n_days for d in range(n_days)]
    weight_slot = [0.6, 1, 0.8, 0.4, 0.2]
    f_time = gp.quicksum(exam_placing[week,day,slot,exam] * (weight_day[day+week*FACTOR]+weight_slot[slot]) * exam_sizes[exam] 
            for exam in range(N_EXAMS) 
            for slot in range(N_SLOTS)
            for day in range(N_DAYS_PER_WEEK) 
            for week in range(N_WEEKS)) 

    exam_cost = daily_double_count*5 + consecutive_count - f_time
    m.ModelSense = GRB.MINIMIZE
    m.setObjective(exam_cost)
    #m.setObjective(combination_cost_total)
    # initialise all start values
    for week in range(N_WEEKS):
        for day in range(N_DAYS_PER_WEEK):
            for slot in range(N_SLOTS):
                for exam in range(N_EXAMS):
                    exam_placing[week,day,slot,exam].start = 0
    
         
    # set startvalues for good solution
    for i_exam in start:
        exam = i_exam[0]
        week = i_exam[1]
        day = i_exam[2]
        slot = i_exam[3]
        exam_placing[week,day,slot,exam].start = 1

    for planned in planned_exams:
        exam = planned[0]
        week = planned[1]
        day = planned[2]
        slot = planned[3]
        m.addConstr((exam_placing[week,day,slot,exam] == 1))

    #set startvalues for every variable!

    m.optimize()

    print('TOTAL COST: %g' % m.objVal)

    l_exam = [(exam, week, day, slot) 
        for day in range(N_DAYS_PER_WEEK) 
        for week in range(N_WEEKS) 
        for exam in range(N_EXAMS) 
        for slot in range(N_SLOTS) if exam_placing[week,day,slot,exam].x>0.99]

    df_exam = pd.DataFrame(l_exam, columns=['Exam_ID','Week','Day','Slot'])
    return df_exam, m.objVal

def solveExamSchedule_noslot(config_object, exam_choice, N_EXAMS, N_STUDENTS, rowweights, planned_exams =[], start =[]):
    '''
    Designed for internal usage within the module. Runs the actual solver for planning the entire exam schedule in one step.
    Distributes Exams onto days but doesn't plan slots.

    Gurobi Version with warm start and added weights to exclude similar solutions.

    arguments:
        config_object: tuple of (N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR)
        exam_choice: List containing different exam choices of students [[1,3,7],[2,5,6]...]
        N_EXAMS: Number of exams to be planned
        N_STUDENTS: Number of Students with different exam combinations
        rowweights: weigthing each exam_choice by the amount of students with the combination
        planned_exams: optional dataframe for preplanned exams
        start: Exam times for warm start
    '''

    # constants:
    N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR = config_object
    m = gp.Model("Exam Scheduling")

    #exam_choice look like [[1,3,7],[2,5,6]...]
    #new decision variable: placement of exams
    exams = range(N_EXAMS)
    days = range(N_DAYS_PER_WEEK)
    weeks = range(N_WEEKS)
    students = range(N_STUDENTS)

    exam_placing = m.addVars(weeks,days,exams,
                 vtype=GRB.BINARY,
                 name="exam_placing")
    #evaluation variable (pseudo decision): Timetable for students
    student_schedule = m.addVars(students,weeks,days,exams,
                 vtype=GRB.BINARY,
                 name="student_schedule")

    #fill timetable accoring to exams
    for student in range(N_STUDENTS):
        for week in range(N_WEEKS):
            for day in range(N_DAYS_PER_WEEK):
                for exam in exam_choice[student]:
                    student_schedule[student,week,day,exam] = exam_placing[week,day,exam]
    
    '''helper variables'''
    double_booked_days = m.addVars(students,weeks,days,
                 vtype=GRB.INTEGER,
                 name="double_booked_days")
    normal_booked_days = m.addVars(students,weeks,days,
                 vtype=GRB.BINARY,
                 name="normal_booked_days")
    consecutive_days_a = m.addVars(students,weeks,days[:-1],
                 vtype=GRB.BINARY,
                 name="consecutive_days_a")
    consecutive_days_b = m.addVars(students,weeks,days[:-1],
                 vtype=GRB.INTEGER,
                 name="consecutive_days_b")
    '''Constraints'''

    '''OPTIMIZE DAYS'''
    # differentiate between normal exam stress and extra exam stress
    for week in range(N_WEEKS):
        for day in range(N_DAYS_PER_WEEK):
            for student in range(N_STUDENTS):
                m.addConstr((double_booked_days[student,week,day] + normal_booked_days[student,week,day] == gp.quicksum([student_schedule[student,week,day,exam] for exam in range(N_EXAMS)])))
                m.addConstr((normal_booked_days[student,week,day] <= 1))
                m.addConstr((double_booked_days[student,week,day] <= DAILY_EXAM_STRESS_ALLOWED))
    '''OPTIMIZE CONSECUTIVE DAYS'''
    for week in range(N_WEEKS):
        for day in range(N_DAYS_PER_WEEK-1):
            for student in range(N_STUDENTS):
                m.addConstr((consecutive_days_a[student,week,day] + consecutive_days_b[student,week,day] == 
                             normal_booked_days[student,week,day] + normal_booked_days[student,week,day+1]))

    '''Other constraints'''

    # every exam gets exactly one slot
    for exam in range(N_EXAMS):
        m.addConstr((gp.quicksum([exam_placing[week,day,exam] for day in range(N_DAYS_PER_WEEK) for week in range(N_WEEKS)]) == 1))

    # every Student a maximum of one two exams per day
    for student in range(N_STUDENTS):
        for week in range(N_WEEKS):
            for day in range(N_DAYS_PER_WEEK):
                m.addConstr((gp.quicksum([student_schedule[student,week,day,exam] for exam in range(N_EXAMS)]) <=2))
    
    #every student writes as much exams as chosen
    for student in range(N_STUDENTS):
        m.addConstr((len(exam_choice[student]) == gp.quicksum([student_schedule[student,week,day,exam]
            for exam in range(N_EXAMS) 
            for day in range(N_DAYS_PER_WEEK) 
            for week in range(N_WEEKS)]) 
            ))
    '''objective/scoring'''
    daily_double_count = gp.quicksum([double_booked_days[student,week,day]*rowweights[student] 
        for day in range(N_DAYS_PER_WEEK)
        for week in range(N_WEEKS) 
        for student in range(N_STUDENTS)])
    consecutive_count = gp.quicksum([consecutive_days_b[student,week,day]*rowweights[student] 
        for day in range(N_DAYS_PER_WEEK-1)
        for week in range(N_WEEKS) 
        for student in range(N_STUDENTS)])

    exam_cost = daily_double_count*FACTOR + consecutive_count
    m.ModelSense = GRB.MINIMIZE
    m.setObjective(exam_cost)
    #m.setObjective(combination_cost_total)
    # initialise all start values
    for week in range(N_WEEKS):
        for day in range(N_DAYS_PER_WEEK):
            for exam in range(N_EXAMS):
                exam_placing[week,day,exam].start = 0
        
    # set startvalues for good solution
    for i_exam in start:
        exam = i_exam[0]
        week = i_exam[1]
        day = i_exam[2]
        exam_placing[week,day,exam].start = 1

    for planned in planned_exams:
        exam = planned[0]
        week = planned[1]
        day = planned[2]
        m.addConstr((exam_placing[week,day,exam] == 1))

    #set startvalues for every variable!
    
    # work agressively on cuts
    m.optimize()

    print('TOTAL COST: %g' % m.objVal)

    l_exam = [(exam, week, day) 
        for day in range(N_DAYS_PER_WEEK) 
        for week in range(N_WEEKS) 
        for exam in range(N_EXAMS) if exam_placing[week,day,exam].x>0.99]

    df_exam = pd.DataFrame(l_exam, columns=['Exam_ID','Week','Day'])
    return df_exam, m.objVal

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

def get_group_to_row_dict(matrix):
    '''
    maps exams columns to their respective rows in the matrix

    arguments:
        exam_matrix_binary: binary matrix as dataframe
    returns:
        dictionary containing the row in the matrix for each indivicual exam
    '''
    group_to_row={}
    n=0
    for row in matrix.columns:
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

def run_solver_exams(config_object, MaxCapaticy, exam_matrix_binary, exam_sizes, start =[],  planned_exams=[]):
    '''
    runs the optimisation process to find the best exam schedule for the compelte period of time

    Gurobi Version with warm start and added weights to exclude similar solutions.

    arguments:
        config_object: tuple of (N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR)
        exam_matrix_binary: binary exam matrix
        exams_to_plan: exams to plan
        start: start values for warm start
        planned_exams: list of exams with fixed times [['G001', 1, 2 ,3],  ...]
    returns:
        dataframe with exam times
    '''
    row_ids = get_necessary_rows(exam_matrix_binary)
    exam_names = exam_matrix_binary.columns
    # get a group to row dict to remap the planned exams
    group_to_row =  get_group_to_row_dict(exam_matrix_binary)
    for exam in range(len(planned_exams)):
       planned_exams[exam][0] = group_to_row[planned_exams[exam][0]]
    for exam in range(len(start)):
       start[exam][0] = group_to_row[start[exam][0]]
    full_matrix = exam_matrix_binary.iloc[row_ids]
    reduced_matrix, rowweights = unique_matrix(full_matrix)
    reduced_choice_1 = get_choices_from_matrix(reduced_matrix)
    N_EXAMS = len(reduced_matrix[0])
    N_STUDENTS = len(reduced_matrix)
    df_exam ,doubleBookingcount = solveExamSchedule(config_object, MaxCapaticy, reduced_choice_1, N_EXAMS, N_STUDENTS, rowweights, exam_sizes, planned_exams, start)
    df_exam = df_exam.sort_values(by=['Exam_ID'])
    #exam_names= np.sort(exam_names)
    df_exam['Exam_ID']= exam_names
    print(df_exam)
    print("Double Bookings and weekly excess booking score")
    print(doubleBookingcount)
    df_exam = df_exam.values
    return doubleBookingcount, df_exam

def run_solver_exams_noslot(config_object, exam_matrix_binary, start =[], planned_exams=[]):
    '''
    Runs the optimisation process to find the best exam schedule for the complete period of time
    Distributes Exams onto days but doesn't plan slots.

    Gurobi Version with warm start and added weights to exclude similar solutions.

    arguments:
        config_object: tuple of (N_WEEKS, N_DAYS_PER_WEEK, N_SLOTS, DAILY_EXAM_STRESS_ALLOWED, WEEKLY_EXAM_STRESS_ALLOWED, FACTOR)
        exam_matrix_binary: binary exam matrix
        exams_to_plan: exams to plan
        start: start values for warm start
        planned_exams: list of exams with fixed times [['G001', 1, 2 ,3],  ...]
    returns:
        dataframe with exam times
    '''

    row_ids = get_necessary_rows(exam_matrix_binary)
    exam_names = exam_matrix_binary.columns
    # get a group to row dict to remap the planned exams
    group_to_row =  get_group_to_row_dict(exam_matrix_binary)
    for exam in range(len(planned_exams)):
       planned_exams[exam][0] = group_to_row[planned_exams[exam][0]]
    for exam in range(len(start)):
       start[exam][0] = group_to_row[start[exam][0]]
    full_matrix = exam_matrix_binary.iloc[row_ids]
    reduced_matrix, rowweights = unique_matrix(full_matrix)
    reduced_choice_1 = get_choices_from_matrix(reduced_matrix)
    N_EXAMS = len(reduced_matrix[0])
    N_STUDENTS = len(reduced_matrix)
    df_exam ,doubleBookingcount = solveExamSchedule_noslot(config_object, reduced_choice_1, N_EXAMS, N_STUDENTS, rowweights, planned_exams, start)
    df_exam = df_exam.sort_values(by=['Exam_ID'])
    #exam_names= np.sort(exam_names)
    df_exam['Exam_ID']= exam_names
    print(df_exam)
    print("Double Bookings and weekly excess booking score")
    print(doubleBookingcount)
    df_exam = df_exam.values
    return doubleBookingcount, df_exam