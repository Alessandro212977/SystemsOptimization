"""
Configuration file for setting experiments parameters
"""
A = "./test_cases/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
B = "./test_cases/taskset__1643188302-a_0.3-b_0.3-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__36__tsk.csv"
C = "./test_cases/taskset__1643188594-a_0.7-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__7__tsk.csv"
Small = "./test_cases/taskset_small.csv"

test_case_path = Small #Test case to optimize

profiling = False #Profile execution

algorithm = "SA" #Choose algorithm (SA or GA)

show_plot = True #show result plot after execution

write_log = True #write log file
log_directory = "./results/" #log file directory
log_name = "test" #run name

SA = dict(
    numinstances=1, #Number of runs
    numworkers=1, #Number of parallel execution workers
    maxiter=1000, #Maximum number of iterations
    toll=0.01, #Absolute tollerance
    convergence=0.1, #Convergence threshold
    extra_ps=0,  # "random", #Number of extra polling servers
    wandblogging=False, #Log data on wandb
    iterationPerTemp=50, #Number of iterations with the same temperature
    initialTemp=2, #Initial Tempreture
    finalTemp=0.0001, #Final Tempreture
    tempReduction="geometric", #Cooling schedule
    alpha=0.4, #Cooling parameter
    beta=5, #Cooling parameter
    dur_radius=200, #Radius for the budget neighbour in search space
    dln_radius=200, #Radius for the deadline neighbour in search space
    priority_prob=0, #probability of reassigning priorities (extension 2)
    free_tasks_switches=3, #Number of tasks switch at each iteration
    no_upper_lim=False, #Whether to set or not an upper limit of the number of polling servers
    temptune=False,
)

GA = dict(
    numinstances=1,
    numworkers=1,
    maxiter=10,
    toll=0.01,
    convergence=0.1,
    wandblogging=False,
    pop_size=16, #The size of population
    num_parents=4, #Number of parents
    p_cross=0.8, #The probability of having crossover
    p_mut=0.1, #The probability of having mutation
    selection="rank", #Selection of parents
    free_tasks_switches=1, #Number of tasks switch at each iteration
)
