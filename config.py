"""
Configuration file for setting experiments parameters
"""
A = "./test_cases/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
B = "./test_cases/taskset__1643188120-a_0.1-b_0.7-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__6__tsk.csv"
C = "./test_cases/taskset__1643188302-a_0.3-b_0.3-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__36__tsk.csv"
D = "./test_cases/taskset__1643188521-a_0.5-b_0.4-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__18__tsk.csv"
E = "./test_cases/taskset__1643188594-a_0.7-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__7__tsk.csv"

test_case_path = A

profiling = False

algorithm = "SA"

show_plot = False

write_log = True
log_directory = "./results/"
log_name = "GA_tournament_A"

SA = dict(
    numinstances=1,
    numworkers=1,
    maxiter=10,
    toll=0.01,
    convergence=0.1,
    extra_ps=0,  # "random",
    wandblogging=False,
    iterationPerTemp=50,
    initialTemp=2,
    finalTemp=0.0001,
    tempReduction="geometric",
    alpha=0.4,
    beta=5,
    dur_radius=200,
    dln_radius=200,
    priority_prob=0,
    free_tasks_switches=3,
    no_upper_lim=False,
    temptune=False,
)

GA = dict(
    numinstances=1,
    numworkers=1,
    maxiter=10,
    toll=0.01,
    convergence=0.1,
    wandblogging=False,
    pop_size=16,
    num_parents=4,
    p_cross=0.8,
    p_mut=0.1,
    selection="tournament",
    free_tasks_switches=1,
)
