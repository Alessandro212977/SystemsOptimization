"""
Configuration file for setting experiments parameters
"""

test_case_path = "./test_cases/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
profiling = False

algorithm = "SA"

SA = dict(
    numinstances=1,
    numworkers=1,
    maxiter=100,
    toll=0.01,
    wandblogging=False,
    iterationPerTemp=100,
    initialTemp=0.1,
    finalTemp=0.0001,
    tempReduction="geometric",
    alpha=0.5,
    beta=5,
    temptune=False,
)

GA = dict(
    numinstances=1,
    numworkers=1,
    maxiter=10,
    toll=0.01,
    wandblogging=False,
    pop_size=16,
    num_parents=4,
    p_cross=0.8,
    p_mut=0.1,
)
