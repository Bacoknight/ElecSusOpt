import chocolate as choco
import numpy as np

numIters = 2

def objective_function(condition, x=None, y=1):
    """An objective function returning ``1 - x`` when *condition* is 1 and 
    ``y - 6`` when *condition* is 2.
    
    Raises:
        ValueError: If condition is different than 1 or 2.
    """
    if condition == 1:
        return 1 - x
    elif condition == 2:
        return y - 6
    raise ValueError("condition must be 1 or 2, got {}.".format(condition))

# Define the conditional search space 
space = [
            {"condition": 1, "x": choco.uniform(low=1, high=10)},
            #{"condition": 2, "y": choco.log(low=-2, high=2, base=10)}
        ]

# Establish a connection to a SQLite local database
conn = choco.SQLiteConnection("sqlite:///my_db.db")

# Construct the optimizer
sampler = choco.Bayes(conn, space, utility_function = "ei", n_bootstrap = np.ceil(numIters/10), clear_db = True)

# Sample the next point
token, params = sampler.next()

# Calculate the loss for the sampled point (minimized)
loss = objective_function(**params)

# Add the loss to the database
sampler.update(token, loss)