import commands
import numpy as np

def oracle(x):
    x = list(x)
    query = "./Oracle.static " + ' '.join(map(lambda y:str(y), x))
    if query.strip() == 'Function is undefined at this point!' or query.strip() == 'inf':
        return None
    return np.array(x + [float(commands.getoutput(query))])