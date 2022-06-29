import os

GUROBI_HOME = "/usr/local/gurobi910/linux64"
LD_LIBRARY_PATH=f"{GUROBI_HOME}/lib"
if "LD_LIBRARY_PATH" in os.environ:
    LD_LIBRARY_PATH=os.environ+":"+LD_LIBRARY_PATH

c.Spawner.environment = {
	'GUROBI_HOME':GUROBI_HOME,
        'PATH':os.environ['PATH']+f":{GUROBI_HOME}/bin",
        'LD_LIBRARY_PATH':LD_LIBRARY_PATH
}
