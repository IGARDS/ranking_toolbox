# Rankability Toolbox
This repo contains various implementations that provide insights into the rankability of data and the linear ordering problem.

# Install instructions
Add the following to /etc/profile
<pre>
export GUROBI_HOME="/usr/local/gurobi910/linux64" 
export PATH="${PATH}:${GUROBI_HOME}/bin" 
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib" 
</pre>

Copy environment.py to /opt/tljh/config/jupyterhub_config.d/environment.py

Run sudo tljh-config reload hub

sudo apt-get install libgraphviz-dev

## Notes on running tests
``bash
cd ranking_toolbox
python3 -m venv ../env
source ../env/bin/activate
cd tests
pytest tests.py
``

## Authors
Paul Anderson, Ph.D.<br>
Department of Computer Science<br>
Director, Data Science Program<br>
College of Charleston<br>

Amy Langville, Ph.D.<br>
Department of Mathematics<br>
College of Charleston<br>

Tim Chartier, Ph.D.<br>
Department of Mathematics<br>
Davidson College

## Acknowledgements
We would like to thank the entire IGARDS team for their invaluable insight and encouragement.

