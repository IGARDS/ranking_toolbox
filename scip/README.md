# SCIP Helper Scripts

## Installation
These helper scripts rely on scip being installed and on the path. For information on installing:
https://www.scipopt.org/index.php#download

## Helper Script Verification
Example setup below. Your system may vary.
```
RANKING_TOOLBOX_DIR=$HOME/ranking_toolbox

SCIP_BIN_DIR=/usr/local/SCIPOptSuite/bin

export PATH=$PATH:$SCIP_BIN_DIR:$RANKING_TOOLBOX_DIR/scip

which scip

which scip_collect.sh

which scip_count.sh
```