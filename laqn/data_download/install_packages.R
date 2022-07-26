# Install the packages required to download the LAQN data through the openair libary: https://davidcarslaw.github.io/openair/

if("devtools" %in% rownames(installed.packages()) == FALSE) {install.packages("devtools", repos="http://cran.us.r-project.org")} 
if("openair" %in% rownames(installed.packages()) == FALSE) {devtools::install_github("davidcarslaw/openair")} 
if("hash" %in% rownames(installed.packages()) == FALSE) {install.packages("hash", repos="http://cran.us.r-project.org")} 