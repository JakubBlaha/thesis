# Available params:
# seglen, CV, mode, oversample, domains, scheme

# 1. Determine best classifier for SAM and HAM
# --------------------------------------------
python3 -m scripts train --seglens 10 --labeling-scheme ham --cv logo  # Using LOGO, because it's the main focus 
python3 -m scripts train --seglens 10 --labeling-scheme sam --cv logo

# Decision between using SAM and HAM for further analysis
# --------------------------------------------
# python3 -m scripts train --seglens 1,2,3,5,10,15,30 --labeling-scheme ham --cv logo
# python3 -m scripts train --seglens 1,2,3,5,10,15,30 --labeling-scheme sam --cv logo


# python3 -m scripts train --seglens 1,2,3,5,10,15,30 --labeling-scheme ham --cv skf