# Available params:
# seglen, CV, mode, oversample, domains, scheme

# 1. Determine best classifier for SAM and HAM
# --------------------------------------------
# python3 -m scripts train --seglens 10 --labeling-scheme ham --cv logo  # Using LOGO, because it's the main focus 
# python3 -m scripts train --seglens 10 --labeling-scheme sam --cv logo

# python3 -m scripts train --labeling-scheme ham --cv logo --seglens 30 
# python3 -m scripts train --labeling-scheme ham --cv logo --seglens 15
# python3 -m scripts train --labeling-scheme ham --cv logo --seglens 10
# python3 -m scripts train --labeling-scheme ham --cv logo --seglens 5 
# python3 -m scripts train --labeling-scheme ham --cv logo --seglens 3 
# python3 -m scripts train --labeling-scheme ham --cv logo --seglens 2 
# python3 -m scripts train --labeling-scheme ham --cv logo --seglens 1 --classifiers svm-poly,rf,knn,mlp

# Evaluate all classifiers on seglen 15 and find best number of features
# python3 -m scripts train --labeling-scheme ham --cv logo --seglens 15 --classifiers svm-lin,svm-rbf,svm-poly,rf,knn,mlp

python3 -m scripts train --labeling-scheme ham --cv skf --seglens 15 --classifiers svm-lin,svm-rbf,svm-poly,rf,knn,mlp