# Comparing accuracy of all classifiers on DASPS

## Time features

```
12 seconds    0.673703
10 seconds    0.673147
15 seconds    0.670305
8 seconds     0.656602
6 seconds     0.646688
1 second      0.643648
2 seconds     0.641091
4 seconds     0.635542
```

12 seconds is the optimal time window for time domain features

### Mean accuracy

```
RF                            0.733653
LGBMClassifier                0.718835
GradientBoostingClassifier    0.713315
XGBClassifier                 0.713265
MLP                           0.695511
SVC_linear                    0.674252
SVC_rbf                       0.654719
LDA                           0.653528
AdaBoostClassifier            0.626001
KNN                           0.622574
SVC_poly                      0.621742
BernoulliNB                   0.561848
SVC_sigmoid                   0.526937
```

### 12 seconds

```
LGBMClassifier                0.753205
GradientBoostingClassifier    0.750509
RF                            0.745028
XGBClassifier                 0.744926
MLP                           0.720658
SVC_linear                    0.702919
SVC_rbf                       0.683062
LDA                           0.670102
KNN                           0.649959
AdaBoostClassifier            0.638121
SVC_poly                      0.604199
BernoulliNB                   0.577532
SVC_sigmoid                   0.517916
```

### Full table

```
                            1 second  2 seconds  4 seconds  6 seconds  8 seconds  10 seconds  12 seconds  15 seconds
BernoulliNB                 0.526651   0.534240   0.557624   0.567947   0.573354    0.592366    0.577532    0.565074
XGBClassifier               0.695499   0.714127   0.681639   0.705591   0.704291    0.732535    0.744926    0.727517
AdaBoostClassifier          0.620466   0.629483   0.593881   0.592253   0.637798    0.642824    0.638121    0.653185
MLP                         0.664188   0.649037   0.680773   0.696188   0.713436    0.736431    0.720658    0.703374
RF                          0.725100   0.727532   0.707335   0.727596   0.739613    0.742977    0.745028    0.754042
KNN                         0.611900   0.618218   0.611495   0.595279   0.604265    0.637163    0.649959    0.652314
LDA                         0.641070   0.616303   0.655095   0.649411   0.655095    0.678157    0.670102    0.662990
SVC_poly                    0.651971   0.617261   0.624060   0.619887   0.630205    0.615115    0.604199    0.611234
SVC_linear                  0.648152   0.628198   0.685873   0.681454   0.656390    0.690773    0.702919    0.700261
SVC_sigmoid                 0.555566   0.558018   0.500671   0.513794   0.511234    0.539135    0.517916    0.519165
SVC_rbf                     0.638449   0.626928   0.603871   0.648705   0.679258    0.688259    0.683062    0.669222
GradientBoostingClassifier  0.687468   0.703318   0.684724   0.698605   0.711971    0.722757    0.750509    0.747166
LGBMClassifier              0.700942   0.711526   0.675008   0.710236   0.718920    0.732422    0.753205    0.748423
```

## Relative power

### Mean accuracy

```
RF                            0.719391
XGBClassifier                 0.711293
LGBMClassifier                0.708830
GradientBoostingClassifier    0.708576
MLP                           0.678412
KNN                           0.619578
SVC_rbf                       0.616273
LDA                           0.609678
AdaBoostClassifier            0.603421
SVC_poly                      0.601937
SVC_linear                    0.598831
BernoulliNB                   0.566822
SVC_sigmoid                   0.537394
dtype: float64

```

### By durations

```
8 seconds     0.661700
1 second      0.640565
10 seconds    0.639855
12 seconds    0.637506
6 seconds     0.635825
15 seconds    0.635196
2 seconds     0.624273
4 seconds     0.620733
```

> No pattern

### 8 seconds (best accuracy)

```
XGBClassifier                 0.754849
RF                            0.746295
LGBMClassifier                0.743303
GradientBoostingClassifier    0.737934
MLP                           0.703735
AdaBoostClassifier            0.653610
SVC_linear                    0.652842
LDA                           0.651854
SVC_rbf                       0.636262
KNN                           0.626631
SVC_poly                      0.592601
BernoulliNB                   0.578167
SVC_sigmoid                   0.524014
```

### Full table

```
                            1 second  2 seconds  4 seconds  6 seconds  8 seconds  10 seconds  12 seconds  15 seconds
BernoulliNB                 0.605458   0.534649   0.542258   0.562770   0.578167    0.576278    0.567005    0.567988
XGBClassifier               0.723712   0.685617   0.699416   0.710860   0.754849    0.696564    0.709667    0.709657
AdaBoostClassifier          0.596278   0.596155   0.552857   0.568643   0.653610    0.611336    0.625896    0.622591
MLP                         0.672320   0.682138   0.673052   0.695648   0.703735    0.672476    0.666536    0.661395
RF                          0.711580   0.703141   0.719695   0.731569   0.746295    0.711065    0.722314    0.709470
KNN                         0.594491   0.603364   0.646211   0.601265   0.626631    0.634066    0.639130    0.611470
LDA                         0.594030   0.619642   0.585248   0.631290   0.651854    0.613835    0.592422    0.589104
SVC_poly                    0.622145   0.621505   0.560783   0.584562   0.592601    0.591029    0.614347    0.628520
SVC_linear                  0.593671   0.606841   0.577896   0.614060   0.652842    0.613482    0.566078    0.565776
SVC_sigmoid                 0.527885   0.491316   0.531674   0.547742   0.524014    0.570230    0.556262    0.550031
SVC_rbf                     0.636528   0.614506   0.590988   0.610215   0.636262    0.613753    0.611864    0.616068
GradientBoostingClassifier  0.719119   0.681318   0.683344   0.706749   0.737934    0.706295    0.711531    0.722314
LGBMClassifier              0.730123   0.675356   0.706109   0.700353   0.743303    0.707701    0.704531    0.703164
```

## Absolute power

### Accuracy mean
```

RF                            0.769482
XGBClassifier                 0.748187
LGBMClassifier                0.747507
GradientBoostingClassifier    0.744827
MLP                           0.707579
KNN                           0.672603
AdaBoostClassifier            0.669676
LDA                           0.647699
SVC_linear                    0.629381
SVC_rbf                       0.626575
BernoulliNB                   0.591328
SVC_poly                      0.552840
SVC_sigmoid                   0.499126
```

### By duration

```
8 seconds     0.682712
10 seconds    0.674749
6 seconds     0.662973
12 seconds    0.660571
15 seconds    0.656183
4 seconds     0.635186
```

### 8 seconds duration

```
RF                            0.784547
LGBMClassifier                0.772412
GradientBoostingClassifier    0.764237
XGBClassifier                 0.761167
MLP                           0.716802
AdaBoostClassifier            0.701971
SVC_linear                    0.686733
LDA                           0.685177
KNN                           0.682386
SVC_rbf                       0.647107
BernoulliNB                   0.604537
SVC_poly                      0.544219
SVC_sigmoid                   0.523963
```

### Full table

```
                            4 seconds  6 seconds  8 seconds  10 seconds  12 seconds  15 seconds
BernoulliNB                  0.569590   0.592939   0.604537    0.614726    0.583267    0.582908
XGBClassifier                0.710589   0.762509   0.761167    0.760599    0.754199    0.740061
AdaBoostClassifier           0.639447   0.675668   0.701971    0.713733    0.646472    0.640763
MLP                          0.686869   0.708205   0.716802    0.705719    0.720809    0.707066
RF                           0.734503   0.783830   0.784547    0.778410    0.774076    0.761528
KNN                          0.619565   0.673666   0.682386    0.676068    0.687194    0.696738
LDA                          0.634629   0.628792   0.685177    0.636841    0.651838    0.648920
SVC_poly                     0.550005   0.543886   0.544219    0.578556    0.553564    0.546810
SVC_linear                   0.617890   0.617931   0.686733    0.627148    0.626871    0.599713
SVC_sigmoid                  0.478500   0.494839   0.523963    0.532709    0.471152    0.493594
SVC_rbf                      0.600655   0.616953   0.647107    0.638725    0.623369    0.632637
GradientBoostingClassifier   0.700699   0.764342   0.764237    0.751723    0.751943    0.736016
LGBMClassifier               0.714470   0.755090   0.772412    0.756779    0.742668    0.743625
```

## Asymmetry

## Mean accuracy

```

RF                            0.754697
GradientBoostingClassifier    0.743541
LGBMClassifier                0.739541
XGBClassifier                 0.739512
MLP                           0.701321
AdaBoostClassifier            0.695065
SVC_rbf                       0.674131
KNN                           0.656918
SVC_poly                      0.625584
SVC_linear                    0.619010
LDA                           0.609520
BernoulliNB                   0.602954
SVC_sigmoid                   0.576426
```

## By duration

```
8 seconds     0.700487
6 seconds     0.689620
10 seconds    0.674953
4 seconds     0.671184
12 seconds    0.648883
15 seconds    0.647898
```

## 8 seconds duration

```
RF                            0.778382
GradientBoostingClassifier    0.772916
XGBClassifier                 0.767860
LGBMClassifier                0.767604
AdaBoostClassifier            0.742622
MLP                           0.726480
SVC_rbf                       0.688141
SVC_linear                    0.675929
KNN                           0.665781
LDA                           0.656395
SVC_poly                      0.650635
BernoulliNB                   0.614183
SVC_sigmoid                   0.599401
```

### PLI

## Mean accuracies

```
RF                            0.731455
XGBClassifier                 0.710928
LGBMClassifier                0.703992
SVC_rbf                       0.698861
GradientBoostingClassifier    0.697793
MLP                           0.678760
SVC_linear                    0.678685
SVC_poly                      0.654310
LDA                           0.645680
AdaBoostClassifier            0.642297
BernoulliNB                   0.613175
SVC_sigmoid                   0.606982
KNN                           0.584634
```

## By duration

```
8 seconds     0.680775
4 seconds     0.675202
10 seconds    0.666829
6 seconds     0.665553
12 seconds    0.659293
15 seconds    0.643526
```

## Full table

```
                            4 seconds  6 seconds  8 seconds  10 seconds  12 seconds  15 seconds
BernoulliNB                  0.590410   0.612079   0.650445    0.638438    0.614357    0.573323
XGBClassifier                0.740323   0.711772   0.719488    0.693810    0.696119    0.704055
AdaBoostClassifier           0.645064   0.657563   0.646416    0.648065    0.631403    0.625274
MLP                          0.688287   0.676339   0.697527    0.684531    0.666621    0.659255
RF                           0.733093   0.730494   0.746418    0.721449    0.723379    0.733899
KNN                          0.590384   0.576288   0.581316    0.573425    0.590886    0.595504
LDA                          0.639339   0.634521   0.675248    0.668515    0.641971    0.614485
SVC_poly                     0.709642   0.686487   0.669145    0.635550    0.671510    0.553523
SVC_linear                   0.687481   0.693548   0.683251    0.700824    0.650302    0.656703
SVC_sigmoid                  0.580855   0.581976   0.647046    0.645212    0.604455    0.582345
SVC_rbf                      0.708935   0.695725   0.702468    0.707337    0.702801    0.675899
GradientBoostingClassifier   0.732158   0.694900   0.708582    0.668984    0.686339    0.695796
LGBMClassifier               0.731659   0.700502   0.722719    0.682637    0.690666    0.695771
```

## All features (PLI)

### Mean accuracy

```
RF                            0.765256
XGBClassifier                 0.751067
LGBMClassifier                0.749301
GradientBoostingClassifier    0.749180
AdaBoostClassifier            0.732456
SVC_rbf                       0.730450
MLP                           0.718298
SVC_linear                    0.710143
SVC_poly                      0.696348
LDA                           0.660367
BernoulliNB                   0.634212
SVC_sigmoid                   0.625791
KNN                           0.619761
```

### By duration

```
8 seconds     0.722357
10 seconds    0.711422
6 seconds     0.702117
12 seconds    0.700405
15 seconds    0.694162
4 seconds     0.689213
```

## 8 seconds

```
RF                            0.786354
GradientBoostingClassifier    0.764987
LGBMClassifier                0.761546
XGBClassifier                 0.756052
AdaBoostClassifier            0.752545
SVC_rbf                       0.746708
MLP                           0.739688
SVC_linear                    0.738116
SVC_poly                      0.685586
LDA                           0.678413
BernoulliNB                   0.663738
SVC_sigmoid                   0.661152
KNN                           0.655750
```

## Full table

```
                            4 seconds  6 seconds  8 seconds  10 seconds  12 seconds  15 seconds
BernoulliNB                  0.612785   0.637404   0.663738    0.661925    0.627506    0.601915
XGBClassifier                0.741608   0.763743   0.756052    0.740614    0.746713    0.757670
AdaBoostClassifier           0.710789   0.741546   0.752545    0.720650    0.729135    0.740072
MLP                          0.708966   0.717737   0.739688    0.736006    0.711528    0.695865
RF                           0.759562   0.768689   0.786354    0.764844    0.747102    0.764985
KNN                          0.604480   0.585842   0.655750    0.621818    0.633308    0.617368
LDA                          0.627911   0.661828   0.678413    0.672023    0.679488    0.642540
SVC_poly                     0.715371   0.694875   0.685586    0.696144    0.698024    0.688090
SVC_linear                   0.701685   0.701874   0.738116    0.731997    0.695509    0.691679
SVC_sigmoid                  0.571536   0.598705   0.661152    0.681726    0.620763    0.620865
SVC_rbf                      0.719519   0.729042   0.746708    0.750271    0.733932    0.703226
GradientBoostingClassifier   0.745302   0.765133   0.764987    0.739050    0.738697    0.741910
LGBMClassifier               0.740251   0.761101   0.761546    0.731423    0.743559    0.757926
```

## Conclusions

Time: 12, 10, 15 seconds
Relative power: 8 seconds
Absolute power: 8 seconds
Asymmetry: 8 seconds
PLI: 8 seconds
All features (PLI): 8 seconds

RF is consistently the best classifier

SVC_sigmoid is consistently doing bad