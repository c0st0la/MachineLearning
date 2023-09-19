import matplotlib.pyplot as plt
import numpy
if __name__ == "__main__":
    applicationWorkingPoint1 = numpy.array([9 / 10, 1 / 10], dtype=float)
    applicationWorkingPoint2 = numpy.array([5 / 10, 5 / 10], dtype=float)
    applicationWorkingPoint3 = numpy.array([1 / 10, 9 / 10], dtype=float)
    dict1= {'10^- 5': 0.9802572591867087, '10^-4': 0.9565877051587186, '10^-3': 0.8563957815245571,
            '10^-2': 0.47249634619171044, '10^-1': 0.3549332625214561, '1': 0.3549332625214561,
            '10^1': 0.3549332625214561, '10^2': 0.3549332625214561, '10^3': 0.3549332625214561,
            '10^4': 0.3549332625214561, '10^5': 0.3549332625214561}

    dict2= {'10^- 5': 0.8641422924608131, '10^-4': 0.7739926503260429, '10^-3': 0.3421482957772074,
            '10^-2': 0.11618006879410814, '10^-1': 0.09683846596356113, '1': 0.09083310321862074,
            '10^1': 0.09083310321862074, '10^2': 0.09083310321862074, '10^3': 0.09083310321862074,
            '10^4': 0.09083310321862074, '10^5': 0.09083310321862074}

    dict3 ={'10^- 5': 1.0, '10^-4': 0.9875010857441335, '10^-3': 0.4779279460193863, '10^-2': 0.20799145120077397,
            '10^-1': 0.18153255544010677, '1': 0.16779518425961054, '10^1': 0.16599397508363795,
            '10^2': 0.16446725752638608, '10^3': 0.16446725752638608, '10^4': 0.16446725752638608,
            '10^5': 0.16446725752638608}



    plt.figure()
    #datiPolySVM_Raw_Pt0_1.txt
    plt.plot([var for var in list(dict1.keys())], [var for var in list(dict1.values())],
                 label=f"minDCF(effPrior={applicationWorkingPoint1[1]})")
    plt.plot([var for var in list(dict2.keys())], [var for var in list(dict2.values())],
                 label=f"minDCF(effPrior={applicationWorkingPoint2[1]})")
    plt.plot([var for var in list(dict3.keys())], [var for var in list(dict3.values())],
                 label=f"minDCF(effPrior={applicationWorkingPoint3[1]})")
    plt.tick_params(axis='x', labelsize=6)
    plt.legend()
    plt.savefig("./figures/PolySVM_Zscore_Pt0_9__DCFxC")
    plt.clf()