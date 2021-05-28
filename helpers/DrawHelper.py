import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def drawMatrix():
    cmap = ListedColormap(['white', 'blue', 'red', 'green'])

    matrix = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [400, 0, 400, 1, 0, 0, 1, 400, 0, 400, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [400, 0, 400, 1, 0, 0, 1, 400, 0, 400, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [400, 0, 400, 1, 0, 0, 1, 400, 0, 400, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [800, 0, 800, 1, 0, 0, 1, 800, 0, 800, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [800, 0, 800, 1, 0, 0, 1, 800, 0, 800, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [800, 0, 1200, 1, 0, 0, 1, 800, 0, 1200, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [800, 0, 800, 1, 0, 0, 1, 800, 0, 800, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [800, 0, 800, 1, 0, 0, 1, 800, 0, 800, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [400, 0, 400, 1, 0, 0, 1, 400, 0, 400, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [400, 0, 400, 1, 0, 0, 1, 400, 0, 400, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [400, 0, 400, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                       [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]])

    # dct = {0: -30., 1: 1}
    # matrix = np.array([[dct[i] for i in j] for j in matrix])
    # matrix[9,2] = 1200
    # matrix[9,9] = 1200
    # matrix[9,18] = 1200

    epoch_1 = [404, 1082, 417, 2433, 1899, 2240, 2918, 3695, 1178, 1540, 1447, 1346, 1194, 574, 1166, 2110, 3835, 906,
               534, 376,
               1559, 828, 639, 2738, 10177, 3845, 1219, 946, 187, 557, 946, 1032, 417, 256, 3648, 1999, 563, 1279, 42,
               4438, 1128,
               9815, 2197, 1629, 298, 968, 386, 3361, 198, 198, 1022, 644, 220, 4643, 15, 3005, 788, 620, 298, 2901,
               993, 1109,
               592, 14030, 701, 2524, 2024, 354, 260, 233, 1223, 2698, 1304, 319, 1044, 341, 1452, 3521, 364, 859, 879,
               3932, 852,
               2061, 3021, 488, 1948, 2281, 2488, 1069, 2257, 935, 952, 3688, 1490, 27, 9598, 991, 1574, 420, 2885,
               1782, 2366,
               3756, 1498, 2507, 1442, 2006, 868, 553, 1896, 341, 903, 2847, 1588, 1164, 6810, 168, 1334, 3438, 2947,
               3320, 2911,
               486, 479, 689, 768, 2470, 3067, 3177, 4247, 1318, 2953, 983, 4660, 1831, 1139, 4600, 522, 1226, 1033,
               1974, 3261,
               1014, 4922, 2224, 1421, 7093, 2105, 2713, 759, 2998, 872, 2925, 172, 6970, 1918, 3276, 847, 4167, 0,
               5976, 2806,
               11731, 4398, 12, 2352, 6289, 1352, 2658, 988, 3503, 2078, 2569, 2618, 2908, 9310, 3007, 2173, 181, 8992,
               2895,
               5253, 4857, 2774, 876, 3778, 2077, 1725, 808, 1458, 5197, 439, 0, 658, 1466, 2940, 3505, 1449, 3399,
               2960, 3807,
               976, 711, 883, 1757, 2290, 1193, 5778, 427, 2346, 3455, 4293, 75, 2261, 2934, 1159, 655, 1099, 3027, 523,
               1177, 2855, 326, 2077, 221, 4308, 1761, 934, 1150, 950, 3574, 393, 1461, 405, 4150, 1092, 2789, 232,
               1601,
               7002, 3746, 3117, 674, 1495, 3303, 3008, 364]

    epoch_2 = [3114, 4222, 456, 1172, 1414, 3677, 1535, 1719, 646, 4635, 943, 1451,
               529, 80, 1546, 1636, 743, 1843, 921, 48, 1067, 4167, 11, 1918,
               183, 507, 1520, 3065, 2752, 519, 1277, 336, 2564, 843, 3536, 206,
               431, 5172, 3345, 280, 274, 3617, 2293, 954, 2610, 633, 2007, 891,
               727, 149, 2231, 363, 402, 3404, 1579, 1987, 2426, 2884, 51, 3022,
               1450, 407, 3096, 3182, 1160, 1552, 1533, 691, 5182, 1141, 452, 591,
               1193, 1382, 4120, 817, 1342, 3983, 1653, 157, 123, 123, 1897, 589,
               1439, 247, 1840, 1401, 1993, 1266, 2845, 1866, 2876, 2232, 4071, 1311,
               8843, 1688, 1135, 476, 1331, 928, 2987, 2693, 945, 2138, 1672, 906,
               1878, 2372, 2021, 3115, 2796, 3397, 914, 1395, 3834, 1266, 189, 829,
               13457, 1056, 304, 372, 835, 850, 686, 286, 3622, 602, 1030, 3837,
               711, 849, 4371, 308, 604, 906, 1454, 1523, 1529, 1008, 3113, 3825,
               3020, 153, 5881, 3333, 1825, 1639, 402, 1166, 567, 1652, 367, 1329,
               3353, 1200, 929, 2032, 582, 15428, 921, 5134, 617, 5644, 2385, 2245,
               922, 2830, 496, 3518, 2885, 2169, 2787, 877, 3385, 2626, 314, 2634,
               15351, 3519, 3018, 1386, 1187, 3449, 587, 2541, 877, 1899, 3431, 5279,
               2698, 1872, 5092, 1364, 1635, 445, 2220, 403, 2235, 2074, 807, 1446,
               1018, 12777, 1392, 890, 3947, 702, 1185, 1668, 3172, 549, 3507, 2741,
               2201, 2806, 911, 4018, 5184, 17804, 3411, 3152, 14150, 1782, 379, 807,
               2664, 1928, 2242, 1190, 4099, 2778, 1101, 1991, 115, 841, 5404, 3049,
               6609, 7213, 2926, 5795, 2263, 3437, 918, 4109]

    # distribute_to_matrix(matrix, epoch_1)

    sns.heatmap(matrix, mask=matrix < 1, linewidth=0.5)
    plt.show()


def distribute_to_matrix(matrix, products):
    j = 0
    for i in range(0, 21):
        for aux in range(0, 23):
            if matrix[aux, i] == -10:
                matrix[aux, i] = products[j]
                j += 1
