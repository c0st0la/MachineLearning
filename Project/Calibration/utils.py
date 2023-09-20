import numpy as np
import os


def v_col(x: np.ndarray) -> np.ndarray:
    return x.reshape((x.size, 1))


def v_row(x: np.ndarray) -> np.ndarray:
    return x.reshape((1, x.size))


def compute_mean(d: np.ndarray) -> np.ndarray:
    return v_col(d.mean(1))

def to_column(array):
    return array.reshape((array.size, 1))


def computes_mean_classes(d: np.ndarray, ltr: np.ndarray) -> list:
    n_classes = np.unique(ltr).size
    means = [compute_mean(d[:, ltr == i]) for i in range(n_classes)]
    return means


def compute_std(d: np.ndarray) -> np.ndarray:
    return v_col(d.std(1))


def compute_covariance_matrix(d: np.ndarray) -> np.ndarray:
    mu = compute_mean(d)
    return (1 / d.shape[1]) * np.dot((d - mu), (d - mu).T)

def center_data(D):
    mu = to_column(D.mean(axis=1))
    DC = D - mu
    return DC
def compute_within_covariance(D, L, labels):
    Covariance_within = 0
    tot_samples = 0
    for label in labels:
        # D_class is the dataset filtered by class
        D_class = D[:, L == label]
        DC_class = center_data(D_class)
        num_samples = D_class.shape[1]
        tot_samples += num_samples
        Covariance_within = Covariance_within + np.dot(DC_class, DC_class.T)
    Covariance_within = Covariance_within / float(tot_samples)
    return Covariance_within


def compute_covariance_matrices_for_classes(dt: np.ndarray, lt: np.ndarray) -> list:
    n_classes = np.unique(lt).size
    cov_matrix = [compute_covariance_matrix(dt[:, lt == i]) for i in range(n_classes)]
    return cov_matrix


def split_data(DTR, LTR, K=5, seed=27):
    np.random.seed(seed)
    idx = np.random.permutation(DTR.shape[1])
    d = np.hsplit(DTR[:, idx], K)
    l = np.hsplit(LTR[idx], K)
    return d, l, idx


# Only for printing the datasets
def check_datasets(dtr, ltr, dte, lte):

    print("=================================")
    print("Male in Training:" + str(dtr[:, ltr == 0].shape))
    print("Female in Training:" + str(dtr[:, ltr == 1].shape))
    print("Male in Evaluation:" + str(dte[:, lte == 0].shape))
    print("Female in Evaluation:" + str(dte[:, lte == 1].shape))
    print("=================================")


def parse_file(file_name: str) -> tuple:
    data = []
    labels = []
    with open(file_name, 'r') as f:
        for line in f:
            sample = [float(i) for i in line.rstrip().split(',')[0:12]]
            label = float(line.rstrip().split(',')[12])
            data.append(sample)
            labels.append(label)

    return np.array(data).T, np.array(labels)


def load_dataset() -> tuple:
    dtr, ltr = parse_file('data/train.txt')
    dte, lte = parse_file('data/test.txt')
    return dtr, ltr, dte, lte


def create_folder_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_conf_mat_uniform(prediction, L):
    conf_mat = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            conf_mat[i][j] = (1 * np.bitwise_and(prediction == i, L == j)).sum()

    return conf_mat


def compute_NDCF_conf_mat(conf_mat, pi, C_fp, C_fn):
    FNR = conf_mat[0][1] / (conf_mat[0][1] + conf_mat[1][1])
    FPR = conf_mat[1][0] / (conf_mat[1][0] + conf_mat[0][0])
    return (pi * C_fn * FNR + (1-pi) * C_fp * FPR) / min([pi * C_fn, (1-pi) * C_fp])


def build_conf_mat(llr: np.ndarray,L: np.ndarray,pi:float, C_fn:float,C_fp:float):
    t = -np.log(pi*C_fn/((1-pi)*C_fp))
    predictions = 1*(llr > t)
    return build_conf_mat_uniform(predictions,L)


def compute_DCF(llr: np.ndarray, L: np.ndarray, pi: float, C_fn: float, C_fp: float):
    conf_mat = build_conf_mat(llr, L, pi, C_fn, C_fp)
    FNR = conf_mat[0][1]/ (conf_mat[0][1] + conf_mat[1][1])
    FPR = conf_mat[1][0]/ (conf_mat[1][0] + conf_mat[0][0])
    return pi * C_fn * FNR + (1-pi) * C_fp * FPR


def compute_NDCF(llr: np.ndarray, L: np.ndarray, pi: float, C_fn: float, C_fp: float):
    return compute_DCF(llr, L, pi, C_fn, C_fp) / min([pi*C_fn, (1-pi)*C_fp])


def compute_minimum_NDCF(llr, L, pi, C_fp, C_fn):
    llr = llr.ravel()
    tresholds = np.concatenate([np.array([-np.inf]), np.sort(llr), np.array([np.inf])])
    DCF = np.zeros(tresholds.shape[0])
    for (idx, t) in enumerate(tresholds):
        pred = 1 * (llr > t)
        conf_mat = build_conf_mat_uniform(pred, L)
        DCF[idx] = compute_NDCF_conf_mat(conf_mat, pi, C_fp, C_fn)
    argmin = DCF.argmin()
    return DCF[argmin], tresholds[argmin]


def compute_roc_points(llr, L):
    tresholds = np.concatenate([np.array([-np.inf]), np.sort(llr), np.array([np.inf])])
    N_label0 = (L == 0).sum()
    N_label1 = (L == 1).sum()
    ROC_points_TPR = np.zeros(L.shape[0] + 2)
    ROC_points_FPR = np.zeros(L.shape[0] + 2)
    for (idx, t) in enumerate(tresholds):
        pred = 1 * (llr > t)
        TPR = np.bitwise_and(pred == 1, L == 1).sum() / N_label1
        FPR = np.bitwise_and(pred == 1, L == 0).sum() / N_label0
        ROC_points_TPR[idx] = TPR
        ROC_points_FPR[idx] = FPR
    return ROC_points_TPR, ROC_points_FPR


def compute_det_points(llr, L):
    threshold = np.concatenate([np.array([-np.inf]), np.sort(llr), np.array([np.inf])])
    FNR_points = np.zeros(L.shape[0] + 2)
    FPR_points = np.zeros(L.shape[0] + 2)
    for (idx, t) in enumerate(threshold):
        pred = 1 * (llr > t)
        FNR = 1 - (np.bitwise_and(pred == 1, L == 1).sum() / (L == 0).sum())
        FPR = np.bitwise_and(pred == 1, L == 0).sum() / (L == 1).sum()
        FNR_points[idx] = FNR
        FPR_points[idx] = FPR
    return FNR_points, FPR_points


