from MUSTBE import *
import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np


def get_useful_data(truth_counts, truth_coords, adata_var_names):
    # DIST需要的是：simu_3D， not_in_tissue_coords， simu_coords，adata.var_names
    # not_in_tissue_coords: 真实值的M矩阵

    if not max(truth_coords[:, 0]) % 2:
        x_index = (truth_coords[:, 0] < max(truth_coords[:, 0]))
        truth_coords = truth_coords[x_index]
        truth_counts = truth_counts[x_index]
    if not max(truth_coords[:, 1]) % 2:
        y_index = (truth_coords[:, 1] < max(truth_coords[:, 1]))
        truth_coords = truth_coords[y_index]
        truth_counts = truth_counts[y_index]

    simu_3D, simu_counts, simu_coords, not_in_tissue_coords, truth_counts = get_down_ST(truth_counts, truth_coords)
    # imputed_x, imputed_y = np.mgrid[min(truth_coords[:, 0]):max(truth_coords[:, 0]) + 1:1,
    #                        min(truth_coords[:, 1]):max(truth_coords[:, 1]) + 1:1]
    imputed_x, imputed_y = np.mgrid[0:max(truth_coords[:, 0])-min(truth_coords[:, 0]) + 1:1,
                           0:max(truth_coords[:, 1])-min(truth_coords[:, 1]) + 1:1]

    impute_position = [imputed_x, imputed_y, not_in_tissue_coords]
    print(f"\n[调试] 坐标映射验证:")
    print(f"原始坐标数量: {len(simu_coords)}")
    print(f"插值坐标数量: {len(impute_position)}")
    # 在 img2expr 返回前添加
    print(f"\n[调试] 插值结果验证:")
    print(f"原始矩阵形状: {truth_counts.shape}")
    b, h, w = simu_3D.shape
    simu_3D = torch.Tensor(simu_3D.reshape((b, 1, h, w)))
    return truth_counts, [simu_3D, impute_position, simu_coords, adata_var_names], [simu_counts, impute_position,
                                                                                    simu_coords, adata_var_names]

def get_mel1_rep1():
    counts = pd.read_csv('/root/SpaVGN/data/mel1_rep1/counts.csv', index_col=0)  # index_col=0:第一列为索引值
    coords = pd.read_csv('/root/SpaVGN/data/mel1_rep1/coords.csv', index_col=0)
    adata = ad.AnnData(X=counts.values, obs=coords, var=pd.DataFrame(index=counts.columns.values))  # var只有行索引

    # 添加指控指标到原adata中 如var添加n_cells_by_counts(列标签)， 表示这个基因出现在了几个点中  基因名字是行索引，n_cells是列标签
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    # 以下两个函数除了有筛选功能，还会分别添加一列，用于表示每个点有几个基因、每个基因存在于几个点中
    # 改变了原adata
    sc.pp.filter_cells(adata, min_genes=20)  # 筛除低于20个基因的点
    sc.pp.filter_genes(adata, min_cells=10)  # 筛除低于在10个点中存在的基因

    train_adata = adata[:, adata.var["n_cells_by_counts"] > len(adata.obs.index) * 0.1]
    truth_counts = np.array(train_adata.X)  # ndarray对象
    truth_coords = train_adata.obs[['array_row', 'array_col']].values

    return get_useful_data(truth_counts, truth_coords, [i for i in range(truth_counts.shape[1])])



def get_10X_Mouse():
    train_adata = sc.read_visium("/root/SpaVGN/data/V1_Mouse_Brain_Sagittal_Posterior")
    train_adata.var_names_make_unique()
    # train_adata.X:稀疏矩阵
    sc.pp.calculate_qc_metrics(train_adata, inplace=True)
    sc.pp.filter_cells(train_adata, min_genes=200)
    sc.pp.filter_genes(train_adata, min_cells=10)

    train_adata = train_adata[:, train_adata.var["n_cells_by_counts"] > len(train_adata.obs.index) * 0.1]
    truth_counts = np.array(train_adata.X.todense())
    truth_coords = train_adata.obs[['array_row', 'array_col']].values

    # 将spot排序，方便最终比较相关性
    con = np.concatenate((truth_coords, truth_counts), axis=1)
    con_df = pd.DataFrame(con).sort_values([0, 1], ascending=[True, True])
    truth_counts = con_df.iloc[:, 2:].values
    truth_coords = con_df.iloc[:, 0:2].values

    return get_useful_data(truth_counts, truth_coords, [i for i in range(truth_counts.shape[1])])

def get_useful_data_10X(truth_counts, truth_coords, adata_var_names):

    if not max(truth_coords[:, 0]) % 2:
        x_index = (truth_coords[:, 0] < max(truth_coords[:, 0]))
        truth_coords = truth_coords[x_index]
        truth_counts = truth_counts[x_index]
    if not max(truth_coords[:, 1]) % 4:
        y_index = (truth_coords[:, 1] < max(truth_coords[:, 1]))
        truth_coords = truth_coords[y_index]
        truth_counts = truth_counts[y_index]

    # DIST需要的是：simu_3D， not_in_tissue_coords， simu_coords，adata.var_names
    # not_in_tissue_coords: 真实值的M矩阵
    simu_3D, simu_counts, simu_coords, not_in_tissue_coords, truth_counts = get_down_10x(truth_counts, truth_coords)
    # 在 img2expr 函数中添加


    #imputed_x, imputed_y = np.mgrid[min(truth_coords[:, 0]):max(truth_coords[:, 0]) + 1:1,
    #                         min(truth_coords[:, 1]):max(truth_coords[:, 1]) + 1:1]
    imputed_x, imputed_y = np.mgrid[0:max(truth_coords[:, 0])-min(truth_coords[:, 0]) + 1:1,
                           0:max(truth_coords[:, 1])-min(truth_coords[:, 1]):2]
    for i in range(1, imputed_y.shape[0], 2):
        imputed_y[i] = imputed_y[i] + 1

    impute_position = [imputed_x, imputed_y, not_in_tissue_coords]
    print(f"\n[调试] 坐标映射验证:")
    print(f"原始坐标数量: {len(simu_coords)}")
    print(f"插值坐标数量: {len(impute_position)}")
    # 在 img2expr 返回前添加
    print(f"\n[调试] 插值结果验证:")
    print(f"原始矩阵形状: {truth_counts.shape}")
    b, h, w = simu_3D.shape
    simu_3D = torch.Tensor(simu_3D.reshape((b, 1, h, w)))
    return truth_counts, [simu_3D, impute_position, simu_coords, adata_var_names], [simu_counts, impute_position,
                                                                                    simu_coords, adata_var_names]
