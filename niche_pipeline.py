#!/usr/bin/env python3

import argparse
import scanpy as sc
from scipy.stats import wilcoxon
import scipy.stats as stats
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")

# function for paired wilcoxon test
def paired_wilcoxon(adata, cluster_info, groupby = 'condition', reference = 'observed', target = 'expected', logfc_threshold=0.25,
                   use_zeros = True):
    """
    Perform a paired Wilcoxon signed-rank test for each gene in each cluster between two specified groups.

    Parameters:
    -----------
    adata : AnnData
        An AnnData object containing the single-cell expression data.
        
    cluster_info : str
        The column name in `adata.obs` that contains cluster information.
        
    groupby : str
        The column name in `adata.obs` that contains the group 
        information to compare (e.g., time points, conditions).
        
    reference : str
        The reference group label within the `groupby` column. (observed)
        
    target : str
        The target group label within the `groupby` column to compare 
        against the reference group. (expected)
        
    logfc_threshold : float, optional
        Log fold-change threshold for filtering genes (default is 0.25).
        
    use_zeros : bool, optional
        Whether to include zero values in the comparison (default is True).

    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing the results of the paired Wilcoxon test for each gene in each cluster. 
        The columns include:
            - 'gene': Gene name.
            - 'stat': Wilcoxon test statistic.
            - 'p_value': P-value of the test.
            - 'p_value_adj': Adjusted p-value after Bonferroni correction.
            - 'mean_ref': Mean expression of the reference(observed expression) group.
            - 'mean_tgt': Mean expression of the target(expected expression) group.
            - 'logfc': Log fold-change between the reference and target groups.
            - 'cluster': Cluster name.
            - 'n_spots': Number of spots (cells) used in the test.

    Example:
    --------
    results_df = paired_wilcoxon(adata, cluster_info='leiden', groupby='condition', 
                                 reference='control', target='treated', logfc_threshold=0.25, use_zeros=True)
    """
        
    pairs = adata.obs.groupby(groupby).groups
    results = []
    clusters = adata.obs[cluster_info].cat.categories
    # print(clusters)
    for i in range(len(clusters)):
        cluster = clusters[i]
        print(f'Obtaining niche gene for cluster {cluster}...')
        sub_adata = adata[adata.obs[cluster_info]== cluster,]
        # print(sub_adata)
        for gene in sub_adata.var_names:
            gene_data = sub_adata[:, gene].X.flatten()
            ref_data = gene_data[sub_adata.obs[groupby] == reference]
            tgt_data = gene_data[sub_adata.obs[groupby] == target]

            if use_zeros == False:
                non_zero_indices = ref_data != 0
                ref_data = ref_data[non_zero_indices]
                tgt_data = tgt_data[non_zero_indices]

            if len(ref_data) > 0 and len(tgt_data) > 0:
                if not np.all(ref_data == tgt_data):  # Check if differences are not all zero

                    stat, p_value = wilcoxon(ref_data, tgt_data, alternative = "greater")
                    mean_ref = np.mean(ref_data).astype(float)
                    mean_tgt = np.mean(tgt_data).astype(float)

                    logfc = np.log2(mean_ref + 1) - np.log2(mean_tgt + 1)  # Adding 1 to avoid log(0)
                else:
                     # Default values when differences are all zero
                    stat, p_value, logfc = 0, 1, 0
                    mean_ref = np.mean(ref_data).astype(float)
                    mean_tgt = np.mean(tgt_data).astype(float)

                results.append([gene, stat, p_value, 
                                mean_ref, mean_tgt, 
                                logfc, cluster, len(ref_data)])
        results_df = pd.DataFrame(results, columns=['gene', 'stat', 'p_value', 'mean_ref', 
                                                    'mean_tgt', 'logfc','cluster','n_spots'])
        
        # Bonferroni correction
        num_tests = len(results_df)
        results_df['p_value_adj'] = np.minimum(results_df['p_value'] * num_tests, 1.0)
        
    return results_df
# clean up the celltype names (remove special symbols and space etc), must be done for regression

def clean_column_names(df):
    df.columns = df.columns.str.replace('-', '_')
    df.columns = df.columns.str.replace('.', '_')
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    return df

def obtain_clean_celltype_names(adata):
    adata.obs = clean_column_names(adata.obs)
    orig_celltypes = list(adata.uns['mod']['factor_names'])
    celltypes = [i.replace(' ', '_') for i in orig_celltypes]
    celltypes = [i.replace('(', '') for i in celltypes]
    celltypes = [i.replace(')', '') for i in celltypes]
    celltypes = [i.replace('.', '_') for i in celltypes]
    celltypes = [i.replace('-', '_') for i in celltypes]
    return celltypes

# Stepwise AIC function
def stepwise_aic(data, response, predictors, direction='both', max_iter=100):
    """
    Perform stepwise regression based on AIC.
    direction: 'forward', 'backward', or 'both'
    """
    def get_aic(model):
        return model.aic
    
    initial_predictors = [] if direction in ['forward', 'both'] else predictors.copy()
    best_aic = np.inf
    best_model = None
    current_predictors = initial_predictors.copy()
    
    for _ in range(max_iter):
        changed = False
        
        # Forward step
        if direction in ['forward', 'both']:
            remaining_predictors = [p for p in predictors if p not in current_predictors]
            aic_with_candidates = []
            for p in remaining_predictors:
                formula = f"{response} ~ {' + '.join(current_predictors + [p])}"
                model = smf.ols(formula=formula, data=data).fit()
                aic_with_candidates.append((get_aic(model), p))
            
            aic_with_candidates.sort()
            if aic_with_candidates and aic_with_candidates[0][0] < best_aic:
                best_aic, best_predictor = aic_with_candidates[0]
                current_predictors.append(best_predictor)
                best_model = smf.ols(f"{response} ~ {' + '.join(current_predictors)}", data=data).fit()
                changed = True
        
        # Backward step
        if direction in ['backward', 'both'] and len(current_predictors) > 1:
            aic_with_candidates = []
            for p in current_predictors:
                formula = f"{response} ~ {' + '.join([x for x in current_predictors if x != p])}"
                model = smf.ols(formula=formula, data=data).fit()
                aic_with_candidates.append((get_aic(model), p))
            
            aic_with_candidates.sort()
            if aic_with_candidates and aic_with_candidates[0][0] < best_aic:
                best_aic, worst_predictor = aic_with_candidates[0]
                current_predictors.remove(worst_predictor)
                best_model = smf.ols(f"{response} ~ {' + '.join(current_predictors)}", data=data).fit()
                changed = True
        
        if not changed:
            break
    return best_model

def valid_variable_name(name):
    name = name.replace('.','_')
    name = name.replace('-','_')
    if name[0].isdigit():
        return f"gene_{name}"
    return name

def extract_interaction_terms(models_per_gene, p_value_threshold=0.05, vif_threshold=10):
    results = []
    
    for gene, clusters in models_per_gene.items():
        for cluster, model in clusters.items():
            # Get the summary of the model
            summary = model.summary2().tables[1]
            # Get the design matrix
            X = model.model.exog
            vif_df = pd.DataFrame()
            vif_df['VIF'] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
            vif_df['term'] = model.model.exog_names
            
            # Iterate over the rows in the summary table
            for term, row in summary.iterrows():
                if term != 'Intercept' and row['Coef.'] > 0:  # Check for positive coefficient
                    p_value = row['P>|t|']
                    coef = row['Coef.']
                    std_err = row['Std.Err.']
                    
                    # Get VIF value for the term
                    vif_value = vif_df.loc[vif_df['term'] == term, 'VIF'].values[0]
                    results.append([gene, cluster, term,coef, std_err, p_value, vif_value])
    
    # Create a dataframe with the results
    results_df = pd.DataFrame(results, columns=['gene', 'cluster', 'term', 'coef', 'std_err','p_value', 'VIF'])
    results_df = results_df.sort_values(by=['cluster', 'gene', 'p_value'])
    
    # Apply p_value and VIF thresholds if provided
    if p_value_threshold is not None:
        results_df = results_df[results_df['p_value'] <= p_value_threshold]
    
    if vif_threshold is not None:
        results_df = results_df[results_df['VIF'] <= vif_threshold]
    
    return results_df


# Main regression function
def regress_residual_on_interaction(observed_expression, expected_expression, 
                                    celltypes, cell_sig,
                                    niche_gene_results,
                                    cluster_summary, 
                                    niche_candidate_threshold = 0.08,
                                    cluster_info = 'proportion_leiden', 
                                    use_zeros = True):
    
    genes = niche_gene_results['gene'].unique()
    genes_with_too_many_candidates = []
    models_per_gene = {}
    i = 0
    total = len(genes)
    print(f"starting regression for {total} genes")
    for gene in genes:
#         print(f"start regression for gene : {gene}")
        valid_gene = valid_variable_name(gene)
        clusters = list(niche_gene_results.loc[niche_gene_results['gene']==valid_gene,'cluster'])
        models_per_cluster = {}
        for cluster in clusters:
            
            sub_observed = observed_expression.loc[observed_expression[cluster_info]==cluster,:]
            sub_expected = expected_expression.loc[expected_expression[cluster_info]==cluster,:]
            sub_deconv = sub_observed.loc[:,celltypes]
#             print(sub_deconv.shape)
            filter_df = sub_deconv.multiply(cell_sig.loc[valid_gene,:]).to_numpy()
                        
            filter_df = filter_df / sub_expected.loc[sub_expected[cluster_info]==cluster,valid_gene].to_numpy().reshape(-1,1)
            
            total_expression_per_celltype = filter_df.sum(axis=0)
            
            # index cell types are the cell types which the total cell type specific expression
            # exceeds threshold (mean + 1 * sd)
            # selecting index cell type incorporates information of cell proportion and
            # cell type signature 
            threshold = np.mean(total_expression_per_celltype) + np.std(total_expression_per_celltype)
            indices = np.where(total_expression_per_celltype > threshold)
            index_candidates = np.array(celltypes)[indices]
            
            # niche candidates are cell types that are enriched in a particular cluster (only takes into account the abundance)
            # niche candidates for this cluster are celltypes which have higher than average proportion or celltypes that exceed
            # certain amount of threshold
            niche_candidates = cluster_summary.loc[cluster, (cluster_summary.loc[cluster] > cluster_summary.loc['mean']) | (cluster_summary.loc[cluster] > niche_candidate_threshold)]
            niche_candidates = niche_candidates.index.to_numpy()
            
            # index candidates can also be niche candidates
            niche_candidates = np.union1d(index_candidates, niche_candidates)

            index, niche = np.meshgrid(index_candidates, niche_candidates)
            
            interaction_pairs = np.array([index.flatten(), niche.flatten()]).T

            interaction_terms = []
            for (f1, f2) in interaction_pairs:
                if f1 != f2: # avoid self-interaction term
                    interaction_term = f'{f1}_{f2}'
                    interaction_terms.append(interaction_term)
                    sub_deconv[interaction_term] = sub_deconv[f1] * sub_deconv[f2]
            final_terms = '+'.join(interaction_terms)
            sub_deconv[valid_gene] = sub_observed[valid_gene].values - sub_expected[valid_gene].values
            
            if len(interaction_terms) > 20: # Too many candidate combinations
                # Create model with all interaction terms
                formula = f"{valid_gene} ~ {' + '.join(interaction_terms)}"
                model = smf.ols(formula=formula, data=sub_deconv).fit()
            else:
                # Perform stepwise regression
                model = stepwise_aic(response = valid_gene, predictors = interaction_terms,
                                     data = sub_deconv, direction = 'backward')
            
            models_per_cluster[cluster] = model
        models_per_gene[gene] = models_per_cluster
        i+= 1
        if i % (total // 5) == 0  :
            print(f"Regression of {i}/{total} genes completed.. ")
    print("Regression completed")
    return models_per_gene, genes_with_too_many_candidates

def get_model_summary(all_models, gene, cluster):
    return all_models[gene][cluster].summary2()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process two h5ad files.')
    parser.add_argument('scRNA_seq_file', type=str, help='Path to the single cell RNA seq reference h5ad file')
    parser.add_argument('Visium_file', type=str, help='Path to the Visium h5ad file')
    parser.add_argument('--cluster_info', type=str, default='leiden', 
                        help='The column name in adata.obs that contains cluster information (default: leiden)')
    
    
    args = parser.parse_args()
    cluster_info = args.cluster_info
    # Read the h5ad files
    adata_ref = sc.read(args.scRNA_seq_file)
    adata_vis = sc.read(args.Visium_file)


    # retrieve average cell type signature obtained from scRNA-seq data
    if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
        inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_ref.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_ref.uns['mod']['factor_names']]].copy()
    inf_aver.columns = adata_ref.uns['mod']['factor_names']

    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # final_df is the final expected_expression based on deconvolution result
    total_df = adata_vis.obs[adata_vis.uns['mod']['factor_names']]@inf_aver.T * adata_vis.uns['mod']['post_sample_means']['m_g']

    final_df = ((total_df + adata_vis.uns['mod']['post_sample_means']['s_g_gene_add'][0]) * adata_vis.uns['mod']['post_sample_means']['detection_y_s'])

    # equalize the total counts of observed data and expected data
    adata_vis.obs['mult_factor'] =  adata_vis.obs['total_counts'] / final_df.sum(axis = 1)
    final_df = (final_df.T * adata_vis.obs['mult_factor'])


   # create one single AnnData object that contains both observed expression and expected expression
    # for paired Wilcoxon ranksum test
    meta_data = adata_vis.obs
    observed_expression = pd.concat([adata_vis.to_df(),meta_data],axis = 1)
    expected_expression = final_df.T.copy()
    expected_expression = pd.concat([expected_expression, meta_data], axis = 1)

    # Add metadata to indicate observed and expected information
    observed_expression['condition'] = 'observed'
    expected_expression['condition'] = 'expected'

    # Append suffix to cell names
    observed_expression.index = [f"{cell}_before" for cell in observed_expression.index]
    expected_expression.index = [f"{cell}_after" for cell in expected_expression.index]

    combined_df = pd.concat([observed_expression, expected_expression])
    drop_cols = list(meta_data.columns)
    drop_cols.extend(['condition'])
    adata = sc.AnnData(X=combined_df.drop(columns=drop_cols))
    adata.obs['condition'] = combined_df['condition'].values

    cluster_key = cluster_info
    adata.obs[cluster_key] = combined_df[cluster_key].astype('category')

    observed_expression = clean_column_names(observed_expression)
    expected_expression = clean_column_names(expected_expression)

    meta_data = clean_column_names(meta_data)
    inf_aver = clean_column_names(inf_aver)
    cell_types = obtain_clean_celltype_names(adata_vis)
    # Perform the paired Wilcoxon signed-rank test
    
    de_results = paired_wilcoxon(adata,cluster_info = cluster_info, groupby='condition', 
                                 reference='observed', target='expected', logfc_threshold=0.5,
                                use_zeros = True)

    # Filter results to only include genes where the 'before' condition has significantly higher values
    # and logFC exceeds the threshold
    significant_results = de_results[(de_results['p_value_adj'] < 0.05) & 
                                    (de_results['mean_ref'] > de_results['mean_tgt']) &
                                    (de_results['logfc'] > 0.5)]
    
    significant_results.to_csv('./niche_gene_list.csv')
    # create cluster_summary needed for creating regression model
    # cluster_summary contains average celltype proportion composition of each cluster

    norm_deconv = meta_data.loc[:,cell_types].div(meta_data.loc[:,cell_types].sum(axis = 1),axis = 0)
    norm_deconv[cluster_key] = meta_data[cluster_key]
    cluster_summary = norm_deconv.groupby(cluster_key).mean()
    column_means = cluster_summary.mean()
    cluster_summary.loc['mean'] = column_means

    all_models, genes_with_too_many_candidates = regress_residual_on_interaction(observed_expression, expected_expression,
                               celltypes = cell_types, cell_sig=inf_aver,
                               niche_gene_results=significant_results,
                               cluster_summary=cluster_summary,
                                niche_candidate_threshold=0.08,
                               cluster_info = cluster_info,
                               use_zeros = True)

    all_interaction_terms = extract_interaction_terms(all_models, p_value_threshold=None, vif_threshold=None)

    all_interaction_terms.to_csv('./interaction_term_results.csv')

if __name__ == "__main__":
    main()