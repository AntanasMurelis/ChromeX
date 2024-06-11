import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from typing import Optional
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import jax

from GEBayesianModel import gene_expression_model

class HoneyBADGER:
    def __init__(self):
        self.gexp_norm = None
        self.mv_fit = None


    def set_mv_fit(self, num_genes=np.arange(5, 105, 5), rep=50, plot=False, verbose=True):
        if verbose:
            print('Modeling expected variance ...')

        mean_var_comp = {}
        for ng in num_genes:
            np.random.seed(0)
            m = np.vstack([self.gexp_norm.sample(n=ng, axis=0).mean(axis=0) for _ in range(rep)])
            mean_var_comp[ng] = m

        fits = {}
        for k in range(self.gexp_norm.shape[1]):
            mean_comp = np.column_stack([mean_var_comp[ng][:, k] for ng in num_genes])

            if plot:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                melted_data = pd.DataFrame(mean_comp, columns=num_genes).melt()
                sns.boxplot(x='variable', y='value', data=melted_data, ax=axes[0])
                axes[0].set_title("Boxplot of Mean Variance")

            log_num_genes = np.log10(num_genes)
            log_var = np.log10(mean_comp.var(axis=0))

            if plot:
                axes[1].plot(log_num_genes, log_var, linestyle='-', marker='o')
                axes[1].set_title("Log-Log Plot of Variance vs Number of Genes")

            df = pd.DataFrame({'x': log_num_genes, 'y': log_var})
            reg = LinearRegression().fit(df[['x']], df['y'])
            fit = reg.predict(df[['x']])

            if plot:
                axes[2].plot(log_num_genes, fit, color='red')
                axes[2].plot(log_num_genes, log_var, linestyle='-', marker='o')
                axes[2].set_title("Linear Fit on Log-Log Data")
                # show plot 
                # plt.show()

            fits[self.gexp_norm.columns[k]] = reg

        self.mv_fit = fits

        if verbose:
            print('Done!')


    def set_gexp_mats(self, gexp_sc_init: pd.DataFrame, gexp_ref_init: pd.DataFrame,
                      filter: bool = True, min_mean_both: float = 0, 
                      min_mean_test: Optional[float] = None, min_mean_ref: Optional[float] = None, 
                      scale_data: bool = True, verbose: bool = True):
        if verbose:
            print("Initializing expression matrices ...")

        if not isinstance(gexp_ref_init, pd.DataFrame):
            gexp_ref_init = pd.DataFrame(gexp_ref_init)

        vi = gexp_sc_init.index.intersection(gexp_ref_init.index)
        if len(vi) < 10:
            print('WARNING! GENE NAMES IN EXPRESSION MATRICES DO NOT SEEM TO MATCH!')
        
        gexp_sc = gexp_sc_init.loc[vi]
        self.gexp_ref = gexp_ref_init.loc[vi]

        if filter:
            mean_sc = gexp_sc.mean(axis=1)
            mean_ref = self.gexp_ref.mean(axis=1)
            vi = ((mean_sc > min_mean_both) & (mean_ref > min_mean_both)) | (mean_sc > min_mean_test) | (mean_ref > min_mean_ref)
            if verbose:
                print(f"{vi.sum()} genes passed filtering ...")
            self.gexp_sc = self.gexp_sc.loc[vi]
            self.gexp_ref = self.gexp_ref.loc[vi]

        if scale_data:
            if verbose:
                print("Scaling coverage ...")
            self.gexp_sc = self.gexp_sc.apply(scale, axis=1, result_type='broadcast')
            self.gexp_ref = self.gexp_ref.apply(scale, axis=1, result_type='broadcast')

        if verbose:
            print(f"Normalizing gene expression for {self.gexp_sc.shape[0]} genes and {self.gexp_sc.shape[1]} cells ...")
        refmean = self.gexp_ref.mean(axis=1)
        self.gexp_norm = self.gexp_sc.sub(refmean, axis=0)

        if verbose:
            print("Done setting initial expression matrices!")

    def set_gexp_mats(self, gexp_sc_init: pd.DataFrame, gexp_ref_init: pd.DataFrame,
                      filter: bool = True, min_mean_both: float = 0, 
                      min_mean_test: Optional[float] = None, min_mean_ref: Optional[float] = None, 
                      scale_data: bool = True, verbose: bool = True):
        if verbose:
            print("Initializing expression matrices ...")

        if not isinstance(gexp_ref_init, pd.DataFrame):
            gexp_ref_init = pd.DataFrame(gexp_ref_init)

        common_genes = gexp_sc_init.index.intersection(gexp_ref_init.index)
        if len(common_genes) < 10:
            print('WARNING! GENE NAMES IN EXPRESSION MATRICES DO NOT SEEM TO MATCH!')
        
        gexp_sc = gexp_sc_init.loc[common_genes]
        gexp_ref = gexp_ref_init.loc[common_genes]

        if filter:
            mean_sc = gexp_sc.mean(axis=1)
            mean_ref = gexp_ref.mean(axis=1)
            filter_mask = ((mean_sc > min_mean_both) & (mean_ref > min_mean_both)) | (mean_sc > min_mean_test) | (mean_ref > min_mean_ref)
            if verbose:
                print(f"{filter_mask.sum()} genes passed filtering ...")
            gexp_sc = gexp_sc.loc[filter_mask]
            gexp_ref = gexp_ref.loc[filter_mask]

        if scale_data:
            if verbose:
                print("Scaling coverage ...")
            gexp_sc = pd.DataFrame(scale(gexp_sc, axis=1), index=gexp_sc.index, columns=gexp_sc.columns)
            gexp_ref = pd.DataFrame(scale(gexp_ref, axis=1), index=gexp_ref.index, columns=gexp_ref.columns)

        if verbose:
            print(f"Normalizing gene expression for {gexp_sc.shape[0]} genes and {gexp_sc.shape[1]} cells ...")
        refmean = gexp_ref.mean(axis=1)
        self.gexp_norm = gexp_sc.sub(refmean, axis=0)

        if verbose:
            print("Done setting initial expression matrices!")

    def calc_gexp_cnv_prob(self, gexp_norm_sub=None, m=0.15, genes_of_interest=None, quiet=True, verbose=False):
        if gexp_norm_sub is not None:
            gexp_norm = gexp_norm_sub
            mv_fit = {k: self.mv_fit[k] for k in gexp_norm_sub.columns}
        else:
            gexp_norm = self.gexp_norm
            mv_fit = self.mv_fit

        if genes_of_interest is not None:
            gexp = gexp_norm.loc[genes_of_interest]
            if gexp.shape[0] < 3:
                print(f"WARNING! ONLY {gexp.shape[0]} GENES IN THE LIST!")
                return {"amplification": np.nan, "deletion": np.nan, "mu0": np.nan}
        else:
            gexp = gexp_norm

        mu0 = gexp.mean(axis=1)
        ng = gexp.shape[0]
        sigma0 = jax.numpy.array([jax.numpy.sqrt(10 ** mv_fit[col].predict(jax.numpy.array([[ng]]))[0].item()) for col in gexp.columns])

        if verbose:
            print('Initializing model ...')

        kernel = NUTS(gene_expression_model)
        kernel = DiscreteHMCGibbs(kernel, modified=True)
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=4, progress_bar=not quiet)
        mcmc.run(jax.random.PRNGKey(0), jax.device_put(jax.numpy.array(gexp.values)), jax.device_put(sigma0), jax.device_put(jax.numpy.array(mu0)))

        samples = mcmc.get_samples()

        if verbose:
            print('...Done!')

        S = samples['S']
        dd = samples['dd']
        mu = samples['mu']

        delcall = np.mean(S * (1 - dd), axis=0)
        ampcall = np.mean(S * dd, axis=0)
        names = gexp.columns

        return {
            'posterior probability of amplification': dict(zip(names, ampcall)),
            'posterior probability of deletion': dict(zip(names, delcall)),
            'estimated mean normalized expression deviation': dict(zip(names, mu0))
        }

# Example usage
gexp_sc_init = pd.DataFrame(np.random.normal(size=(1000, 50)), index=[f'gene_{i}' for i in range(1000)])
gexp_ref_init = pd.DataFrame(np.random.normal(size=(1000, 50)), index=[f'gene_{i}' for i in range(1000)])

hb = HoneyBADGER()
hb.set_gexp_mats(gexp_sc_init, gexp_ref_init, filter=False, scale_data=True, verbose=True)
print(hb.gexp_norm)

        

    # def set_mv_fit(self, num_genes=np.arange(5, 105, 5), rep=50, plot=False, verbose=True):
    #     if verbose:
    #         print('Modeling expected variance ...')

    #     # Initialize mean_var_comp dictionary
    #     mean_var_comp = {}
    #     np.random.seed(0)
        
    #     # Precompute indices to sample
    #     sample_indices = {ng: [np.random.choice(self.gexp_norm.index, ng, replace=False) for _ in range(rep)] for ng in num_genes}
        
    #     for ng in num_genes:
    #         samples = np.array([self.gexp_norm.loc[idx].mean(axis=0) for idx in sample_indices[ng]])
    #         mean_var_comp[ng] = samples
        
    #     fits = {}
    #     log_num_genes = np.log10(num_genes)
        
    #     for k in range(self.gexp_norm.shape[1]):
    #         mean_comp = np.column_stack([mean_var_comp[ng][:, k] for ng in num_genes])
    #         log_var = np.log10(mean_comp.var(axis=0))
            
    #         if plot:
    #             fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
    #             melted_data = pd.DataFrame(mean_comp, columns=num_genes).melt()
    #             sns.boxplot(x='variable', y='value', data=melted_data, ax=axes[0])
    #             axes[0].set_title("Boxplot of Mean Variance")
                
    #             axes[1].plot(log_num_genes, log_var, linestyle='-', marker='o')
    #             axes[1].set_title("Log-Log Plot of Variance vs Number of Genes")

    #         df = pd.DataFrame({'x': log_num_genes, 'y': log_var})
    #         reg = LinearRegression().fit(df[['x']], df['y'])
    #         fit = reg.predict(df[['x']])

    #         if plot:
    #             axes[2].plot(log_num_genes, fit, color='red')
    #             axes[2].plot(log_num_genes, log_var, linestyle='-', marker='o')
    #             axes[2].set_title("Linear Fit on Log-Log Data")
    #             plt.show()

    #         fits[self.gexp_norm.columns[k]] = reg

    #     self.mv_fit = fits

    #     if verbose:
    #         print('Done!')

# Example usage
hb.set_mv_fit(plot=True)

# Example usage
gexp_norm = pd.DataFrame(np.random.normal(size=(100, 50)), index=[f'gene_{i}' for i in range(100)], columns=[f'cell_{i}' for i in range(50)])
genes = pd.Series([f'gene_{i}' for i in range(100)], index=[f'gene_{i}' for i in range(100)])
mv_fit = {f'cell_{i}': LinearRegression().fit(np.log10(np.arange(1, 101).reshape(-1, 1)), np.random.normal(size=(100,))) for i in range(50)}

results = hb.calc_gexp_cnv_prob(genes_of_interest=['gene_1', 'gene_2', 'gene_3'], verbose=True)
print(results)
