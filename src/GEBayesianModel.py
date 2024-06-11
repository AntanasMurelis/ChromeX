import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from jax import random
import numpy as np
import jax
from numpyro.util import format_shapes

def gene_expression_model(gexp, G, C):
    # Hyperparameters
    sigma = numpyro.sample('sigma', dist.Gamma(1.0, 1.0))
    
    # Mixing proportions for the clusters
    alpha = jnp.ones(3)
    theta = numpyro.sample('theta', dist.Dirichlet(alpha))

    # Fixed effects means and precisions for clusters
    mu = jnp.array([0.0, 0.5, 1.0])
    tau = jnp.array([1.0, 2.0, 3.0])
    
    with numpyro.plate('cells', C):
        epsilon = numpyro.sample('epsilon', dist.Categorical(theta), infer={"enumerate": "parallel"})
        
        mu_1 = mu[epsilon]
        tau_1 = tau[epsilon]
        
        with numpyro.plate('genes', G):
            numpyro.sample('gexp', dist.Normal(mu_1, 1.0 / jnp.sqrt(tau_1)), obs=gexp)

# Define the NumPyro model
def gene_expression_model(gexp, sigma0, mag0):
    K, JJ = gexp.shape

    beta = numpyro.sample('beta', dist.Uniform(0, 1))
    dd = numpyro.sample('dd', dist.Bernoulli(beta))

    alpha = numpyro.sample('alpha', dist.Uniform(0, 1).expand([K]))
    S = numpyro.sample('S', dist.Bernoulli(alpha))

    mu = -mag0 * S * (1 - dd) + mag0 * S * dd

    with numpyro.plate('cells', K):
        with numpyro.plate('genes', JJ):
            numpyro.sample('gexp', dist.Normal(mu[:, None], sigma0[:, None]), obs=gexp)
            
# Define the NumPyro model
def gene_expression_model(gexp, sigma0, mag0):
    K, JJ = gexp.shape

    beta = numpyro.sample('beta', dist.Uniform(0, 1))
    dd = numpyro.sample('dd', dist.Bernoulli(beta))

    with numpyro.plate('cells', K):
        alpha = numpyro.sample('alpha', dist.Uniform(0, 1))
        S = numpyro.sample('S', dist.Bernoulli(alpha))

        mu = 0 * (1 - S) + (-mag0 * S * (1 - dd)) + (mag0 * S * dd)
        # mu = jnp.expand_dims(mu, axis=1)  # Expand dims to make it compatible with genes plate
        # sigma0 = jnp.expand_dims(sigma0, axis=1)  # Expand dims to make it compatible with genes plate

        with numpyro.plate('genes', JJ):
            numpyro.sample('gexp', dist.Normal(mu, sigma0), obs=gexp.T)
            

            
# Test Case for the Model
def test_gene_expression_model():
    # Simulate data
    K = 100  # Number of cells
    JJ = 50  # Number of genes
    gexp = jax.random.normal(jax.random.PRNGKey(0), (K, JJ))
    sigma0 = jax.random.uniform(jax.random.PRNGKey(1), (K,), minval=0.1, maxval=1.0)
    mag0 = jax.random.uniform(jax.random.PRNGKey(2), (K,), minval=0.1, maxval=1.0)

    # Define the data for the model
    data = {
        'gexp': gexp,
        'sigma0': sigma0,
        'mag0': mag0
    }

    # Run MCMC
    kernel = NUTS(gene_expression_model)
    kernel = DiscreteHMCGibbs(kernel, modified=True)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=4)
    mcmc.run(jax.random.PRNGKey(2), **data)

    # Get samples
    samples = mcmc.get_samples()

    # Print the shapes of the samples
    print(samples)

    # Assertions to check the expected structure of samples
    # assert 'beta' in samples
    # assert 'dd' in samples
    # assert 'alpha' in samples
    # assert 'S' in samples
    # assert samples['S'].shape == (4000, K)
    # assert samples['dd'].shape == (4000,)
    # assert samples['beta'].shape == (4000,)
    # assert samples['alpha'].shape == (4000, K)

# Run the test case
test_gene_expression_model()