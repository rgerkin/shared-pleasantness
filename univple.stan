functions {
    // This implements a Plackett-Luce ranking model
    // See https://cran.rstudio.com/web/packages/PlackettLuce/vignettes/Overview.html
    // or https://icml.cc/Conferences/2009/papers/347.pdf
    real log_p_ranks(row_vector w, int[] ranks, int n) {
        real log_p = 0; // Initializing log-probability
        row_vector[n] ew = exp(w[ranks]);
        for (i in 1:n) {
            // For each rank (descending), the probability is the e^valence
            // normalized to all the sum of the remaining e^valence
            log_p += log(ew[i]/sum(ew[i:n]));
            }
        return log_p;
        }
    }
    
data {
    int<lower=1> n_odorants; // e.g. 10
    int<lower=1> n_individuals; // e.g. ~200
    int<lower=1, upper=n_individuals> n_groups; // e.g. ~9
    int<lower=1, upper=n_groups> group_id[n_individuals]; // The integer group IDs of each indivudal
    int<lower=1, upper=n_odorants> ranks[n_individuals, n_odorants]; // The matrix of ranks
    }
    
// These are the values for the model to estimate
parameters {
    vector[n_odorants] mu_global; // The global valence for each odorant
    matrix[n_groups, n_odorants] mu_group; // The group-level valences for each odorant
    matrix[n_individuals, n_odorants] mu_ind; // The individual-level valences for each odorant
    real<lower=0> sigma_global; // The variability of the distribution from which global valences are drawn
    real<lower=0> sigma_group; // The variability of the conditional distribution from which group valences are drawn
    //real<lower=0> sigma_ind; // The variability of the conditional distribution from which individual valences are drawn
    real<lower=0> sigma_ind[n_groups]; // The variability of the conditional distribution from which individual valences are drawn
    }
    
// The model also indirectly estimates this, but it is only used to center the valences
transformed parameters {
    real mu_mean = mean(mu_global);
}
    
model {
    mu_mean ~ normal(0, 0.001); // Enforce that the mean (latent) global valence is 0.  Required for identifiability.
    mu_global ~ normal(0, sigma_global); // Global odorant valences have a very weak prior
    sigma_global ~ cauchy(0, 3); // Group level variance has a very weak prior
    sigma_group ~ cauchy(0, 3); // Group level variance has a very weak prior
    sigma_ind ~ cauchy(0, 3);
    for (i in 1:n_groups) {
        mu_group[i] ~ normal(mu_global, sigma_group); // Group odorant valences will be distributed around the global values.
    }
    for (i in 1:n_individuals) {
        //mu_ind[i] ~ normal(mu_group[group_id[i]], sigma_ind); // Individual odorant valences will be distributed around the group values.
        mu_ind[i] ~ normal(mu_group[group_id[i]], sigma_ind[group_id[i]]); // Individual odorant valences will be distributed around the group values.
        target += log_p_ranks(mu_ind[i], ranks[i], n_odorants); // The data likelihood (using Plackett-Luce model)
        }
    }
