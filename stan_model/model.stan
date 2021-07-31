
    data {
        int N;  // Number of observations
        int p;  // Number of variables
        vector[N] y;
        matrix[N, p] X;
    }
    parameters {
        real mu;
        real<lower=0> rho;
        vector[p] beta;
        real theta;
        real<lower=0> sigma;
    }
    transformed parameters {
        real epsilon[N];
        real nu[N];
        
        epsilon[1] = y[1] - (mu + X[1] * beta);
        for (t in 2:N) {
            epsilon[t] = y[t] - (mu + (X[t] * beta) + theta * epsilon[t-1]);
        }
        
        nu[1] = mu + X[1] * beta;
        for (t in 2:N) {
            nu[t] = mu + X[t] * beta + theta * epsilon[t-1];
        }
    }
    model {
        mu ~ normal(0, 10);
        rho ~ normal(0, 1);
        beta ~ normal(0, rho);
        theta ~ normal(0, 2);
        sigma ~ normal(0, 10);
        
        y ~ student_t(5, nu, sigma);
    }
    generated quantities {
        real y_tilde[N];
        y_tilde = student_t_rng(5, nu, sigma);
    }
    