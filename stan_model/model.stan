
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
        real<lower=0> sigma;
    }
    model {
        mu ~ normal(0, 10);
        rho ~ normal(0, 1);
        beta ~ normal(0, rho);
        sigma ~ normal(0, 10);

        y ~ student_t(10, mu + (X * beta), sigma);
    }
    generated quantities {
        real y_tilde[N] = student_t_rng(10, mu + (X * beta), sigma);
    }
    