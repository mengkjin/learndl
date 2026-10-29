# param recommendation for models

## day gru model

- to predict rtn_lag1_10 / std_lag1_10 is indifferent in result
- set criterion.weight to true to improve top performance
- set enc_in to true can slightly improve results, but set enc_att to true will dampen them (overfit possible)
- global2top waits testing
