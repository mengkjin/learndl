# param recommendation for models

## day gru model

- **labels** std_lag1_10 better than rtn_lag1_10, but quite similar
- **global2top** true to improve top performance
- **criterion.weight** 'polar' to improve top performance
- set enc_in to true can slightly improve results, but set enc_att to true will dampen them (overfit possible)

## 15m resnet gru model

- to predict rtn_lag1_10 is better than predicting std_lag1_10 in overall return and stability, slightly lower IC
- enc_dim 16 is better than enc_dim 64