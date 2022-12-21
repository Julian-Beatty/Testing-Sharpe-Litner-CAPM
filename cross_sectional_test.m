%%------------------------------------------------------------Appendendix----------------------------------------------------------------------------%%%%%
%%HML-porfolio mimicking the excess returns of high book-market stocks Tx1
%%SMB-portfolio mimicking the excess returns of small cap stocks Tx1
%%Market_return-returns of the Market, the SPY Tx1
%%Returns matrix- contains the raw returns of 25 stocks, over our time period TxN
%%RF-risk free rate Tx1
%%Factors_matrix- Matrix containing [market_return  SMB HML] Tx3

%%------------------------------------------------------------Objective----------------------------------------------------------------------------%%%%%
%%We want to test the validity of the Sharpe-Litner CAPM. The most straightfoward way to test this to do a cross sectional test and 
%%and show alpha=0.
%%------------------------------------------------------------Preliminary calculations----------------------------------------------------------------------------%%%%%
returns_matrix=returns_matrix(1:100,:)
RF=RF(1:100)
T=size(returns_matrix,1);
N=size(returns_matrix,2);
for i=1:N
    risk_free(:,i)=RF;
end
global z_i z_m
z_i=returns_matrix-risk_free ; %%converts returns into excess returns
z_m = market_return(1:100);
%%------------------------------------------------------------Time Series regression----------------------------------------------------------------------------%%%%%
%We perform the time series regression using excess returns.Since we have N test assets, we perform N regressions. 
%%Each regression is of the form z_i=alpha+beta*z_m+e_i. We use a for loop with fitlm to perform this regression 
%%and store the relevant information in a 25x1 cell.

for i=1:N
    time_series_regression{i,1} =fitlm(z_m,z_i(:,i));
    time_series_coefficients(i,1:2)=transpose(table2array(time_series_regression{i,1}.Coefficients(:,1))); %% (contains alpha and beta coeffecients)
    
    alpha_time_series(i,1)=transpose(table2array(time_series_regression{i,1}.Coefficients(1,1)));
    beta_time_series(i,1)=transpose(table2array(time_series_regression{i,1}.Coefficients(2,1)));
    
    alpha_time_series_se(i,1)=transpose(table2array(time_series_regression{i,1}.Coefficients(1,2)))
    beta_time_series_se(i,1)=transpose(table2array(time_series_regression{i,1}.Coefficients(2,2)))

    alpha_time_series_tstat(i,1)=transpose(table2array(time_series_regression{i,1}.Coefficients(1,3)))
    beta_time_series_tstat(i,1)=transpose(table2array(time_series_regression{i,1}.Coefficients(2,3)))
end
%%-----------------------------------------------------------OLS/GSL cross sectional test-----------------------------------------------------------------------------%%%%%
%%We perform OLS and GLS cross-sectional regression: E[R_i]=B_i_a*lambda_a+u
%%We perform OLS cross sectional regression to get Lambda, and its standard error (se)
E_r_i=(mean(z_i))' %%25x1 vector with each row containing the average returns of a portfolio across Time.
OLS_cross_sectional = fitlm(beta_time_series,E_r_i,Intercept=false)
OLS_cross_sectional_lambda=table2array(OLS_cross_sectional.Coefficients(1,1))
OLS_cross_sectional_lambda_se=table2array(OLS_cross_sectional.Coefficients(1,2))

%%compute covariance matrix of residuals
for i=1:T
    OLS_time_series_residual_t(:,1)=transpose(z_i(i,:))-alpha_time_series-beta_time_series*z_m(i,1)
    OLS_time_series_stack_residuals(:,:,i)=(OLS_time_series_residual_t)*(OLS_time_series_residual_t)'
end
OLS_time_series_cov_residuals=(1/T)*sum(OLS_time_series_stack_residuals,3)

%%use covariance matrix of residuals to calculus the covariance matrix of the alphas, then use to calculate the Chi square stat
OLS_cross_sectional_cov_alpha_hat=(1/T)*(eye(25)-beta_time_series*inv(beta_time_series'*beta_time_series)*beta_time_series')*OLS_time_series_cov_residuals*(eye(25)-beta_time_series*inv(beta_time_series'*beta_time_series)')
OLS_cross_sectional_alpha_chi_squared=alpha_time_series'*inv(OLS_cross_sectional_cov_alpha_hat)*alpha_time_series
%%Massive chi squared statistic. P value = 0. Which means we reject the null that all alphas are jointly zero. Evidence against SL capm.


%%-----------------------------------------------------------GLS Part 2-----------------------------------------------------------------%%%
%%We perform GLS cross sectional regression. We compute covariance-variance matrix of the residuals from the time series OLS and
%%use that to compute GLS estimates using kris formula slide 62/118.


gls_cross_sectional_lambda=inv((beta_time_series)'*(inv(OLS_time_series_cov_residuals))*(beta_time_series))*(beta_time_series)'*inv(OLS_time_series_cov_residuals)*(E_r_i)

for i=1:N
    gls_cross_sectional_alpha(i,1)=(E_r_i(i))-(gls_cross_sectional_lambda)*beta_time_series(i);
end

gls_cross_sectional_lambda_se=(1/T)*(inv((beta_time_series'*inv(OLS_time_series_cov_residuals)*beta_time_series))+var(z_m))
gls_cross_sectional_cov_alpha_hat=(1/T)*(OLS_time_series_cov_residuals-beta_time_series*inv(beta_time_series'*inv(OLS_time_series_cov_residuals)*beta_time_series)*beta_time_series')
gls_cross_sectional_alpha_chi_squared=T*gls_cross_sectional_alpha'*inv(OLS_time_series_cov_residuals)*gls_cross_sectional_alpha
%%Chi squared statistic is massive. P value of zero. This is saying that all of our Alphas are jointly not equal to zero.
%%-----------------------------------------------------------Conclusion-----------------------------------------------------------------%%%
%%based on this sample period, we reject the SL CAPM.